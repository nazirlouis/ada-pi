"""Unit tests for backend/tools.d/kanban.py (absorbs ada_board_write).

httpx is faked — no network. Covers happy paths, error honesty, the
respond owner-gate, the move authority matrix + confirm gate, and
manifest/alias wiring.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from backend import tools_loader
from backend.tool_runner import common as tr_common

TOOL_PATH = (Path(__file__).resolve().parent.parent
             / "backend" / "tools.d" / "kanban.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "tools_d.kanban_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


board = _load_module()


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class FakeClient:
    """httpx.AsyncClient stand-in recording requests; keyed canned replies."""

    def __init__(self, routes=None, fail=None, *a, **kw):
        self.routes = routes or {}
        self.fail = fail
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self.fail is not None:
            raise self.fail
        for key, resp in self.routes.items():
            if url.endswith(key):
                return resp
        return FakeResp(404, {"error": "not found"})

    def posts(self, suffix):
        return [kw for m, u, kw in self.calls
                if m == "POST" and u.endswith(suffix)]


def _runner(owner=True, secondary=False, confirm_ok=False):
    """ToolRunner stand-in: policy_identity/policy_for drive the owner
    gate; _is_secondary_turn the write split; _require_confirmation
    emulates the real gate (denies unless confirmed/known token)."""
    r = MagicMock()
    r.policy_identity.return_value = "person.tony"
    r.banks.policy_for.return_value = {"full": True} if owner else {}
    r._is_secondary_turn.return_value = secondary

    def gate(name, args, confirmed, token, instruction):
        if confirmed is True or token == "good-token":
            return None
        raise PermissionError(
            f"{instruction} TO EXECUTE: replay the SAME call adding "
            "confirm_token='cfm-test' (single use, expires in 120s) — "
            "or resend the identical call with confirmed=true.")

    r._require_confirmation.side_effect = gate
    return r


def _client(routes=None, fail=None):
    def factory(*a, **kw):
        return FakeClient(routes=routes, fail=fail)
    return factory


class ListTest(unittest.IsolatedAsyncioTestCase):

    PAYLOAD = {
        "columns": [{"id": "backlog"}, {"id": "doing"}, {"id": "done"}],
        "cards": [
            {"id": "a", "title": "card A", "column": "doing",
             "updated": "2026-10-04 10:00",
             "requests": [{"id": "r1", "status": "open"}],
             "comms": [{"from": "devin", "text": "working on it"}]},
            {"id": "b", "title": "card B", "column": "backlog",
             "updated": "2026-10-04 12:00"},
            {"id": "c", "title": "card C", "column": "done",
             "updated": "2026-10-03 09:00",
             "requests": [{"id": "r2", "status": "answered"}]},
        ],
    }

    async def test_list_compacts_board(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="list")
        self.assertTrue(out["ok"])
        self.assertEqual(out["columns"], {"doing": 1, "backlog": 1,
                                         "done": 1})
        self.assertEqual(out["open_requests"], 1)
        self.assertEqual(out["total"], 3)
        # sorted by updated desc
        self.assertEqual([c["id"] for c in out["cards"]], ["b", "a", "c"])
        a = next(c for c in out["cards"] if c["id"] == "a")
        self.assertEqual(a["open_requests"], 1)
        self.assertEqual(a["requests"], [{"id": "r1", "ask": ""}])
        self.assertEqual(a["last_comm"], "devin: working on it")
        b = next(c for c in out["cards"] if c["id"] == "b")
        self.assertNotIn("open_requests", b)
        self.assertNotIn("requests", b)
        c_ = next(c for c in out["cards"] if c["id"] == "c")
        self.assertNotIn("requests", c_)  # answered requests hidden

    async def test_list_column_filter_and_limit(self):
        payload = dict(self.PAYLOAD)
        client = FakeClient({"/cards": FakeResp(200, payload)})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="list",
                                  column="doing", limit=5)
        self.assertEqual([c["id"] for c in out["cards"]], ["a"])
        self.assertEqual(out["total"], 1)
        # columns counts are still the whole board
        self.assertEqual(out["columns"]["backlog"], 1)

    async def test_default_action_is_list(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner())
        self.assertTrue(out["ok"])
        self.assertIn("columns", out)

    async def test_list_unreachable_is_honest(self):
        client = FakeClient(fail=httpx.ConnectError("refused"))
        runner = _runner()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(runner, action="list")
        self.assertFalse(out["ok"])
        self.assertIn("couldn't reach", out["error"])
        # transport outage -> one ops event via the context facade
        emit = runner.context.emit_ops_event
        emit.assert_called_once()
        self.assertEqual(emit.call_args.args[0],
                         "kanban_board_unreachable")
        self.assertEqual(emit.call_args.kwargs["tool"], "kanban")


class ReadCardTest(unittest.IsolatedAsyncioTestCase):

    PAYLOAD = {
        "cards": [
            {"id": "probe", "title": "probe card", "column": "review",
             "priority": "medium", "review_kind": "verify",
             "tags": ["kanban"], "note": "probe note",
             "spec": "x" * 600,
             "requests": [{"id": "r1", "status": "open", "ask": "ok?"}],
             "comms": [{"at": f"t{i}", "from": "ada",
                        "text": f"line {i}"} for i in range(15)]},
        ],
    }

    async def test_read_full_card(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read", id="probe")
        self.assertTrue(out["ok"])
        card = out["card"]
        self.assertEqual(card["id"], "probe")
        self.assertEqual(card["priority"], "medium")
        self.assertEqual(card["review_kind"], "verify")
        self.assertEqual(card["tags"], ["kanban"])
        self.assertEqual(len(card["comms"]), board._COMMS_KEEP)
        self.assertEqual(card["comms_earlier"], 5)
        self.assertEqual(len(card["spec"]), 400)

    async def test_read_unknown_card(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read", id="nope")
        self.assertFalse(out["ok"])
        self.assertIn("no card", out["error"])


class ReportLookupTest(unittest.IsolatedAsyncioTestCase):
    """The opinion-loop lookup: action=read report=<slug> finds the
    card whose report: field links a CMS page. Open cards outrank
    done; among open ties the most recently updated wins."""

    PAYLOAD = {
        "cards": [
            {"id": "open-new", "title": "newer open", "column": "doing",
             "report": "dev-kanban", "updated": "2026-10-05 09:00"},
            {"id": "open-old", "title": "older open", "column": "backlog",
             "report": "dev-kanban", "updated": "2026-10-01 09:00"},
            {"id": "done-card", "title": "finished", "column": "done",
             "report": "dev-kanban", "updated": "2026-10-06 09:00"},
            {"id": "done-only", "title": "done only", "column": "done",
             "report": "flood-report", "updated": "2026-10-02 09:00"},
            {"id": "plain", "title": "no report field",
             "column": "doing", "updated": "2026-10-06 10:00"},
        ],
    }

    def test_match_prefers_open_and_picks_most_recent(self):
        m = board._match_report_card(self.PAYLOAD["cards"], "dev-kanban")
        self.assertEqual([c["id"] for c in m["open"]],
                         ["open-new", "open-old"])
        self.assertEqual([c["id"] for c in m["done"]], ["done-card"])

    def test_match_normalizes_slug(self):
        m = board._match_report_card(self.PAYLOAD["cards"],
                                     "  Dev-Kanban ")
        self.assertEqual(m["open"][0]["id"], "open-new")
        m = board._match_report_card(
            [{"id": "c", "report": " Flood-Report ", "column": "todo"}],
            "flood-report")
        self.assertEqual(m["open"][0]["id"], "c")

    def test_match_none_and_done_only(self):
        m = board._match_report_card(self.PAYLOAD["cards"], "no-such")
        self.assertEqual(m["open"], [])
        self.assertEqual(m["done"], [])
        m = board._match_report_card(self.PAYLOAD["cards"], "flood-report")
        self.assertEqual(m["open"], [])
        self.assertEqual([c["id"] for c in m["done"]], ["done-only"])

    async def test_read_report_returns_best_open_card(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read",
                                  report="dev-kanban")
        self.assertTrue(out["ok"])
        self.assertEqual(out["report"], "dev-kanban")
        self.assertEqual(out["card"]["id"], "open-new")
        self.assertEqual(out["card"]["report"], "dev-kanban")
        self.assertEqual(out["open_matches"], 2)
        self.assertEqual(out["done_matches"], 1)
        self.assertEqual(out["also"], ["open-old"])
        self.assertIn("several open cards", out["note"])

    async def test_read_report_done_only_surfaces_done_card(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read",
                                  report="flood-report")
        self.assertEqual(out["card"]["id"], "done-only")
        self.assertEqual(out["card"]["column"], "done")
        self.assertEqual(out["open_matches"], 0)
        self.assertIn("done", out["note"])
        self.assertIn("file", out["note"])

    async def test_read_report_no_match_offers_filing(self):
        client = FakeClient({"/cards": FakeResp(200, dict(self.PAYLOAD))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read",
                                  report="no-such-report")
        self.assertIsNone(out["card"])
        self.assertEqual(out["open_matches"], 0)
        self.assertIn("file", out["note"])


class CommentTest(unittest.IsolatedAsyncioTestCase):

    async def test_comment_posts_as_ada(self):
        client = FakeClient({"/comment": FakeResp(200, {"ok": True,
                                                      "message": "comment added"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="comment",
                                  id="kanban-selftest", text="hello board")
        self.assertTrue(out["ok"])
        method, url, kw = client.calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/apps/board-api/comment"))
        self.assertEqual(kw["json"]["from"], "ada")
        self.assertEqual(kw["json"]["id"], "kanban-selftest")
        self.assertEqual(kw["json"]["text"], "hello board")

    async def test_comment_requires_text(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(), action="comment",
                                  id="kanban-selftest", text="  ")
        self.assertFalse(out["ok"])
        self.assertIn("text", out["error"])

    async def test_comment_rejects_bad_card_id(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(), action="comment",
                                  id="../etc/passwd", text="x")
        self.assertFalse(out["ok"])

    async def test_comment_surfaces_server_error(self):
        client = FakeClient({"/comment": FakeResp(400, {"error": "no card nope"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="comment",
                                  id="nope", text="x")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "no card nope")

    async def test_unreachable_is_honest(self):
        client = FakeClient(fail=httpx.ConnectError("refused"))
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="comment",
                                  id="kanban-selftest", text="x")
        self.assertFalse(out["ok"])
        self.assertIn("couldn't reach", out["error"])


class FileTest(unittest.IsolatedAsyncioTestCase):

    def _routes(self, post_payload, cards):
        return {"/card": FakeResp(200, post_payload),
                "/cards": FakeResp(200, {"cards": cards})}

    async def test_file_creates_backlog_card_as_ada(self):
        client = FakeClient(self._routes(
            {"ok": True, "message": "card p1 created"},
            [{"id": "p1", "title": "probe card", "column": "backlog"}]))
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="file",
                                  title="probe card", note="one liner",
                                  priority="high")
        self.assertTrue(out["ok"])
        self.assertEqual(out["id"], "p1")
        self.assertNotIn("warning", out)
        body = client.posts("/card")[0]["json"]
        self.assertEqual(body["from"], "ada")
        self.assertEqual(body["title"], "probe card")
        self.assertEqual(body["priority"], "high")

    async def test_file_ok_but_no_card_landed_is_honest_failure(self):
        """Phantom guard (card ada-phantom-card-claims): a 200 that
        filed nothing must NOT return ok — the narration would claim a
        card that silently drops the ask."""
        client = FakeClient(self._routes(
            {"ok": True, "message": "card created"},
            [{"id": "other", "title": "unrelated", "column": "backlog"}]))
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="file",
                                  title="missing card")
        self.assertFalse(out["ok"])
        self.assertIn("did not file", out["error"])

    async def test_file_unverifiable_read_warns_but_stays_ok(self):
        """Write succeeded; the verify read failed — don't flip a real
        write to failure, but mark it unverified."""
        client = FakeClient({"/card": FakeResp(
            200, {"ok": True, "message": "card p2 created"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="file", title="t")
        self.assertTrue(out["ok"])
        self.assertIn("warning", out)
        self.assertEqual(out["id"], "p2")

    async def test_file_verifies_by_title_when_id_absent(self):
        client = FakeClient(self._routes(
            {"ok": True, "message": "done"},
            [{"id": "real-id", "title": "probe card",
              "column": "backlog"}]))
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="file",
                                  title="probe card")
        self.assertTrue(out["ok"])
        self.assertEqual(out["id"], "real-id")

    async def test_file_requires_title(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(), action="file", title=" ")
        self.assertFalse(out["ok"])

    async def test_file_rejects_bad_priority(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(), action="file",
                                  title="x", priority="urgent")
        self.assertFalse(out["ok"])
        self.assertIn("priority", out["error"])

    async def test_create_synonym_files(self):
        """ada_board_write's action='create' still lands on file."""
        client = FakeClient({"/card": FakeResp(200, {"ok": True,
                                                    "message": "created"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="create", title="t")
        self.assertTrue(out["ok"])
        self.assertTrue(client.posts("/card"))


class AskTest(unittest.IsolatedAsyncioTestCase):

    async def test_ask_posts_request_as_ada(self):
        client = FakeClient({"/request": FakeResp(200, {
            "ok": True, "message": "request r1 raised"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="ask",
                                  id="probe", ask="can I close this?")
        self.assertTrue(out["ok"])
        body = client.posts("/request")[0]["json"]
        self.assertEqual(body["id"], "probe")
        self.assertEqual(body["from"], "ada")
        self.assertEqual(body["ask"], "can I close this?")

    async def test_ask_requires_ask_text(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(), action="ask", id="p1")
        self.assertFalse(out["ok"])
        self.assertIn("ask", out["error"])

    async def test_ask_passes_options_and_suggestion(self):
        client = FakeClient({"/request": FakeResp(200, {"ok": True})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            await board.run(_runner(), action="ask", id="p1",
                            ask="pick one", options=["A", "B"],
                            suggested="A")
        body = client.posts("/request")[0]["json"]
        # options are normalized to {label: ...} dicts (cbce30d — the union
        # schema with bare strings was killing all voice connects)
        self.assertEqual(body["options"], [{"label": "A"}, {"label": "B"}])
        self.assertEqual(body["suggested"], "A")


class RespondTest(unittest.IsolatedAsyncioTestCase):

    async def test_respond_owner_succeeds(self):
        client = FakeClient({"/respond": FakeResp(200, {"ok": True,
                                                      "message": "answer saved"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(owner=True), action="respond",
                                  id="c1", request_id="r1", answer="do A")
        self.assertTrue(out["ok"])
        self.assertIn("note", out)  # tony-attribution caveat surfaced
        method, url, kw = client.calls[0]
        self.assertEqual(kw["json"]["request_id"], "r1")
        self.assertEqual(kw["json"]["answer"], "do A")

    async def test_respond_non_owner_denied(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(owner=False), action="respond",
                                  id="c1", request_id="r1", answer="x")
        self.assertFalse(out["ok"])
        self.assertIn("owner", out["error"])

    async def test_respond_requires_request_id_and_answer(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            out = await board.run(_runner(), action="respond", id="c1",
                                  request_id="", answer="x")
            self.assertFalse(out["ok"])
            out = await board.run(_runner(), action="respond", id="c1",
                                  request_id="r1", answer=" ")
            self.assertFalse(out["ok"])


def _board(cards, doing_limit=2):
    return {
        "columns": [{"id": c}
                    for c in ("lab", "backlog", "doing", "review", "done")],
        "doing_limit": doing_limit,
        "cards": cards,
    }


class MoveTest(unittest.IsolatedAsyncioTestCase):
    """The §3 authority matrix: free triage moves run; evidence gates
    review->done; Tony-decides moves hit the runner confirm gate."""

    CARDS = [
        {"id": "idle", "title": "backlog card", "column": "backlog"},
        {"id": "wip", "title": "in flight", "column": "doing"},
        {"id": "rev", "title": "awaiting review", "column": "review",
         "priority": "medium"},
        {"id": "hot", "title": "hot card", "column": "review",
         "priority": "high"},
        {"id": "dec", "title": "decision card", "column": "review",
         "review_kind": "decide"},
        {"id": "sec", "title": "security card", "column": "review",
         "tags": ["security"]},
        {"id": "gone", "title": "done card", "column": "done"},
    ]

    def _client(self, cards=None):
        routes = {
            "/cards": FakeResp(200, _board(list(cards or self.CARDS))),
            "/action": FakeResp(200, {"ok": True, "message": "moved to x"}),
            "/comment": FakeResp(200, {"ok": True,
                                      "message": "comment added"}),
        }
        return FakeClient(routes)

    async def test_backlog_to_doing_free(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="idle", column="doing")
        self.assertTrue(out["ok"], out)
        self.assertEqual(client.posts("/action")[0]["json"],
                         {"id": "idle", "do": "move", "column": "doing"})
        # audit comment as ada
        self.assertEqual(client.posts("/comment")[0]["json"]["from"], "ada")

    async def test_doing_cap_gated(self):
        cards = self.CARDS + [{"id": "wip2", "column": "doing"}]
        client = self._client(cards)
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="idle", column="doing")
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("needs_confirm"))
        self.assertIn("limit", out["error"])
        self.assertFalse(client.posts("/action"))  # nothing moved

    async def test_doing_cap_confirmed_executes(self):
        cards = self.CARDS + [{"id": "wip2", "column": "doing"}]
        client = self._client(cards)
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="idle", column="doing",
                                  confirmed=True)
        self.assertTrue(out["ok"], out)
        self.assertTrue(client.posts("/action"))
        # the audit line records the override
        self.assertIn("confirmed",
                      client.posts("/comment")[0]["json"]["text"])

    async def test_doing_to_backlog_and_review_free(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            for col in ("backlog", "review"):
                out = await board.run(_runner(), action="move",
                                      id="wip", column=col)
                self.assertTrue(out["ok"], (col, out))

    async def test_review_to_done_needs_evidence(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="rev", column="done")
        self.assertFalse(out["ok"])
        self.assertNotIn("needs_confirm", out)  # hard rule, not confirmable
        self.assertIn("evidence", out["error"])
        self.assertFalse(client.posts("/action"))

    async def test_review_to_done_with_evidence(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="rev", column="done",
                                  evidence="service is-active chaba-board-api")
        self.assertTrue(out["ok"], out)
        audit = client.posts("/comment")[0]["json"]["text"]
        self.assertIn("verified:", audit)
        self.assertIn("service is-active", audit)

    async def test_review_to_done_high_priority_gated(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="hot", column="done",
                                  evidence="checked")
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("needs_confirm"))
        self.assertIn("priority:high", out["error"])
        self.assertFalse(client.posts("/action"))

    async def test_review_to_done_decide_card_gated(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="dec", column="done",
                                  evidence="checked")
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("needs_confirm"))
        self.assertIn("decide", out["error"])

    async def test_review_to_done_prodsec_tag_gated(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="sec", column="done",
                                  evidence="checked")
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("needs_confirm"))

    async def test_gated_move_confirmed_executes(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="hot", column="done",
                                  evidence="checked", confirmed=True)
        self.assertTrue(out["ok"], out)
        self.assertTrue(client.posts("/action"))

    async def test_done_reopen_gated(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="gone", column="backlog")
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("needs_confirm"))

    async def test_move_unknown_column_and_card(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="idle", column="sideways")
            self.assertFalse(out["ok"])
            self.assertIn("column", out["error"])
            out = await board.run(_runner(), action="move",
                                  id="ghost", column="doing")
            self.assertFalse(out["ok"])
            self.assertIn("no card", out["error"])

    async def test_move_same_column_noop(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="idle", column="backlog")
        self.assertTrue(out["ok"])
        self.assertFalse(client.posts("/action"))

    async def test_move_server_error_surfaces(self):
        client = FakeClient({
            "/cards": FakeResp(200, _board(self.CARDS)),
            "/action": FakeResp(400, {"error": "merge guard blocked"}),
        })
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move",
                                  id="idle", column="doing")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "merge guard blocked")


class SecondaryTurnTest(unittest.IsolatedAsyncioTestCase):
    """Design §6: board ops are Tony-tier — an identified non-owner
    voice gets list/read only."""

    async def test_secondary_can_list(self):
        client = FakeClient({"/cards": FakeResp(200, {"cards": []})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(secondary=True), action="list")
        self.assertTrue(out["ok"])

    async def test_secondary_writes_denied(self):
        with patch.object(board.httpx, "AsyncClient",
                          side_effect=AssertionError("no http expected")):
            for action in ("comment", "file", "ask", "respond", "move"):
                out = await board.run(
                    _runner(secondary=True), action=action,
                    id="c1", text="x", column="doing")
                self.assertFalse(out["ok"], action)
                self.assertIn("owner", out["error"], action)


class MiscTest(unittest.IsolatedAsyncioTestCase):

    async def test_unknown_action(self):
        client = FakeClient()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="explode")
        self.assertFalse(out["ok"])
        self.assertIn("unknown action", out["error"])

    async def test_base_url_env_override(self):
        client = FakeClient({"/comment": FakeResp(200, {"ok": True})})
        with patch.object(board.httpx, "AsyncClient", return_value=client), \
                patch.dict("os.environ",
                           {"ADA_BOARD_API_URL": "http://127.0.0.1:8787"}):
            out = await board.run(_runner(), action="comment",
                                  id="c", text="x")
        self.assertTrue(out["ok"])
        self.assertEqual(client.calls[0][1],
                         "http://127.0.0.1:8787/comment")


class ManifestWiringTest(unittest.TestCase):

    def test_tool_loads_from_real_manifest(self):
        reg = tools_loader.load()  # real backend/tools.d
        self.assertIn("kanban", reg.tools, reg.errors)
        tool = reg.tools["kanban"]
        self.assertEqual(tool.policy, "read")
        self.assertTrue(tool.secondary_allowed)  # in-module write split
        self.assertEqual(tool.declaration["name"], "kanban")
        self.assertNotIn("ada_board_write", reg.tools)

    def test_absorbed_name_aliases_to_kanban(self):
        self.assertEqual(tr_common._ALIASES.get("ada_board_write"),
                         "kanban")


if __name__ == "__main__":
    unittest.main()


class LooseIdResolutionTest(unittest.IsolatedAsyncioTestCase):
    """The model often shortens a slug-suffixed id — reads/moves resolve
    unique prefixes/titles client-side; blind writes retry after a
    'no card' reply."""

    CARDS = [
        {"id": "kanban-review-probe-0be154", "title": "kanban-review "
         "probe", "column": "backlog"},
        {"id": "kanban-other", "title": "other", "column": "backlog"},
    ]

    async def test_read_resolves_unique_prefix(self):
        client = FakeClient({"/cards": FakeResp(
            200, _board(list(self.CARDS)))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read",
                                  id="kanban-review-probe")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["card"]["id"], "kanban-review-probe-0be154")

    async def test_read_ambiguous_prefix_errors(self):
        client = FakeClient({"/cards": FakeResp(
            200, _board(list(self.CARDS)))})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="read",
                                  id="kanban")
        self.assertFalse(out["ok"])
        self.assertIn("no card", out["error"])

    async def test_ask_retries_with_resolved_id(self):
        client = FakeClient({
            "/request": FakeResp(404, {"error": "no card probe"}),
            "/cards": FakeResp(200, _board(list(self.CARDS))),
        })
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="ask",
                                  id="kanban-review-probe",
                                  ask="can it go?")
        posts = client.posts("/request")
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[1]["json"]["id"],
                         "kanban-review-probe-0be154")
        self.assertFalse(out["ok"])  # retry still 404s — error surfaces

    async def test_move_resolves_prefix(self):
        routes = {
            "/cards": FakeResp(200, _board(list(self.CARDS))),
            "/action": FakeResp(200, {"ok": True, "message": "moved"}),
            "/comment": FakeResp(200, {"ok": True}),
        }
        client = FakeClient(routes)
        # backlog->doing is free but the doing cap is 0 cards here —
        # need a card already in doing to avoid the cap refusal.
        client2 = FakeClient({
            "/cards": FakeResp(200, _board(
                [{"id": "kanban-review-probe-0be154",
                  "title": "probe", "column": "doing"}])),
            "/action": FakeResp(200, {"ok": True, "message": "moved"}),
            "/comment": FakeResp(200, {"ok": True}),
        })
        with patch.object(board.httpx, "AsyncClient",
                          return_value=client2):
            out = await board.run(_runner(), action="move",
                                  id="kanban-review-probe",
                                  column="backlog")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["id"], "kanban-review-probe-0be154")


class StaleGuardTest(unittest.IsolatedAsyncioTestCase):
    """expected_updated= optimistic-concurrency guard
    (report_first_protocol#write_guard): a write must refuse when the
    card's updated stamp moved since the caller's last read."""

    CARDS = [{"id": "x1", "title": "stale probe", "column": "backlog",
              "updated": "2026-10-10 09:00"}]

    def _routes(self, extra=None):
        routes = {"/cards": FakeResp(200, _board(list(self.CARDS)))}
        routes.update(extra or {})
        return FakeClient(routes)

    async def test_comment_stale_refuses_before_write(self):
        client = self._routes({"/comment": FakeResp(200, {"ok": True})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="comment", id="x1",
                                  text="hi",
                                  expected_updated="2026-10-10 08:00")
        self.assertFalse(out["ok"])
        self.assertTrue(out["stale"])
        self.assertEqual(out["updated"], "2026-10-10 09:00")
        self.assertEqual(client.posts("/comment"), [])

    async def test_comment_matching_stamp_writes(self):
        client = self._routes(
            {"/comment": FakeResp(200, {"ok": True, "message": "ok"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="comment", id="x1",
                                  text="hi",
                                  expected_updated="2026-10-10 09:00")
        self.assertTrue(out["ok"], out)
        self.assertEqual(len(client.posts("/comment")), 1)

    async def test_no_expected_updated_writes_unchecked(self):
        client = FakeClient({"/comment": FakeResp(200, {"ok": True,
                                                      "message": "ok"})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="comment", id="x1",
                                  text="hi")
        self.assertTrue(out["ok"], out)
        # no pre-flight /cards read — the guard is opt-in
        self.assertFalse(any(u.endswith("/cards")
                             for m, u, kw in client.calls))

    async def test_move_stale_refuses(self):
        client = self._routes(
            {"/action": FakeResp(200, {"ok": True}),
             "/comment": FakeResp(200, {"ok": True})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="move", id="x1",
                                  column="doing",
                                  expected_updated="2026-10-10 08:00")
        self.assertFalse(out["ok"])
        self.assertTrue(out["stale"])
        self.assertEqual(client.posts("/action"), [])

    async def test_respond_stale_refuses(self):
        client = self._routes({"/respond": FakeResp(200, {"ok": True})})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(owner=True), action="respond",
                                  id="x1", request_id="r1", answer="a",
                                  expected_updated="2026-10-10 08:00")
        self.assertFalse(out["ok"])
        self.assertTrue(out["stale"])
        self.assertEqual(client.posts("/respond"), [])

class NextMenuTest(unittest.IsolatedAsyncioTestCase):
    async def test_next_checks_current_work_and_returns_choices_without_writes(self):
        payload = {
            "cards": [
                {"id": "doing-1", "title": "Current task", "column": "doing",
                 "updated": "2026-11-06 10:00"},
                {"id": "review-1", "title": "Review task", "column": "review",
                 "updated": "2026-11-06 09:00"},
                {"id": "backlog-1", "title": "Safe next task", "column": "backlog",
                 "priority": "high", "updated": "2026-11-06 08:00"},
                {"id": "blocked-1", "title": "Needs decision", "column": "backlog",
                 "requests": [{"id": "r1", "status": "open"}]},
            ]
        }
        client = FakeClient({"/cards": FakeResp(200, payload)})
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next")
        self.assertTrue(out["ok"])
        self.assertEqual(out["current_work"][0]["id"], "doing-1")
        self.assertEqual(out["review_ready"][0]["id"], "review-1")
        self.assertEqual(out["safe_candidates"][0]["id"], "backlog-1")
        self.assertIn("Stop", [x["label"] for x in out["next_menu"]])
        self.assertTrue(out["question_id"].startswith("next-"))
        self.assertGreater(out["expires_at"], 0)
        self.assertEqual(client.posts("/action"), [])

    async def test_next_board_failure_is_honest(self):
        client = FakeClient(fail=httpx.ConnectError("refused"))
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next")
        self.assertFalse(out["ok"])
        self.assertEqual(out["operation"], "next")


class NextChoiceTest(unittest.IsolatedAsyncioTestCase):
    """action=next_choice resolves a menu pick against FRESH board state:
    valid ids return the card + read-only step; stale/unknown ids are
    refused with a fresh menu; nothing is ever written."""

    CARDS = [
        {"id": "doing-1", "title": "Current task", "column": "doing",
         "updated": "2026-11-06 10:00",
         "action": {"type": "devin", "status": "running"}},
        {"id": "review-1", "title": "Review task", "column": "review",
         "updated": "2026-11-06 09:00"},
        {"id": "backlog-1", "title": "Safe next task", "column": "backlog",
         "priority": "high", "updated": "2026-11-06 08:00"},
    ]

    def _client(self, cards=None):
        payload = {"cards": list(self.CARDS if cards is None else cards)}
        return FakeClient({"/cards": FakeResp(200, payload)})

    async def test_plan_choice_returns_full_card_and_guidance(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice",
                                  choice="plan:backlog-1")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["operation"], "next_choice")
        self.assertEqual(out["menu_operation"], "plan")
        self.assertEqual(out["card"]["id"], "backlog-1")
        self.assertIn("do not queue", out["next_safe_action"])
        self.assertEqual(client.posts("/action"), [])
        self.assertEqual(client.posts("/comment"), [])

    async def test_continue_and_review_choices_resolve(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            cont = await board.run(_runner(), action="next_choice",
                                   choice="continue:doing-1")
            rev = await board.run(_runner(), action="next_choice",
                                  choice="review:review-1")
        self.assertTrue(cont["ok"], cont)
        self.assertEqual(cont["card"]["column"], "doing")
        self.assertIn("doing", cont["next_safe_action"])
        self.assertTrue(rev["ok"], rev)
        self.assertEqual(rev["card"]["column"], "review")
        self.assertIn("review", rev["next_safe_action"])

    async def test_stop_choice_needs_no_card(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice",
                                  choice="stop")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["menu_operation"], "stop")
        self.assertIn("stop", out["next_safe_action"])
        self.assertNotIn("card", out)

    async def test_unknown_choice_is_refused_with_fresh_menu(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice",
                                  choice="plan:no-such-card")
        self.assertFalse(out["ok"])
        self.assertTrue(out["stale"])
        self.assertEqual(out["choice"], "plan:no-such-card")
        self.assertTrue(out["next_menu"])
        self.assertTrue(out["question_id"].startswith("next-"))

    async def test_moved_card_drops_out_as_stale(self):
        # backlog-1 was in the old menu but has since moved to done —
        # the fresh ranking no longer offers plan:backlog-1.
        cards = [dict(c) for c in self.CARDS]
        cards[2]["column"] = "done"
        client = self._client(cards)
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice",
                                  choice="plan:backlog-1")
        self.assertFalse(out["ok"])
        self.assertTrue(out["stale"])

    async def test_blocked_card_is_not_a_valid_plan_pick(self):
        # A backlog card with an open request never enters the menu.
        cards = [dict(c) for c in self.CARDS]
        cards[2]["requests"] = [{"id": "r1", "status": "open"}]
        client = self._client(cards)
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice",
                                  choice="plan:backlog-1")
        self.assertFalse(out["ok"])
        self.assertTrue(out["stale"])

    async def test_missing_choice_is_an_honest_error(self):
        client = self._client()
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice")
        self.assertFalse(out["ok"])
        self.assertIn("choice", out["error"])

    async def test_board_failure_is_honest(self):
        client = FakeClient(fail=httpx.ConnectError("refused"))
        with patch.object(board.httpx, "AsyncClient", return_value=client):
            out = await board.run(_runner(), action="next_choice",
                                  choice="stop")
        self.assertFalse(out["ok"])
        self.assertEqual(out["operation"], "next_choice")

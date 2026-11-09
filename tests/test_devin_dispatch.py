import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from backend import devin_dispatch as dd


class _FakeProc:
    def __init__(self, stdout: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, b""

    def kill(self):
        pass

    async def wait(self):
        return self.returncode


def _patch_exec(captured: list, stdout: bytes = b"ok\n"):
    async def fake_exec(*cmd, **kwargs):
        captured.append(cmd)
        return _FakeProc(stdout)

    return patch("asyncio.create_subprocess_exec", fake_exec)


class RunQuotingTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_command_shell_quotes_args(self):
        captured: list = []
        with _patch_exec(captured):
            await dd._run("start", "chaba", "Investigate (sensor.nobito_pm2_5) stuck; check logs")
        cmd = captured[0]
        # ssh args: ssh -o BatchMode=yes -o ConnectTimeout=10 HOST <remote_cmd>
        remote_cmd = cmd[-1]
        self.assertTrue(remote_cmd.startswith(dd.BIN + " start chaba '"))
        self.assertIn("(sensor.nobito_pm2_5)", remote_cmd)
        self.assertIn("'", remote_cmd)
        # BIN keeps its leading ~ unquoted so the remote shell expands it.
        self.assertNotIn("'" + dd.BIN, remote_cmd)


def _patch_board(captured: list | None = None, status: int = 200,
                 data: dict | None = None):
    """Mock board_client.post -> (status, data); captures (path, body)."""
    async def fake_post(path, body, timeout=10.0):
        if captured is not None:
            captured.append((path, body))
        return status, (data if data is not None else {"id": "card-b1"})

    return patch.object(dd.board_client, "post", new=fake_post)


def _no_remote_dupe():
    return patch.object(dd, "_remote_dupe", new=AsyncMock(return_value=None))


class DispatchDedupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        dd._recent_dispatches.clear()

    async def test_similar_retry_reuses_card_id(self):
        with _patch_board(), _no_remote_dupe():
            first = await dd.dispatch(
                "chaba",
                "Investigate why the Nobito PM2.5 sensor reading has been "
                "stuck at 72.0. Check history trends, device availability, "
                "and Home Assistant integration logs.",
            )
        captured: list = []
        with _patch_board(captured), _no_remote_dupe():
            second = await dd.dispatch(
                "chaba",
                "Investigate stuck Nobito PM2.5 sensor reading. "
                "Check history, availability, and logs.",
            )
        self.assertEqual(second["card_id"], first["card_id"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(captured, [])  # no second card filed

    async def test_different_task_still_dispatches(self):
        with _patch_board(data={"id": "card-a"}), _no_remote_dupe():
            await dd.dispatch("chaba", "Investigate stuck Nobito PM2.5 sensor reading.")
        with _patch_board(data={"id": "card-b"}), _no_remote_dupe():
            second = await dd.dispatch("chaba", "Check the Rika RK600 weather station battery.")
        self.assertEqual(second["card_id"], "card-b")
        self.assertNotIn("deduplicated", second)

    def _status_line(self, task_id: str) -> str:
        return (f"{task_id:<42} inactive success    repo=chaba          "
                "transcript=2026-09-24 07:52:21\n")

    async def test_remote_dupe_reuses_recent_task(self):
        # Empty in-process cache (e.g. after a backend restart) — the remote
        # status list still catches a retry of a recently-started task.
        recent = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tid = f"{recent}-investigate-why-the-nobito-pm2"
        with patch.object(dd, "_run", new=AsyncMock(return_value=self._status_line(tid))):
            res = await dd.dispatch(
                "chaba", "Investigate stuck Nobito PM2.5 sensor reading.")
        self.assertEqual(res["task_id"], tid)
        self.assertTrue(res["deduplicated"])

    async def test_remote_dupe_ignores_old_task(self):
        old = (datetime.now(timezone.utc)
               - timedelta(seconds=dd.DEDUP_WINDOW_S + 60)
               ).strftime("%Y%m%d-%H%M%S")
        tid = f"{old}-investigate-why-the-nobito-pm2"
        with patch.object(dd, "_run", new=AsyncMock(
                return_value=self._status_line(tid))):
            with _patch_board(data={"id": "card-new"}):
                res = await dd.dispatch(
                    "chaba", "Investigate stuck Nobito PM2.5 sensor reading.")
        self.assertEqual(res["card_id"], "card-new")
        self.assertNotIn("deduplicated", res)

    async def test_remote_dupe_ignores_other_repo(self):
        recent = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tid = f"{recent}-investigate-why-the-nobito-pm2"
        status_out = self._status_line(tid).replace("repo=chaba", "repo=ada-voice")
        with patch.object(dd, "_run", new=AsyncMock(return_value=status_out)):
            with _patch_board(data={"id": "card-new"}):
                res = await dd.dispatch(
                    "chaba", "Investigate stuck Nobito PM2.5 sensor reading.")
        self.assertEqual(res["card_id"], "card-new")
        self.assertNotIn("deduplicated", res)


class _FakeMddb:
    """Key->doc map for get_document + a record of add_document calls."""

    def __init__(self, docs: dict | None = None, add_ok: bool = True):
        self._docs = dict(docs or {})
        self._add_ok = add_ok
        self.gets: list[tuple[str, str]] = []
        self.added: list[dict] = []

    async def get_document(self, collection, key, lang="en"):
        self.gets.append((collection, key))
        return self._docs.get(key)

    async def add_document(self, collection, key, lang, content_md,
                           meta=None, **kw):
        self.added.append({"collection": collection, "key": key,
                           "content_md": content_md, "meta": meta or {}})
        return {"ok": True} if self._add_ok else None


def _spec_doc(status: str, kind: str = "spec-review") -> dict:
    return {"key": "spec/20260930-125800-a-tool",
            "contentMd": "# Dev-team review\n",
            "meta": {"kind": [kind], "status": [status],
                     "bank": ["devin-handoff"]}}


_SPEC_KEY = "spec/20260930-125800-a-tool"


class PlaybookRegistryTests(unittest.TestCase):
    def test_all_playbooks_load_and_validate(self):
        names = dd.list_playbooks()
        self.assertEqual(names, ["build-tool", "cms-regen",
                                 "fix-scenario", "investigate"])
        for name in names:
            pb = dd.load_playbook(name)
            for field in ("params", "task_template", "memory_domains",
                          "gate", "verify", "report_to", "max_runtime_min"):
                self.assertIn(field, pb, f"{name} missing {field}")
            self.assertIn("confirm", pb["gate"])
            self.assertIn("identity", pb["gate"])
            self.assertIsInstance(pb["verify"], list)
            self.assertGreater(pb["max_runtime_min"], 0)

    def test_build_tool_requires_pass_status(self):
        pb = dd.load_playbook("build-tool")
        self.assertEqual((pb.get("requires") or {}).get("spec_status"),
                         "pass")

    def test_unknown_playbook_lists_available(self):
        with self.assertRaises(dd.PlaybookError) as ctx:
            dd.load_playbook("no-such-playbook")
        self.assertIn("build-tool", str(ctx.exception))

    def test_bad_playbook_name_rejected(self):
        for bad in ("../x", "a/b", "", ".."):
            with self.assertRaises(dd.PlaybookError):
                dd.load_playbook(bad)


class PlaybookDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        dd._recent_dispatches.clear()

    async def test_build_tool_refuses_missing_spec(self):
        mddb = _FakeMddb()
        with _patch_exec([], stdout=b"should-not-run\n") as _:
            res = await dd.dispatch(
                "ada-voice", "build the daily-status tool",
                playbook="build-tool",
                params={"spec_key": _SPEC_KEY},
                mddb=mddb, caller="person.tony", is_owner=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["gate"], "spec_status")
        self.assertEqual(res["offer"], "ada_devteam_review")
        self.assertIn(_SPEC_KEY, res["error"])
        self.assertEqual(mddb.added, [])

    async def test_build_tool_refuses_blocked_spec(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("block")})
        res = await dd.dispatch(
            "ada-voice", "build it", playbook="build-tool",
            params={"spec_key": _SPEC_KEY},
            mddb=mddb, caller="person.tony", is_owner=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["spec_status"], "block")
        self.assertIn("block", res["error"])
        self.assertIn("ada_devteam_review", res["error"])

    async def test_build_tool_refuses_wrong_kind(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("pass", kind="note")})
        res = await dd.dispatch(
            "ada-voice", "build it", playbook="build-tool",
            params={"spec_key": _SPEC_KEY},
            mddb=mddb, caller="person.tony", is_owner=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["gate"], "spec_status")

    async def test_build_tool_refuses_without_mddb(self):
        # Fail closed: the gate cannot be verified without the bank.
        res = await dd.dispatch(
            "ada-voice", "build it", playbook="build-tool",
            params={"spec_key": _SPEC_KEY},
            mddb=None, caller="person.tony", is_owner=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["gate"], "spec_status")

    async def test_build_tool_owner_gate(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("pass")})
        res = await dd.dispatch(
            "ada-voice", "build it", playbook="build-tool",
            params={"spec_key": _SPEC_KEY},
            mddb=mddb, caller="person.kk", is_owner=False)
        self.assertFalse(res["ok"])
        self.assertEqual(res["gate"], "identity")
        self.assertEqual(mddb.gets, [])  # refused before touching the bank

    async def test_build_tool_pass_dispatches_and_stamps(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("pass")})
        captured: list = []
        with _patch_exec(captured, stdout=b"20261005-180000-build-a-tool\n"):
            res = await dd.dispatch(
                "ada-voice", "the daily status thing Tony asked for",
                playbook="build-tool",
                params={"spec_key": _SPEC_KEY, "tool_name": "ada_x"},
                mddb=mddb, caller="person.tony", is_owner=True)
        self.assertEqual(res["task_id"], "20261005-180000-build-a-tool")
        self.assertEqual(res["playbook"], "build-tool")
        self.assertEqual(res["report_to"], "ada-ha-bank-devin-handoff")
        self.assertTrue(res["verify"])
        self.assertEqual(res["job_contract"],
                         "job/20261005-180000-build-a-tool/contract")
        # Rendered task carries the spec ref, the override line, the
        # contract trailer, and the request context. (captured[0] is the
        # dedup 'status' probe; find the 'start' call.)
        task_text = next(c[-1] for c in captured if " start " in c[-1])
        self.assertIn(_SPEC_KEY, task_text)
        self.assertIn("Tool name override: ada_x", task_text)
        self.assertIn("python3 scripts/tool-lint.py", task_text)
        self.assertIn("Request context:", task_text)
        # Ledger stamp: job/<id>/contract with playbook + verify meta.
        stamp = mddb.added[0]
        self.assertEqual(stamp["collection"], "ada-ha-bank-devin-handoff")
        self.assertEqual(stamp["key"],
                         "job/20261005-180000-build-a-tool/contract")
        self.assertEqual(stamp["meta"]["kind"], ["job-contract"])
        self.assertEqual(stamp["meta"]["playbook"], ["build-tool"])
        self.assertIn("python3 scripts/tool-lint.py",
                      stamp["meta"]["verify"])
        self.assertEqual(stamp["meta"]["spec_key"], [_SPEC_KEY])

    async def test_build_tool_bare_slug_resolves(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("pass")})
        with _patch_exec([], stdout=b"t-1\n"):
            res = await dd.dispatch(
                "ada-voice", "build it", playbook="build-tool",
                params={"spec_key": "20260930-125800-a-tool"},
                mddb=mddb, caller="person.tony", is_owner=True)
        self.assertEqual(res["task_id"], "t-1")
        # First lookup misses the bare slug, second hits spec/<slug>.
        self.assertEqual(mddb.gets[0][1], "20260930-125800-a-tool")
        self.assertEqual(mddb.gets[1][1], _SPEC_KEY)
        self.assertEqual(mddb.added[0]["meta"]["spec_key"], [_SPEC_KEY])

    async def test_missing_required_param_refused(self):
        res = await dd.dispatch(
            "ada-voice", "build it", playbook="build-tool",
            params={}, mddb=_FakeMddb(), caller="person.tony",
            is_owner=True)
        self.assertFalse(res["ok"])
        self.assertIn("spec_key", res["error"])

    async def test_unknown_param_refused(self):
        res = await dd.dispatch(
            "ada-voice", "build it", playbook="build-tool",
            params={"spec_key": _SPEC_KEY, "bogus": "x"},
            mddb=_FakeMddb(), caller="person.tony", is_owner=True)
        self.assertFalse(res["ok"])
        self.assertIn("bogus", res["error"])

    async def test_stamp_failure_is_nonfatal(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("pass")}, add_ok=False)
        with _patch_exec([], stdout=b"t-2\n"):
            res = await dd.dispatch(
                "ada-voice", "build it", playbook="build-tool",
                params={"spec_key": _SPEC_KEY},
                mddb=mddb, caller="person.tony", is_owner=True)
        self.assertEqual(res["task_id"], "t-2")
        self.assertNotIn("job_contract", res)
        self.assertIn("stamp skipped", res["note"])

    async def test_playbook_dedup_on_rendered_task(self):
        mddb = _FakeMddb({_SPEC_KEY: _spec_doc("pass")})
        with _patch_exec([], stdout=b"t-3\n"):
            first = await dd.dispatch(
                "ada-voice", "build the tool", playbook="build-tool",
                params={"spec_key": _SPEC_KEY},
                mddb=mddb, caller="person.tony", is_owner=True)
        with patch.object(dd, "_run", new=AsyncMock()) as run_mock:
            second = await dd.dispatch(
                "ada-voice", "build the tool", playbook="build-tool",
                params={"spec_key": _SPEC_KEY},
                mddb=mddb, caller="person.tony", is_owner=True)
        self.assertEqual(second["task_id"], first["task_id"])
        self.assertTrue(second["deduplicated"])
        run_mock.assert_not_called()

    async def test_investigate_identity_gate(self):
        # identity=identified: anonymous caller refused, identified passes.
        res = await dd.dispatch(
            "chaba", "why is it stuck", playbook="investigate",
            params={"subject": "the nobito pm2.5 sensor"},
            mddb=_FakeMddb(), caller=None, is_owner=False)
        self.assertFalse(res["ok"])
        self.assertEqual(res["gate"], "identity")
        with _patch_exec([], stdout=b"t-4\n"):
            res = await dd.dispatch(
                "chaba", "why is it stuck", playbook="investigate",
                params={"subject": "the nobito pm2.5 sensor"},
                mddb=_FakeMddb(), caller="person.kk", is_owner=False)
        self.assertEqual(res["task_id"], "t-4")

    async def test_fix_scenario_renders_params(self):
        captured: list = []
        with _patch_exec(captured, stdout=b"t-5\n"):
            res = await dd.dispatch(
                "ada-voice", playbook="fix-scenario",
                params={"scenario": "devin_answer_gate",
                        "symptom": "times out on turn 2"},
                mddb=_FakeMddb(), caller="person.tony", is_owner=True)
        self.assertEqual(res["task_id"], "t-5")
        task_text = next(c[-1] for c in captured if " start " in c[-1])
        self.assertIn("devin_answer_gate", task_text)
        self.assertIn("Reported symptom: times out on turn 2", task_text)

    async def test_plain_dispatch_files_card(self):
        # No playbook -> armed kanban card, not an ssh devin-dispatch start.
        with _patch_board(data={"id": "card-plain"}), _no_remote_dupe():
            res = await dd.dispatch("chaba", "Check the weather station.")
        self.assertTrue(res["ok"])
        self.assertEqual(res["card_id"], "card-plain")
        self.assertNotIn("playbook", res)


class BoardDispatchTests(unittest.IsolatedAsyncioTestCase):
    """No-playbook dispatch files an armed card via board-api."""

    def setUp(self):
        dd._recent_dispatches.clear()

    async def test_card_payload(self):
        task = ("Check why the living room plug drops off wifi every "
                "night around 2am and report back with the cause")
        captured: list = []
        with _patch_board(captured), _no_remote_dupe():
            res = await dd.dispatch("chaba", task)
        self.assertTrue(res["ok"])
        self.assertEqual(len(captured), 1)
        path, body = captured[0]
        self.assertEqual(path, "/card")
        self.assertLessEqual(len(body["title"]), 80)
        self.assertTrue(body["title"].startswith("Check why the living"))
        self.assertIn(task, body["spec"])
        self.assertIn("Dispatched by Ada voice/chat", body["spec"])
        self.assertEqual(body["column"], "backlog")
        self.assertEqual(body["tags"], ["voice-dispatch"])
        self.assertEqual(body["priority"], "medium")
        self.assertEqual(body["action"],
                         {"type": "dispatch", "repo": "chaba"})
        self.assertTrue(body["queue"])
        self.assertIn("Ada voice/chat", body["text"])

    async def test_title_truncated_to_80(self):
        captured: list = []
        with _patch_board(captured), _no_remote_dupe():
            await dd.dispatch("chaba", "x" * 200)
        self.assertEqual(len(captured[0][1]["title"]), 80)

    async def test_result_reports_card_id_and_queued_note(self):
        with _patch_board(
                data={"id": "voice-dispatch-99",
                      "message": "card voice-dispatch-99 created"}), \
                _no_remote_dupe():
            res = await dd.dispatch("chaba", "Check the weather station.")
        self.assertEqual(res, {
            "ok": True,
            "card_id": "voice-dispatch-99",
            "note": ("queued on the board — kanban-dispatch will claim "
                     "and run it"),
        })

    async def test_card_id_parsed_from_message(self):
        with _patch_board(
                data={"message": "card check-the-weather created"}), \
                _no_remote_dupe():
            res = await dd.dispatch("chaba", "Check the weather station.")
        self.assertEqual(res["card_id"], "check-the-weather")

    async def test_card_id_falls_back_to_deterministic_id(self):
        with _patch_board(data={"message": "ok"}), _no_remote_dupe():
            res = await dd.dispatch("chaba", "Check the weather station.")
        self.assertTrue(res["card_id"].startswith("voice-dispatch-"))

    async def test_unreachable_returns_error_no_ssh_fallback(self):
        captured: list = []
        with _patch_board(captured, status=0,
                          data={"error": "I couldn't reach the board"}), \
                _no_remote_dupe(), \
                patch.object(dd, "_run", new=AsyncMock()) as run_mock:
            res = await dd.dispatch("chaba", "Check the weather station.")
        self.assertFalse(res["ok"])
        self.assertIn("unreachable", res["error"])
        # no direct devin-dispatch ssh fallback — the board IS the rail
        run_mock.assert_not_called()

    async def test_whitelist_400_surfaced_verbatim(self):
        err = "repo 'nope' not in dispatch whitelist"
        with _patch_board(status=400, data={"error": err}), \
                _no_remote_dupe():
            res = await dd.dispatch("nope", "Check the weather station.")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], err)
        self.assertEqual(res["status"], 400)


class StatusTrimTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_caps_to_latest(self):
        lines = "\n".join(f"2026092{i}-000000-task-{i} inactive success repo=chaba"
                          for i in range(12)) + "\n"
        with patch.object(dd, "_run", new=AsyncMock(return_value=lines)):
            out = await dd.status()
        self.assertIn(f"showing latest {dd.STATUS_MAX_LINES} of 12", out)
        self.assertIn("task-11", out)
        self.assertNotIn("task-0\n", out)

    async def test_single_task_not_trimmed(self):
        payload = "20260924-073049-task-a inactive success repo=chaba\n"
        with patch.object(dd, "_run", new=AsyncMock(return_value=payload)) as run_spy:
            out = await dd.status("20260924-073049-task-a")
        self.assertEqual(out, payload)
        run_spy.assert_called_once_with("status", "20260924-073049-task-a")


if __name__ == "__main__":
    unittest.main()

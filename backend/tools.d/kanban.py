"""kanban — drop-in tool.

Ada's kanban board surface (design: docs/design/ada-kanban-access.md in
chaba): she reviews and acts on the board within a bounded authority —
triage and housekeeping, not judgment. One action= tool; absorbs
ada_board_write, which stays callable via tool_runner._ALIASES.

Reach path: backend/board_client -> board-api on tony-dell (Caddy
/apps/board-api). Write auth is the tailnet identity tailscale serve
injects — verified from idc03 (via=tonezzzz@gmail.com) — so no Ada
credential ever crosses this path. GET /cards is open; every write the
tool makes posts a comms line from 'ada' as the audit anchor (board-api
logs column moves under 'tony' — a known misattribution the ada
comment corrects).

Authority matrix (design §3):

  free        list / read / comment / file / ask — always allowed.
  move        backlog->doing (under doing_limit), doing->backlog,
              doing->review — her own work, free.
              review->done — only with evidence= (a checkable fact she
              verified with a tool this session); refused outright on
              priority:high, review_kind:decide, or prod/security-tagged
              cards — those are Tony's calls.
              every other transition (done->*, *->done other than
              review->done, lab moves, over-cap doing claims) — Tony
              decides: the runner confirm gate (confirmed=true after he
              says yes aloud) or action='ask' for an async board request.
  respond     owner-only — /respond closes a request and the board logs
              it under 'tony'; only full-access identities may speak a
              request closed.
  secondary turns (an identified non-owner voice): list/read only —
  board ops are Tony-tier (design §6). manifest secondary_allowed is
  true precisely so this split can live in-module.
"""

from __future__ import annotations

import re
import secrets
import time
from typing import Any

import httpx

from backend import board_client

# Board writes through Caddy can take ~45s while board-api
# rebuilds the page under flock — reads are fast but share the
# client, so one generous timeout for both.
_HTTP_TIMEOUT = 75.0

DECLARATION = {
    "name": "kanban",
    "description": (
        "Kanban board — action='next'|'list'|'read'|'comment'|'move'|'ask'|"
        "'file'|'respond'. `next` inspects current work and returns "
        "read-only choices; it never queues or dispatches. Moves need the right authority: free within "
        "triage, evidence= to close review cards, Tony's confirm beyond."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "'list' (default read of the board), "
                               "'read' (one card, id= or report=), "
                               "'comment', 'move', 'ask', 'file' "
                               "('create' accepted), 'respond'.",
            },
            "id": {
                "type": "string",
                "description": "Card id, e.g. 'mha-log-noise'.",
            },
            "column": {
                "type": "string",
                "description": "action=list filter; action=move target "
                               "column; action=file start column "
                               "(default backlog).",
            },
            "text": {
                "type": "string",
                "description": "Comment body (action=comment). Prefix "
                               "'[opinion]' for report takes.",
            },
            "evidence": {
                "type": "string",
                "description": "action=move review->done only: the "
                               "checkable fact you verified with a tool "
                               "(service state, page live, log line) — "
                               "written into comms as 'verified: ...'.",
            },
            "ask": {
                "type": "string",
                "description": "Question to Tony on the card "
                               "(action=ask).",
            },
            "options": {
                "type": "array",
                # genai rejects union-typed items (["string","object"]) —
                # it fails LiveConnectConfig validation and kills every
                # voice session at connect (2026-10-09). Declare object;
                # runtime coerces bare strings to {"label": s}.
                "items": {"type": "object"},
                "description": "action=ask only: answer choices rendered "
                               "as one-click buttons for Tony — offer "
                               "whenever the decision has a small option "
                               "set. {id?, label} objects.",
            },
            "suggested": {
                "type": "string",
                "description": "action=ask only: your recommendation — "
                               "must match one of options[] (id, label, "
                               "or 'id — label'). Renders ★ on the "
                               "option; never auto-answers.",
            },
            "title": {
                "type": "string",
                "description": "Card title (action=file).",
            },
            "note": {
                "type": "string",
                "description": "One-line card note (action=file).",
            },
            "priority": {
                "type": "string",
                "description": "action=file only: high|medium|low.",
            },
            "request_id": {
                "type": "string",
                "description": "Id of the card's open request "
                               "(action=respond).",
            },
            "answer": {
                "type": "string",
                "description": "Answer text for the request "
                               "(action=respond).",
            },
            "report": {
                "type": "string",
                "description": "action=read: CMS page slug — returns "
                               "the card whose 'report' field links it "
                               "(the report-opinion lookup).",
            },
            "limit": {
                "type": "integer",
                "description": "Max cards for action=list (default 12).",
            },
            "expected_updated": {
                "type": "string",
                "description": "Stale-write guard (report-first "
                               "protocol): pass the card's 'updated' "
                               "stamp from your last read — comment/"
                               "move/respond refuse when the card "
                               "changed underneath you; re-read, "
                               "reconcile, retry if still applicable.",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Set true only after the user explicitly "
                               "confirmed a gated move aloud.",
            },
            "confirm_token": {
                "type": "string",
                "description": "Bound token returned by a denied call; "
                               "after the user confirms, replay the "
                               "same call with it. Single use, "
                               "expires in 120s.",
            },
        },
    },
}

_CARD_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$", re.IGNORECASE)
_READ_LIMIT_DEFAULT = 12
_READ_LIMIT_MAX = 40
_TEXT_MAX = 480          # server truncates comms/answers at 500
_TITLE_MAX = 80
_ASK_MAX = 100
_COMMS_KEEP = 10         # full-card read keeps the last N comms lines
_COMM_TEXT_MAX = 240
_DEFAULT_COLUMNS = ("lab", "backlog", "doing", "review", "done")
_PRODSEC_TAG = re.compile(r"(?i)prod|security|infra|deploy")
_SECONDARY_OK = {"list", "read"}        # design §6: Tony-tier writes
# absorb ada_board_write's action vocabulary (the alias passes the old
# names through untouched)
_ACTION_SYNONYMS = {"create": "file", "new": "file"}


def _card_id(args: dict[str, Any]) -> str | None:
    cid = str(args.get("id") or "").strip()
    return cid if _CARD_ID.match(cid) else None


def _slug(text: str) -> str:
    return re.sub(r"-+", "-",
                  re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def _cards_only(cards: Any) -> list[dict[str, Any]]:
    """Drop non-dict card entries (malformed YAML) instead of poisoning
    the whole board read."""
    if not isinstance(cards, list):
        return []
    return [c for c in cards if isinstance(c, dict)]


def _resolve_in(cards: list[Any], cid: str) -> dict[str, Any] | None:
    """Card lookup tolerant of the model shortening an id — exact,
    then UNIQUE id-prefix or slugified-title match."""
    for c in cards:
        if c.get("id") == cid:
            return c
    pre = [c for c in cards if str(c.get("id") or "").startswith(cid)]
    if len(pre) == 1:
        return pre[0]
    tit = [c for c in cards if _slug(str(c.get("title") or "")) == cid]
    return tit[0] if len(tit) == 1 else None


async def _resolve(client: httpx.AsyncClient,
                   cid: str, ctx: Any = None) -> dict[str, Any] | None:
    """Fetch the board and resolve a loose id — used to retry writes
    that came back 'no card'."""
    data, err = await _request(client, "GET", "/cards", ctx=ctx)
    if err or not data:
        return None
    return _resolve_in(_cards_only(data.get("cards")), cid)


def _is_owner(runner: Any) -> bool:
    try:
        ident = runner.policy_identity()
        policy = runner.banks.policy_for(ident) or {}
    except Exception:
        return False
    return bool(policy.get("full"))


def _is_secondary(runner: Any) -> bool:
    try:
        return bool(runner._is_secondary_turn())
    except Exception:
        return False


async def _request(client: httpx.AsyncClient, method: str, path: str,
                   ctx: Any = None,
                   **kw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """board_client.request funnel. status 0 = board unreachable — emit
    a throttled ops event through the runner.context facade so the
    outage lands in the hourly digest (runner-facade, 2026-10-10)."""
    data, err, status = await board_client.request(
        client, method, path, **kw)
    if status == 0 and ctx is not None:
        ctx.emit_ops_event(
            "kanban_board_unreachable",
            f"board-api unreachable during {method} {path}: {err}",
            tool="kanban")
    return data, err


async def _comment(client: httpx.AsyncClient, cid: str,
                   text: str, ctx: Any = None) -> tuple[bool, str | None]:
    data, err = await _request(
        client, "POST", "/comment",
        json={"id": cid, "from": "ada", "text": text[:_TEXT_MAX]},
        ctx=ctx)
    if err:
        return False, err
    return True, (data or {}).get("message", "comment added")


# ---------------------------------------------------------------- reads

def _compact_card(c: dict[str, Any]) -> dict[str, Any]:
    # malformed cards (bare-string requests/comms) must not poison the
    # whole board read — coerce str entries to {'needed': text} dicts
    reqs_raw = c.get("requests") or []
    reqs_norm = [r if isinstance(r, dict)
                 else {"needed": str(r)} for r in reqs_raw]
    open_reqs = [r for r in reqs_norm
                 if r.get("status") != "answered"]
    row: dict[str, Any] = {
        "id": c.get("id"),
        "title": str(c.get("title") or "")[:_TITLE_MAX],
        "column": c.get("column") or "backlog",
    }
    if c.get("updated"):
        row["updated"] = c["updated"]
    if c.get("report"):
        row["report"] = str(c["report"])
    if c.get("priority"):
        row["priority"] = str(c["priority"])
    if c.get("review_kind"):
        row["review_kind"] = str(c["review_kind"])
    if open_reqs:
        row["open_requests"] = len(open_reqs)
        row["requests"] = [
            {"id": r.get("id"),
             "ask": str(r.get("ask") or "")[:_ASK_MAX]}
            for r in open_reqs
        ]
    comms = c.get("comms") or []
    comms = [m if isinstance(m, dict) else {"from": "?", "text": str(m)}
             for m in comms]
    if comms:
        last = comms[-1]
        row["last_comm"] = (
            f"{last.get('from', '?')}: "
            f"{str(last.get('text') or '')[:80]}")
    return row


def _full_card(c: dict[str, Any]) -> dict[str, Any]:
    """action=read id=<card> — the whole card, comms trimmed to the last
    few lines so a fat card can't blow the tool-result budget."""
    row = _compact_card(c)
    row.pop("last_comm", None)
    for k in ("note", "brief", "spec", "verify", "metric"):
        if c.get(k):
            row[k] = str(c[k])[:400]
    if c.get("tags"):
        row["tags"] = [str(t) for t in c["tags"]][:10]
    act = c.get("action") or {}
    if act:
        row["action"] = {k: act[k] for k in
                         ("type", "status", "runner", "result")
                         if act.get(k)}
    reqs = c.get("requests") or []
    reqs = [r if isinstance(r, dict) else {"needed": str(r)} for r in reqs]
    if reqs:
        row["requests"] = [
            {"id": r.get("id"), "status": r.get("status") or "open",
             "ask": str(r.get("ask") or "")[:_ASK_MAX],
             **({"answer": str(r.get("answer"))[:_ASK_MAX]}
                if r.get("answer") else {}),
             **({"to": r["to"]} if r.get("to") else {})}
            for r in reqs
        ]
    comms = c.get("comms") or []
    comms = [m if isinstance(m, dict) else {"from": "?", "text": str(m)}
             for m in comms]
    if comms:
        row["comms"] = [
            {"at": m.get("at"), "from": m.get("from"),
             "text": str(m.get("text") or "")[:_COMM_TEXT_MAX]}
            for m in comms[-_COMMS_KEEP:]
        ]
        if len(comms) > _COMMS_KEEP:
            row["comms_earlier"] = len(comms) - _COMMS_KEEP
    return row


def _match_report_card(cards: list[dict[str, Any]], slug: str
                       ) -> dict[str, Any]:
    """Opinion-loop lookup: find the kanban cards whose `report` field
    links a CMS page slug. Open (non-'done' column) cards outrank done
    ones; within each tier the most recently `updated` wins. Returns
    {'open': [...], 'done': [...]}, each sorted updated-desc."""
    want = slug.strip().lower()
    cands = {want}
    if want.endswith("-report"):
        cands.add(want[: -len("-report")])
    else:
        cands.add(f"{want}-report")
    open_, done = [], []
    for c in cards:
        if str(c.get("report") or "").strip().lower() not in cands:
            continue
        col = str(c.get("column") or "backlog")
        (done if col == "done" else open_).append(c)
    key = lambda c: str(c.get("updated") or "")
    open_.sort(key=key, reverse=True)
    done.sort(key=key, reverse=True)
    return {"open": open_, "done": done}


def _report_lookup(cards: list[dict[str, Any]], slug: str
                   ) -> dict[str, Any]:
    """action=read report=<slug> — returns the single best card linking
    the slug plus counts so the model can narrate ambiguity aloud or
    offer to file a card when nothing links it."""
    match = _match_report_card(cards, slug)
    open_, done = match["open"], match["done"]
    out: dict[str, Any] = {
        "ok": True,
        "report": slug.strip().lower(),
        "open_matches": len(open_),
        "done_matches": len(done),
    }
    if open_:
        out["card"] = _compact_card(open_[0])
        if len(open_) > 1:
            out["also"] = [str(c.get("id")) for c in open_[1:]]
            out["note"] = (
                "several open cards link this report — the most "
                "recently updated was picked; say which card got the "
                "comment")
    elif done:
        out["card"] = _compact_card(done[0])
        out["note"] = (
            "only done cards link this report — the comment still "
            "lands on it, or offer to file a fresh card "
            "(action='file')")
    else:
        out["card"] = None
        out["note"] = (
            "no card links this report — answer aloud and offer to "
            "file one (action='file')")
    return out


async def _list(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err = await _request(client, "GET", "/cards", ctx=ctx)
    if err:
        return {"ok": False, "error": err}
    assert data is not None
    slug = str(args.get("report") or "").strip()
    if slug:
        return _report_lookup(_cards_only(data.get("cards")), slug)
    want_col = str(args.get("column") or "").strip().lower()
    try:
        limit = int(args.get("limit") or _READ_LIMIT_DEFAULT)
    except (TypeError, ValueError):
        limit = _READ_LIMIT_DEFAULT
    limit = max(1, min(limit, _READ_LIMIT_MAX))
    cards = _cards_only(data.get("cards"))
    columns: dict[str, int] = {}
    open_requests = 0
    rows = []
    for c in cards:
        col = str(c.get("column") or "backlog")
        columns[col] = columns.get(col, 0) + 1
        open_requests += sum(
            1 for r in (c.get("requests") or [])
            if r.get("status") != "answered")
        if want_col and col != want_col:
            continue
        rows.append(_compact_card(c))
    rows.sort(key=lambda r: str(r.get("updated") or ""), reverse=True)
    out: dict[str, Any] = {
        "ok": True,
        "columns": columns,
        "open_requests": open_requests,
        "total": len(rows),
        "cards": rows[:limit],
    }
    if len(rows) > limit:
        out["truncated"] = len(rows) - limit
    return out


def _next_rank(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a read-only next-step menu from current board truth."""
    active, review, backlog = [], [], []
    blockers = []
    for card in cards:
        col = str(card.get("column") or "backlog")
        compact = _compact_card(card)
        open_reqs = compact.get("open_requests", 0)
        if open_reqs:
            blockers.append(compact)
        if col == "doing":
            active.append(compact)
        elif col == "review":
            review.append(compact)
        elif col == "backlog" and not open_reqs:
            backlog.append(compact)
    key = lambda c: (str(c.get("updated") or ""), str(c.get("id") or ""))
    active.sort(key=key, reverse=True)
    review.sort(key=key, reverse=True)
    backlog.sort(key=lambda c: (str(c.get("priority") or "medium") != "high", -len(str(c.get("title") or "")), str(c.get("updated") or "")))
    menu = []
    for c in active[:3]:
        menu.append({"id": f"continue:{c['id']}", "label": f"Continue: {c['title']}", "card_id": c["id"], "operation": "plan"})
    for c in review[:3]:
        menu.append({"id": f"review:{c['id']}", "label": f"Review: {c['title']}", "card_id": c["id"], "operation": "review"})
    for c in backlog[:3]:
        menu.append({"id": f"plan:{c['id']}", "label": f"Plan: {c['title']}", "card_id": c["id"], "operation": "plan"})
    menu.append({"id": "stop", "label": "Stop", "operation": "stop"})
    return {
        "ok": True, "operation": "next",
        "question_id": "next-" + secrets.token_urlsafe(9),
        "expires_at": time.time() + 900,
        "current_work": active[:3], "review_ready": review[:3],
        "safe_candidates": backlog[:3], "blockers": blockers[:8],
        "next_menu": menu, "mutations_performed": [],
        "next_safe_action": "present the menu and wait for Tony's choice; do not queue or dispatch",
    }


async def _next(ctx: Any) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err = await _request(client, "GET", "/cards", ctx=ctx)
    if err:
        return {"ok": False, "operation": "next", "error": err}
    return _next_rank(_cards_only((data or {}).get("cards")))


async def _read(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    cid = _card_id(args)
    if not cid:
        return await _list(ctx, args)
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err = await _request(client, "GET", "/cards", ctx=ctx)
    if err:
        return {"ok": False, "error": err}
    assert data is not None
    card = _resolve_in(_cards_only(data.get("cards")), cid)
    if card is not None:
        return {"ok": True, "card": _full_card(card)}
    return {"ok": False,
            "error": f"no card {cid!r} on the board — check the id "
                     "with action='list'"}


# ---------------------------------------------------------------- writes

async def _stale_check(client: httpx.AsyncClient, args: dict[str, Any],
                       cid: str,
                       card: dict[str, Any] | None = None,
                       ctx: Any = None
                       ) -> dict[str, Any] | None:
    """Optimistic-concurrency guard (ssot.apps.cms-reports.yml
    report_first_protocol#write_guard): when the caller passes
    expected_updated=<stamp from the last read>, a mismatch means the
    card moved under us — refuse so the caller re-reads and reconciles
    instead of clobbering a newer update. Reports are snapshots; the
    stamp is what turns a stale read into a visible conflict."""
    exp = str(args.get("expected_updated") or "").strip()
    if not exp:
        return None
    if card is None:
        data, err = await _request(client, "GET", "/cards", ctx=ctx)
        if err:
            return {"ok": False, "error": err}
        card = _resolve_in(_cards_only(data.get("cards")), cid)
    if card is None:
        return {"ok": False,
                "error": f"no card {cid!r} on the board — check the id "
                         "with action='list'"}
    cur = str(card.get("updated") or "")
    if cur != exp:
        return {"ok": False, "stale": True,
                "id": str(card.get("id") or cid), "updated": cur,
                "error": f"card changed since your read (expected "
                         f"updated={exp}, now {cur or '?'}) — re-read "
                         f"it, reconcile, and retry if it still applies"}
    return None


async def _do_comment(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    cid = _card_id(args)
    text = str(args.get("text") or "").strip()
    if not cid:
        return {"ok": False, "error": "a valid card id is required"}
    if not text:
        return {"ok": False, "error": "comment text is required"}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        stale = await _stale_check(client, args, cid, ctx=ctx)
        if stale:
            return stale
        posted, msg = await _comment(client, cid, text, ctx=ctx)
        if not posted and msg and "no card" in msg:
            card = await _resolve(client, cid, ctx)
            if card is not None:
                cid = str(card["id"])
                posted, msg = await _comment(client, cid, text, ctx=ctx)
    if not posted:
        return {"ok": False, "error": msg}
    return {"ok": True, "id": cid, "posted": text[:80],
            "message": msg or "comment added"}


_CARD_CREATED_RE = re.compile(r"card\s+([a-z0-9][a-z0-9._-]*)",
                              re.IGNORECASE)


def _filed_card_id(data: dict[str, Any] | None) -> str | None:
    """Pull the created card's id out of a POST /card response — the
    field name isn't fixed, so check the known spots then the message
    text ('card <id> created')."""
    data = data or {}
    for key in ("id", "card_id"):
        cid = str(data.get(key) or "").strip()
        if cid:
            return cid
    card = data.get("card")
    if isinstance(card, dict) and card.get("id"):
        return str(card["id"])
    m = _CARD_CREATED_RE.search(str(data.get("message") or ""))
    return m.group(1) if m else None


async def _file(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Drop a new card onto the board — the voice-side capture of the
    request lifecycle. The card lands in backlog for triage, comms
    record 'ada'. Verify-after-write (card ada-phantom-card-claims): the
    board IS the record — a 200 that landed nothing would let the
    narration claim a card that silently drops the ask, so the tool
    confirms the card is readable before reporting ok."""
    title = str(args.get("title") or "").strip()
    if not title:
        return {"ok": False, "error": "a card title is required"}
    body: dict[str, Any] = {"title": title[:_TITLE_MAX * 2],
                            "from": "ada"}
    note = str(args.get("note") or "").strip()
    if note:
        body["note"] = note[:_TEXT_MAX]
        body["text"] = note[:_TEXT_MAX]
    col = str(args.get("column") or "").strip().lower()
    if col:
        body["column"] = col
    pri = str(args.get("priority") or "").strip().lower()
    if pri:
        if pri not in ("high", "medium", "low"):
            return {"ok": False,
                    "error": "priority must be high|medium|low"}
        body["priority"] = pri
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err = await _request(client, "POST", "/card", json=body,
                                   ctx=ctx)
        if err:
            return {"ok": False, "error": err}
        cid = _filed_card_id(data)
        verified = False
        vdata, verr = await _request(client, "GET", "/cards", ctx=ctx)
        if not verr and vdata is not None:
            cards = _cards_only(vdata.get("cards"))
            found = _resolve_in(cards, cid) if cid else None
            if found is None:
                want = _slug(title)
                titled = [c for c in cards
                          if _slug(str(c.get("title") or "")) == want]
                found = titled[0] if len(titled) == 1 else None
            if found is None:
                return {"ok": False,
                        "error": "the board acknowledged the write but "
                                 "no card appeared — it did not file. "
                                 "Say the write didn't land; retry or "
                                 "leave it for Tony."}
            cid = str(found.get("id") or "") or cid
            verified = True
    out: dict[str, Any] = {
        "ok": True,
        "message": (data or {}).get("message", "card created"),
        "note": "the card is on the board — triage picks it up from "
                "there",
    }
    if cid:
        out["id"] = cid
        out["note"] = (f"card '{cid}' is on the board — say 'on the "
                       "board' with the id so the claim stays checkable")
    if not verified:
        out["warning"] = ("couldn't verify the card landed — the board "
                          "read failed after the write; check "
                          "action='list' before claiming it's filed")
    return out


async def _ask(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Raise a board request — the escalation path for decisions that
    are Tony's (gated moves, unclear triage). The request surfaces on
    the card and pings him via board-notify."""
    cid = _card_id(args)
    ask = str(args.get("ask") or args.get("text") or "").strip()
    if not cid:
        return {"ok": False, "error": "a valid card id is required"}
    if not ask:
        return {"ok": False, "error": "ask text is required"}
    body: dict[str, Any] = {"id": cid, "from": "ada",
                            "ask": ask[:_TEXT_MAX]}
    options = args.get("options")
    if isinstance(options, list) and options:
        opts = [o if isinstance(o, dict) else {"label": str(o)}
                for o in options[:6]]
        body["options"] = opts
        sug = str(args.get("suggested") or "").strip()
        if sug:
            body["suggested"] = sug
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err = await _request(client, "POST", "/request", json=body,
                                   ctx=ctx)
        if err and "no card" in err:
            card = await _resolve(client, cid, ctx)
            if card is not None:
                body["id"] = cid = str(card["id"])
                data, err = await _request(
                    client, "POST", "/request", json=body, ctx=ctx)
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, "id": cid,
            "message": (data or {}).get("message", "request raised"),
            "note": "the request sits on the card for Tony — say you'll "
                    "leave it for him, not that it is decided"}


def _gate_reason(card: dict[str, Any], target: str,
                 doing_count: int, doing_limit: int,
                 evidence: str) -> str | None:
    """None = free move; a string = why Tony's yes is needed (feeds the
    confirm-gate instruction). Hard refusals are handled by the caller
    before this runs (review->done without evidence)."""
    cur = str(card.get("column") or "backlog")
    if target == "doing":
        if cur != "backlog":
            return f"only backlog->doing is a free pickup; {cur}->doing " \
                   "is a reopen Tony should sanction"
        if doing_count >= doing_limit:
            return (f"doing already holds {doing_count} cards "
                    f"(limit {doing_limit}) — the same cap agents work "
                    "under; Tony can confirm overriding it")
        return None
    if cur == "doing" and target in ("backlog", "review"):
        return None                      # releasing / handing off work
    if cur == "review" and target == "done":
        gated: list[str] = []
        if str(card.get("priority") or "").lower() == "high":
            gated.append("priority:high")
        if str(card.get("review_kind") or "").lower() == "decide":
            gated.append("review_kind:decide")
        if any(_PRODSEC_TAG.search(str(t))
               for t in (card.get("tags") or [])):
            gated.append("prod/security tag")
        if gated:
            return ("closing this card is Tony's call — it is marked "
                    + ", ".join(gated))
        return None
    return (f"{cur}->{target} is outside Ada's triage authority "
            "(design §3) — Tony decides")


async def _move(runner: Any, args: dict[str, Any],
                confirmed: Any, confirm_token: Any) -> dict[str, Any]:
    ctx = getattr(runner, "context", None)
    cid = _card_id(args)
    target = str(args.get("column") or "").strip().lower()
    evidence = str(args.get("evidence") or "").strip()
    if not cid:
        return {"ok": False, "error": "a valid card id is required"}
    if not target:
        return {"ok": False, "error": "a target column is required"}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err = await _request(client, "GET", "/cards", ctx=ctx)
        if err:
            return {"ok": False, "error": err}
        assert data is not None
        payload, cards = data, _cards_only(data.get("cards"))
        card = _resolve_in(cards, cid)
        if card is None:
            return {"ok": False,
                    "error": f"no card {cid!r} on the board — check the "
                             "id with action='list'"}
        cid = str(card["id"])
        col_ids = {str(c.get("id")) for c in payload.get("columns") or []}
        columns = col_ids or set(_DEFAULT_COLUMNS)
        if target not in columns:
            return {"ok": False,
                    "error": f"no column {target!r} — board columns: "
                             + ", ".join(sorted(columns))}
        stale = await _stale_check(client, args, cid, card=card, ctx=ctx)
        if stale:
            return stale
        cur = str(card.get("column") or "backlog")
        if cur == target:
            return {"ok": True, "id": cid,
                    "message": f"card is already in {target}"}
        doing_limit = int(payload.get("doing_limit") or 2)
        doing_count = sum(1 for c in cards
                          if str(c.get("column") or "") == "doing"
                          and c.get("id") != cid)
        # review->done without evidence is a hard rule, not confirmable —
        # the evidence IS what makes a machine-close trustworthy.
        if cur == "review" and target == "done" and not evidence:
            return {"ok": False,
                    "error": "closing a review card needs evidence= — a "
                             "checkable fact you verified with a tool "
                             "(service active, page live, log line). "
                             "Verify first, then move with evidence."}
        reason = _gate_reason(card, target, doing_count, doing_limit,
                              evidence)
        if reason is not None:
            # Tony-decides move — the standard runner confirm gate:
            # denial mints a bound token; his spoken yes + a replay with
            # confirmed=true (or the token) executes it. The async path
            # is action='ask'. The raise is converted into a
            # needs_confirm result so the denial breaker tracks retries.
            try:
                runner._require_confirmation(
                    "kanban",
                    {"action": "move", "id": cid, "column": target},
                    confirmed, confirm_token,
                    f"NOT EXECUTED — kanban move {cid} {cur}->{target}: "
                    f"{reason}. Say what you want to do and get Tony's "
                    "explicit yes; or leave him a board question with "
                    "action='ask'.")
            except PermissionError as exc:
                return {"ok": False, "needs_confirm": True,
                        "error": str(exc)}
        # authorized — do the move, then write the audit comment as ada
        mdata, merr = await _request(
            client, "POST", "/action",
            json={"id": cid, "do": "move", "column": target}, ctx=ctx)
        if merr:
            return {"ok": False, "error": merr}
        audit = f"moved {cur} -> {target}"
        if evidence:
            audit += f" — verified: {evidence}"
        if reason is not None:
            audit += " (Tony confirmed)"
        posted, cerr = await _comment(client, cid, audit, ctx=ctx)
        out: dict[str, Any] = {
            "ok": True, "id": cid, "from": cur, "to": target,
            "message": (mdata or {}).get("message",
                                        f"moved to {target}"),
        }
        if evidence:
            out["evidence"] = evidence[:160]
        if not posted:
            out["warning"] = ("the move landed but the audit comment "
                              f"failed: {cerr} — retry action='comment'")
        return out


async def _respond(runner: Any, args: dict[str, Any]) -> dict[str, Any]:
    ctx = getattr(runner, "context", None)
    if not _is_owner(runner):
        return {"ok": False,
                "error": "answering board requests is restricted to the "
                         "owner — suggest the answer aloud instead"}
    cid = _card_id(args)
    rid = str(args.get("request_id") or "").strip()
    answer = str(args.get("answer") or "").strip()
    if not cid:
        return {"ok": False, "error": "a valid card id is required"}
    if not rid:
        return {"ok": False, "error": "request_id is required"}
    if not answer:
        return {"ok": False, "error": "answer text is required"}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        stale = await _stale_check(client, args, cid, ctx=ctx)
        if stale:
            return stale
        body = {"id": cid, "request_id": rid,
                "answer": answer[:_TEXT_MAX], "from": "ada"}
        data, err = await _request(client, "POST", "/respond", json=body,
                                   ctx=ctx)
        if err and "no card" in err:
            card = await _resolve(client, cid, ctx)
            if card is not None:
                body["id"] = cid = str(card["id"])
                data, err = await _request(
                    client, "POST", "/respond", json=body, ctx=ctx)
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, "id": cid, "request_id": rid,
            "message": (data or {}).get("message", "answer saved"),
            "note": "the board logs respond comms under 'tony' — a "
                    "board-api limitation, not something you did wrong"}


async def run(runner: Any, confirmed: Any = None,
              confirm_token: Any = None, **args: Any) -> dict[str, Any]:
    action = str(args.get("action") or "list").strip().lower()
    action = _ACTION_SYNONYMS.get(action, action)
    ctx = getattr(runner, "context", None)
    # design §6: board ops are Tony-tier — an identified non-owner voice
    # gets the read actions only.
    if action not in _SECONDARY_OK and _is_secondary(runner):
        return {"ok": False,
                "error": "board writes are the session owner's — ask "
                         "them to make the change in their own voice"}
    if action == "next":
        return await _next(ctx)
    if action == "list":
        return await _list(ctx, args)
    if action == "read":
        return await _read(ctx, args)
    if action == "comment":
        return await _do_comment(ctx, args)
    if action == "move":
        return await _move(runner, args, confirmed, confirm_token)
    if action == "ask":
        return await _ask(ctx, args)
    if action == "file":
        return await _file(ctx, args)
    if action == "respond":
        return await _respond(runner, args)
    return {"ok": False,
            "error": f"unknown action {action!r} — use next, list, read, "
                     "comment, move, ask, file, or respond"}

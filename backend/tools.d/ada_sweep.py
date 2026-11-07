"""ada_sweep — batch preflight + queue of executor-tagged backlog cards.

One confirmed tool call walks the backlog instead of per-card rounds:
read the column, preflight each card that carries the requested
executor envelope, report per-card gaps, and queue the ready ones.
'sweep my opencode cards' -> one confirm, one report (ADA-1108).

Queue is a plain /action do=queue — the card already carries its full
executor envelope (that's how it became sweep-eligible), so no receipt
re-binding is needed the way ada_opencode_dispatch requires.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend import board_client
from backend.ada_opencode_control import preflight

_HTTP_TIMEOUT = 60.0
_MAX_SWEEP = 20

DECLARATION = {
    "name": "ada_sweep",
    "description": (
        "Batch-process eligible backlog cards: preflight each executor-"
        "tagged card, report per-card gaps, and queue the ready ones "
        "under this single confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "executor": {
                "type": "string",
                "description": "Executor to sweep (default opencode).",
            },
            "column": {
                "type": "string",
                "description": "Board column to sweep (default backlog).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Preflight and report only; queue nothing.",
            },
            "limit": {
                "type": "integer",
                "description": f"Max cards to process (cap {_MAX_SWEEP}).",
            },
        },
    },
}


def _entry(card: dict[str, Any]) -> dict[str, Any]:
    return {"id": card.get("id"), "ref": card.get("ref"),
            "title": card.get("title")}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    executor = str(args.get("executor") or "opencode").strip() or "opencode"
    column = str(args.get("column") or "backlog").strip() or "backlog"
    dry_run = bool(args.get("dry_run"))
    try:
        limit = max(1, min(int(args.get("limit") or _MAX_SWEEP), _MAX_SWEEP))
    except (TypeError, ValueError):
        limit = _MAX_SWEEP

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data, err, status = await board_client.request(
            client, "GET",
            f"/cards?column={column}&fields=id,title,ref,updated,action")
        if err:
            return {"ok": False, "error": err, "status": status}
        cards = [c for c in ((data or {}).get("cards") or [])
                 if isinstance(c, dict)]

        candidates, skipped = [], []
        for c in cards:
            a = c.get("action") or {}
            if str(a.get("executor") or "") != executor:
                continue
            st = str(a.get("status") or "")
            if st in ("queued", "running", "claimed"):
                skipped.append({**_entry(c), "why": f"already {st}"})
            elif str(a.get("type") or "") != "dispatch":
                skipped.append({**_entry(c),
                                "why": f"not dispatch "
                                       f"({a.get('type') or 'no type'})"})
            else:
                candidates.append(c)
        capped = len(candidates) > limit
        candidates = candidates[:limit]

        ready, blocked = [], []
        for c in candidates:
            a = c.get("action") or {}
            gate = preflight(
                c, {**a,
                    "task": str(a.get("prompt") or c.get("title") or "")})
            if gate.get("ready_for_dispatch"):
                ready.append({**_entry(c),
                              "workflow_id": gate.get("workflow_id")})
            else:
                blocked.append({**_entry(c),
                                "missing": gate.get("missing") or [],
                                "errors": gate.get("errors") or []})

        queued, queue_errors = [], []
        if not dry_run:
            for r in ready:
                qdata, qerr, qstatus = await board_client.request(
                    client, "POST", "/action",
                    json={"id": r["id"], "do": "queue", "from": "ada"})
                if qerr or (isinstance(qdata, dict)
                            and qdata.get("ok") is False):
                    queue_errors.append(
                        {"id": r["id"],
                         "error": qerr or (qdata or {}).get("error")})
                else:
                    queued.append(r["id"])

    n_q = len(queued)
    summary = (
        f"{column} sweep ({executor}): {len(candidates)} candidate(s), "
        f"{len(ready)} ready, {len(blocked)} blocked"
        + (f", {len(skipped)} already active" if skipped else "")
        + (f"; queued {n_q}" + (f", {len(queue_errors)} failed"
                                if queue_errors else "")
           if not dry_run else " (dry run — nothing queued)"))
    return {
        "ok": True,
        "summary": summary,
        "column": column,
        "executor": executor,
        "dry_run": dry_run,
        "scanned": len(cards),
        "candidates": len(candidates),
        "capped": capped,
        "ready": ready,
        "blocked": blocked,
        "skipped": skipped,
        "queued": queued,
        "queue_errors": queue_errors,
        "mutations": [f"queued {n_q} card(s)"] if n_q else [],
    }

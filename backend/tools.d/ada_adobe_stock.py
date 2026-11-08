"""ada_adobe_stock — Adobe Stock campaign QC surface (card ADA-1130).

Read-only campaign status + Tony's keep/reject/revise decisions for the
AS-2026-11-05-REMOTE-WORK-001 pipeline. Data path: picker-server on
tony-dell (:8777, tailnet 100.68.142.13) which proxies rclone/GDrive —
no Drive creds ever reach Ada.

  action='status' (default)
      -> {campaign, focus_asset, candidates[], next_menu[], picker_url}
      next_menu renders as buttons in the webchat card; a click sends
      'next choice adobe_qc:<decision>:<asset>' as a text turn — map it
      to action='decide' with the same asset+decision.
  action='decide'
      args: asset, decision(keep|reject|revise), note
      -> POSTs /api/qc-decision (updates SESSION.json + 40-qc-review.md
      on Drive) then returns refreshed status.
  action='repair_info'
      -> picker URL for region-select defect repair (mask-picker page;
      submissions auto-land via inbox-processor).

Authority: status/repair_info are reads (secondary_allowed true).
decide records TONY's decision — only call it when Tony actually picked
(a menu click or explicit words); never decide for him.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

PICKER = os.environ.get("ADA_ADOBE_PICKER",
                        "http://100.68.142.13:8777")
CAMPAIGN = "AS-2026-11-05-REMOTE-WORK-001"
_TIMEOUT = 30.0
_DECISIONS = ("keep", "reject", "revise")

TOOL = {
    "name": "ada_adobe_stock",
    "description": (
        "Adobe Stock campaign QC — status/decide/repair_info. status "
        "returns candidate + next_menu; 'next choice adobe_qc:<decision>:"
        "<asset>' -> decide. repair_info returns picker URL. No upload."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "'status' (default), 'decide', "
                               "'repair_info'.",
            },
            "asset": {
                "type": "string",
                "description": "Candidate asset id, e.g. "
                               "'comparison-e-gemini-fix3-region'.",
            },
            "decision": {
                "type": "string",
                "description": "action=decide only: keep | reject | revise.",
            },
            "note": {
                "type": "string",
                "description": "action=decide only: Tony's QC note.",
            },
        },
    },
}


async def _get(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(f"{PICKER}{path}")
        r.raise_for_status()
        return r.json()


async def _status() -> dict[str, Any]:
    camp, sess = await _gather()
    assets = camp.get("generated_files") or []
    focus = ((sess.get("comparison") or {}).get("last_landed") or {}) \
        .get("asset_id") or (assets[-1] if assets else None)
    pending = [a for a in assets if a != focus] + ([focus] if focus else [])
    candidates = [{"asset_id": a,
                   "image_url": f"{PICKER}/img/{a}"} for a in pending[-6:]]
    menu = []
    if focus:
        for dec in _DECISIONS:
            menu.append({"id": f"adobe_qc:{dec}:{focus}",
                         "label": f"{dec.capitalize()} {focus}"})
        menu.append({"id": f"adobe_qc:repair:{focus}",
                     "label": f"Repair region… ({focus})"})
    return {
        "ok": True, "campaign": CAMPAIGN,
        "stage": camp.get("stage"), "status": camp.get("status"),
        "focus_asset": focus, "candidates": candidates,
        "pending_decisions": camp.get("pending_decisions") or [],
        "next_menu": menu,
        "picker_url": (f"{PICKER}/?img=/img/{focus}" if focus else PICKER),
        "adobe_upload": False,
    }


async def _gather() -> tuple[dict, dict]:
    import asyncio
    camp, sess = await asyncio.gather(_get("/api/campaign"),
                                      _get("/api/session"))
    return camp, sess


async def _decide(args: dict[str, Any]) -> dict[str, Any]:
    asset = str(args.get("asset") or "").strip()
    decision = str(args.get("decision") or "").strip().lower()
    note = str(args.get("note") or "")[:400]
    if not asset:
        return {"ok": False, "error": "asset required"}
    if decision == "repair":
        return await _repair_info()
    if decision not in _DECISIONS:
        return {"ok": False,
                "error": f"decision must be one of {_DECISIONS} or 'repair'"}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.post(f"{PICKER}/api/qc-decision",
                         json={"asset": asset, "decision": decision,
                               "note": note})
        data = r.json() if r.headers.get("content-type", "") \
            .startswith("application/json") else {}
    if not r.is_success or not data.get("ok"):
        return {"ok": False,
                "error": f"qc write failed: {data.get('result') or r.status_code}"}
    out = await _status()
    out["recorded"] = {"asset": asset, "decision": decision}
    return out


async def _repair_info() -> dict[str, Any]:
    return {"ok": True, "picker_url": f"{PICKER}/",
            "howto": "Open picker_url, load the candidate image "
                     "(?img=<name> or file picker), drag a rectangle over "
                     "the defect, comment, Submit — the repair auto-lands "
                     "as a new candidate within ~5 min."}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    action = str(args.get("action") or "status").strip().lower()
    if action in ("status", "qc", "pending"):
        return await _status()
    if action == "decide":
        return await _decide(args)
    if action in ("repair", "repair_info", "picker"):
        return await _repair_info()
    return {"ok": False,
            "error": "unknown action — use status, decide, repair_info"}

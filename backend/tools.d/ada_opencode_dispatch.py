"""Confirmed Ada -> OpenCode queue adapter."""

from __future__ import annotations

from typing import Any

import httpx

from backend import board_client
from backend.ada_opencode_control import approval_receipt, card, preflight

DECLARATION = {
    "name": "ada_opencode_dispatch",
    "description": (
        "Queue a preflight-approved OpenCode job for an exact card. "
        "Requires explicit confirmation; never changes model, key, host, "
        "profile, artifact, or limits implicitly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card_reference": {"type": "string"},
            "task": {"type": "string", "description": "Bounded task prompt."},
            "repo": {"type": "string", "description": "Declared repository."},
            "model": {"type": "string"},
            "llm_key_alias": {"type": "string"},
            "profile": {"type": "string"},
            "target_host": {"type": "string"},
            "artifact": {"type": "string"},
            "max_usd": {"type": "number"},
            "max_turns": {"type": "integer"},
            "max_tokens": {"type": "integer"},
            "max_wall_time_s": {"type": "integer"},
            "max_retries": {"type": "integer"},
            "approval_receipt": {"type": "string", "description": "Exact receipt returned by ada_opencode preflight; do not invent."},
            "approval_issued_at": {"type": "integer", "description": "Exact issue timestamp returned by preflight."},
            "approval_expires_at": {"type": "integer", "description": "Exact expiry timestamp returned by preflight."},
        },
        "required": ["card_reference", "task", "approval_receipt", "approval_issued_at", "approval_expires_at"],
    },
}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    reference = str(args.get("card_reference") or "").strip()
    task = str(args.get("task") or "").strip()
    if not reference or not task:
        return {"ok": False, "error": "card_reference and task are required"}
    if len(task) > 6000:
        return {"ok": False, "error": "task exceeds 6000 characters"}
    card_data, err = await card(reference)
    if err or card_data is None:
        return {"ok": False, "error": err or "card not found"}
    gate = preflight(card_data, args)
    now = int(__import__("time").time())
    if int(args.get("approval_expires_at") or 0) < now:
        return {"ok": False, "error": "approval_receipt_expired", "preflight": gate}
    issued_claim = int(args.get("approval_issued_at") or 0)
    expected = approval_receipt(card_data, gate["limits"], task, issued_claim)
    if issued_claim <= 0 or args.get("approval_receipt") != expected:
        return {"ok": False, "error": "approval_receipt_mismatch", "preflight": gate}
    if not gate["ready_for_dispatch"]:
        return {"ok": False, "error": "preflight_failed", "preflight": gate}
    limits = gate["limits"]
    action = {
        "type": "dispatch",
        "executor": "opencode",
        "repo": limits["repo"],
        "model": limits["model"],
        "llm_key_alias": limits["llm_key_alias"],
        "profile": limits["profile"],
        "target_host": limits["target_host"],
        "artifact": limits["artifact"],
        "verifier": limits["verifier"],
        "prompt": task,
        "expect": {
            "required_keys": ["artifact", "tests", "verifier"],
            "facts": [
                f"budget_usd<={limits['max_usd']}",
                f"max_turns={limits['max_turns']}",
                f"max_tokens={limits['max_tokens']}",
                f"max_wall_time_s={limits['max_wall_time_s']}",
                f"max_retries={limits['max_retries']}",
            ],
        },
    }
    payload = {
        "id": card_data["id"],
        "title": card_data.get("title") or reference,
        "brief": card_data.get("brief") or task[:600],
        "action": action,
        "on_exists": "queue",
        "from": "ada",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        data, api_err, status = await board_client.request(
            client, "POST", "/card", json=payload
        )
    if api_err:
        return {"ok": False, "error": api_err, "status": status}
    return {
        "ok": True,
        "queued": True,
        "card_id": card_data["id"],
        "workflow_id": gate["workflow_id"],
        "executor": "opencode",
        "limits": limits,
        "board": data or {},
        "mutations": ["card queued"],
    }

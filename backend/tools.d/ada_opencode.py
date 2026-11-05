"""Ada's read-only OpenCode control-plane adapter.

Ada owns routing and workflow truth. This module only reads the board and
performs a deterministic preflight; the separate confirmed dispatch tool is
the only path that queues work.
"""

from __future__ import annotations

from typing import Any

from backend.ada_opencode_control import action as card_action
from backend.ada_opencode_control import card, preflight as control_preflight

DECLARATION = {
    "name": "ada_opencode",
    "description": (
        "Ada/OpenCode control plane — action='preflight'|'status'|'receipts'. "
        "Read-only: validates a card's explicit OpenCode envelope, budget "
        "limits, artifact, and verifier before any dispatch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "preflight, status, or receipts.",
            },
            "card_reference": {
                "type": "string",
                "description": "Exact card id or ref, for example ADA-1010.",
            },
            "task": {"type": "string", "description": "Optional exact bounded task for receipt binding."},
            "repo": {"type": "string", "description": "Declared repository."},
            "model": {"type": "string"},
            "llm_key_alias": {
                "type": "string",
                "description": "Explicit LiteLLM alias; never infer or switch it.",
            },
            "profile": {"type": "string"},
            "target_host": {"type": "string"},
            "artifact": {
                "type": "string",
                "description": "Required repo-relative result artifact.",
            },
            "max_usd": {"type": "number"},
            "max_turns": {"type": "integer"},
            "max_tokens": {"type": "integer"},
            "max_wall_time_s": {"type": "integer"},
            "max_retries": {"type": "integer"},
        },
        "required": ["action", "card_reference"],
    },
}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    action = str(args.get("action") or "").strip().lower()
    if action not in {"preflight", "status", "receipts"}:
        return {"ok": False, "error": "action must be preflight|status|receipts"}
    reference = str(args.get("card_reference") or "").strip()
    if not reference:
        return {"ok": False, "error": "card_reference is required"}
    try:
        card_data, err = await card(reference)
    except Exception as exc:
        return {"ok": False, "error": f"board read failed: {exc}"}
    if err or card_data is None:
        return {"ok": False, "error": err or "card not found"}
    if action == "preflight":
        return control_preflight(card_data, args)
    current = card_action(card_data)
    result: dict[str, Any] = {
        "ok": True,
        "card_id": card_data.get("id"),
        "card_ref": card_data.get("ref"),
        "title": card_data.get("title"),
        "column": card_data.get("column"),
        "action": {
            key: current.get(key)
            for key in ("executor", "status", "runner", "task_id", "model",
                        "llm_key_alias", "profile", "target_host", "artifact")
            if current.get(key) is not None
        },
    }
    if action == "receipts":
        comms = card_data.get("comms") or []
        result["receipts"] = [
            entry for entry in comms[-10:] if isinstance(entry, dict)
        ]
        result["mutations"] = []
    return result

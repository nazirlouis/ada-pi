"""wiki_extension — confirm-gated, allowlisted MediaWiki extension control."""

from __future__ import annotations

from typing import Any

from backend import board_client

_ALLOWED = {
    "CategoryTree", "Mermaid", "MsUpload", "MultimediaViewer",
    "Cargo", "PageForms", "Lockdown",
}
_ACTIONS = {"list", "install", "enable", "disable"}

DECLARATION = {
    "name": "wiki_extension",
    "description": (
        "List or manage the approved MediaWiki extension bundle through "
        "board-api. Install, enable, and disable require explicit "
        "confirmation. Never accept arbitrary repositories or shell commands."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "list, install, enable, or disable.",
            },
            "name": {
                "type": "string",
                "description": "Approved extension name.",
            },
        },
        "required": ["action"],
    },
}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    action = str(args.get("action") or "").strip().lower()
    name = str(args.get("name") or "").strip()
    if action not in _ACTIONS:
        return {"ok": False, "error": "action must be list|install|enable|disable"}
    if action != "list" and name not in _ALLOWED:
        return {"ok": False, "error": f"name is not approved: {sorted(_ALLOWED)}"}
    body = {"action": action, "from": "ada"}
    if name:
        body["name"] = name
    status, result = await board_client.post("/wiki/extension", body,
                                             timeout=90)
    if status <= 0:
        return {"ok": False, "error": result.get("error", "board unavailable")}
    if status >= 400 or result.get("ok") is False:
        return {"ok": False, "error": result.get("error", result)}
    return {"ok": True, **result}

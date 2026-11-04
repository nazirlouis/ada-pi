"""Shared, side-effect-free helpers for Ada's OpenCode control tools."""

from __future__ import annotations

from typing import Any

import httpx

from backend import board_client

_READ_TIMEOUT = 15.0


async def card(reference: str) -> tuple[dict[str, Any] | None, str | None]:
    async with httpx.AsyncClient(timeout=_READ_TIMEOUT) as client:
        data, err, _ = await board_client.request(
            client, "GET", f"/card/{reference}?compact=0"
        )
    if err:
        return None, err
    result = (data or {}).get("card") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return None, "board response did not contain a card"
    return result, None


def action(card_data: dict[str, Any]) -> dict[str, Any]:
    value = card_data.get("action")
    return value if isinstance(value, dict) else {}


def limits(args: dict[str, Any], action_data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "repo", "model", "llm_key_alias", "profile", "target_host", "artifact",
        "max_usd", "max_turns", "max_tokens", "max_wall_time_s",
        "max_retries",
    )
    return {
        key: args[key] if args.get(key) is not None else action_data.get(key)
        for key in keys
        if args.get(key) is not None or action_data.get(key) is not None
    }


def preflight(card_data: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    action_data = action(card_data)
    resolved = limits(args, action_data)
    missing = [
        key for key in (
            "repo", "model", "llm_key_alias", "profile", "target_host", "artifact",
            "max_usd", "max_turns", "max_tokens", "max_wall_time_s",
            "max_retries",
        )
        if resolved.get(key) in (None, "")
    ]
    errors: list[str] = []
    if action_data.get("executor") not in (None, "opencode"):
        errors.append("card executor is not opencode")
    if action_data.get("status") in ("queued", "running"):
        errors.append(f"card is already {action_data['status']}")
    if resolved.get("max_usd") is not None and float(resolved["max_usd"]) <= 0:
        errors.append("max_usd must be positive")
    for key in ("max_turns", "max_tokens", "max_wall_time_s"):
        if resolved.get(key) is not None and int(resolved[key]) <= 0:
            errors.append(f"{key} must be positive")
    if resolved.get("max_retries") is not None and int(resolved["max_retries"]) < 0:
        errors.append("max_retries must be zero or greater")
    if str(resolved.get("artifact") or "").startswith("/"):
        errors.append("artifact must be repo-relative")
    return {
        "ok": not missing and not errors,
        "card_id": card_data.get("id"),
        "card_ref": card_data.get("ref"),
        "title": card_data.get("title"),
        "workflow_id": f"ada-opencode:{card_data.get('id')}:{card_data.get('updated', '')}",
        "executor": "opencode",
        "limits": resolved,
        "missing": missing,
        "errors": errors,
        "ready_for_dispatch": not missing and not errors,
        "approval_required": True,
        "mutations": [],
    }

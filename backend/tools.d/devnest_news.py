"""devnest_news — publish a verified public DevNest News article.

Ada orchestrates the event; board-api owns validation, idempotency, retry,
and the MediaWiki credential boundary. This tool never talks to MediaWiki
directly and never receives Wiki credentials.
"""

from __future__ import annotations

from typing import Any

from backend import board_client

_GEO_SCOPES = {"world", "region", "country", "subnational", "local", "project"}
_VISIBILITY = "public"

DECLARATION = {
    "name": "devnest_news",
    "description": (
        "Publish one verified, source-backed public DevNest News article "
        "through board-api into the MediaWiki News hierarchy. Use only "
        "after the user explicitly approves the exact title and body. "
        "Private or members-only content is not supported yet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Article title."},
            "body": {"type": "string", "description": "Article body."},
            "source_url": {
                "type": "string",
                "description": "Primary evidence/source URL.",
            },
            "source_published_at": {
                "type": "string",
                "description": "Exact source publication timestamp.",
            },
            "wiki_title": {
                "type": "string",
                "description": (
                    "Canonical page title under News/, e.g. "
                    "News/Desks/DevNest/Projects/Release."
                ),
            },
            "desk": {
                "type": "string",
                "description": "Editorial desk, normally DevNest.",
            },
            "geo_scope": {
                "type": "string",
                "description": (
                    "One of world, region, country, subnational, local, "
                    "project."
                ),
            },
            "topic": {"type": "string", "description": "News topic."},
            "parent": {
                "type": "string",
                "description": "Parent News/ hierarchy page.",
            },
            "country": {"type": "string"},
            "region": {"type": "string"},
        },
        "required": [
            "title", "body", "source_url", "source_published_at",
            "wiki_title", "desk", "geo_scope", "topic", "parent",
        ],
    },
}


def _text(args: dict[str, Any], key: str) -> str:
    return str(args.get(key) or "").strip()


def _validate(args: dict[str, Any]) -> str | None:
    for key in ("title", "body", "source_url", "source_published_at",
                "wiki_title", "desk", "geo_scope", "topic", "parent"):
        if not _text(args, key):
            return f"{key} is required"
    wiki_title = _text(args, "wiki_title")
    parent = _text(args, "parent")
    if not wiki_title.startswith("News/"):
        return "wiki_title must start with News/"
    if "\n" in wiki_title or "\r" in wiki_title:
        return "wiki_title must be one line"
    if not parent.startswith("News/"):
        return "parent must start with News/"
    scope = _text(args, "geo_scope").lower()
    if scope not in _GEO_SCOPES:
        return f"geo_scope must be one of {sorted(_GEO_SCOPES)}"
    if not _text(args, "source_url").startswith(("https://", "http://")):
        return "source_url must be an http(s) URL"
    return None


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    error = _validate(args)
    if error:
        return {"ok": False, "error": error}

    event = {
        "kind": "news",
        "title": _text(args, "title"),
        "body": _text(args, "body"),
        "source": "ada",
        "confidence": "verified",
        "visibility": _VISIBILITY,
        "evidence": [_text(args, "source_url")],
        "sinks": ["wiki"],
        "meta": {
            "wiki_title": _text(args, "wiki_title"),
            "status": "published",
            "desk": _text(args, "desk"),
            "geo_scope": _text(args, "geo_scope").lower(),
            "topic": _text(args, "topic"),
            "parent": _text(args, "parent"),
            "source_published_at": _text(args, "source_published_at"),
            "source_name": _text(args, "desk"),
        },
    }
    for key in ("country", "region"):
        value = _text(args, key)
        if value:
            event["meta"][key] = value

    status, result = await board_client.post("/publish", event, timeout=45)
    if status <= 0:
        return {"ok": False, "error": result.get("error", "board unavailable")}
    if status >= 400 or not result.get("ok", True):
        return {"ok": False, "error": result.get("error", result)}
    receipt = result.get("receipts", {}).get("wiki", {})
    if receipt.get("status") != "delivered":
        return {
            "ok": False,
            "error": f"Wiki publication was not delivered: {receipt}",
            "publication": result,
        }
    return {
        "ok": True,
        "status": "published",
        "title": event["title"],
        "wiki_title": event["meta"]["wiki_title"],
        "receipt": receipt,
        "event_id": result.get("event_id"),
        "note": "Published through board-api; Ada did not hold Wiki credentials.",
    }

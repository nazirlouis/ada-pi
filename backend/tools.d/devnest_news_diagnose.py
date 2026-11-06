"""Read-only diagnosis for a protected News promotion."""

from __future__ import annotations

from typing import Any

from backend import board_client


DECLARATION = {
    "name": "devnest_news_diagnose",
    "description": (
        "Diagnose a protected DevNest News promotion without publishing. "
        "Checks the private revision, public target, review revision, and "
        "board-card authorization, then reports the exact blocking gate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "private_wiki_title": {"type": "string"},
            "public_wiki_title": {"type": "string"},
            "review_revid": {"type": "integer"},
            "card_ref": {"type": "string"},
        },
        "required": ["private_wiki_title", "public_wiki_title"],
    },
}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    private_title = str(args.get("private_wiki_title") or "").strip()
    public_title = str(args.get("public_wiki_title") or "").strip()
    if not private_title.startswith("PrivateNews:"):
        return {"ok": False, "error": "private_wiki_title must start with PrivateNews:"}
    if not public_title.startswith("News/"):
        return {"ok": False, "error": "public_wiki_title must start with News/"}
    body = {
        "private_wiki_title": private_title,
        "public_wiki_title": public_title,
        "from": "ada",
    }
    for key in ("review_revid", "card_ref"):
        if args.get(key) not in (None, ""):
            body[key] = args[key]
    status, result = await board_client.post(
        "/wiki/news-diagnose", body, timeout=60)
    if status <= 0:
        return {"ok": False, "error": result.get("error", "board unavailable")}
    if status >= 400 or result.get("ok") is False:
        return {"ok": False, "error": result.get("error", result)}
    return {"ok": True, **result}

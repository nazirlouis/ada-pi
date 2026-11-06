"""Read one protected PrivateNews page for Ada's editorial review.

Ada receives the page through board-api; Wiki credentials remain on the
board-api service host. Promotion is a separate, confirmation-gated call to
devnest_news.
"""

from __future__ import annotations

from typing import Any

from backend import board_client


DECLARATION = {
    "name": "devnest_news_review",
    "description": (
        "Read one protected PrivateNews review page through board-api. "
        "Use the returned source-backed draft to assess factual accuracy, "
        "metadata, evidence, and readiness for explicit promotion. This "
        "tool is read-only and never publishes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_title": {
                "type": "string",
                "description": (
                    "Protected page under PrivateNews:, for example "
                    "PrivateNews:Countries/Thailand/Story."
                ),
            },
        },
        "required": ["wiki_title"],
    },
}


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    title = str(args.get("wiki_title") or "").strip()
    if not title.startswith("PrivateNews:"):
        return {"ok": False, "error": "wiki_title must start with PrivateNews:"}
    status, result = await board_client.post(
        "/wiki/news-review", {"wiki_title": title, "from": "ada"},
        timeout=45,
    )
    if status <= 0:
        return {"ok": False, "error": result.get("error", "board unavailable")}
    if status >= 400 or result.get("ok") is False:
        return {"ok": False, "error": result.get("error", result)}
    return {"ok": True, **result}

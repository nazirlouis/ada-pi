"""ada_lesson — file or update an Education lesson via board-api /lesson.

Tony's "save this to education" voice path. The WP lesson CPT upserts
by topic slug, so re-filing the same topic updates in place and Thai
translation is automatic server-side. Confirmed policy — Ada should say
what she is about to file before the gate fires; owner-only so guests
cannot write lessons.
"""
from __future__ import annotations
from typing import Any
from backend import board_client

DECLARATION = {
    "name": "ada_lesson",
    "description": (
        "Save knowledge to Education (the WP lesson CPT) through "
        "board-api. Use when Tony asks to remember, save, or write up a "
        "topic — re-filing the same topic updates the existing lesson in "
        "place. content should be concise HTML (<h2>/<ul>/<code> allowed)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Lesson topic slug text, e.g. docker-volumes-basics.",
            },
            "content": {
                "type": "string",
                "description": "Lesson body as concise HTML.",
            },
            "level": {
                "type": "string",
                "description": "intro, working, or deep (default intro).",
            },
        },
        "required": ["topic", "content"],
    },
}

async def run(runner: Any, **args: Any) -> dict[str, Any]:
    topic = str(args.get("topic") or "").strip()
    content = str(args.get("content") or "").strip()
    if not topic or not content:
        return {"ok": False, "error": "topic and content are required"}
    if len(topic) > 200 or len(content) > 12000:
        return {"ok": False, "error": "topic/content exceed limits"}
    level = str(args.get("level") or "").strip().lower()
    body: dict[str, Any] = {"topic": topic, "content": content,
                            "from": "ada", "source": "voice"}
    if level in ("intro", "working", "deep"):
        body["level"] = level
    status, result = await board_client.post("/lesson", body, timeout=60)
    if status <= 0:
        return {"ok": False, "error": result.get("error", "board unavailable")}
    if status >= 400:
        return {"ok": False, "error": result.get("error", f"board-api {status}")}
    return {"ok": True, **result}

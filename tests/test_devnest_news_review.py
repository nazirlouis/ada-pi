"""Tests for the read-only protected News review tool."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

TOOL_PATH = (Path(__file__).resolve().parent.parent
             / "backend" / "tools.d" / "devnest_news_review.py")


def _load():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tools_d.devnest_news_review_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_reads_private_page_without_publishing():
    tool = _load()
    response = {
        "ok": True,
        "wiki_title": "PrivateNews:Countries/Thailand/Test",
        "revid": 45,
        "wikitext": "draft",
    }
    with patch.object(
        tool.board_client, "post",
        new=AsyncMock(return_value=(200, response)),
    ) as post:
        result = await tool.run(
            SimpleNamespace(),
            wiki_title="PrivateNews:Countries/Thailand/Test",
        )
    assert result["ok"] is True
    assert result["revid"] == 45
    post.assert_awaited_once_with(
        "/wiki/news-review",
        {"wiki_title": "PrivateNews:Countries/Thailand/Test", "from": "ada"},
        timeout=45,
    )


@pytest.mark.asyncio
async def test_rejects_public_title_before_write():
    tool = _load()
    with patch.object(tool.board_client, "post",
                      new=AsyncMock()) as post:
        result = await tool.run(
            SimpleNamespace(), wiki_title="News/Countries/Thailand/Test")
    assert result["ok"] is False
    assert "PrivateNews:" in result["error"]
    post.assert_not_awaited()

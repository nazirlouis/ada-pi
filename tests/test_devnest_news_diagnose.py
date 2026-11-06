"""Tests for the read-only News promotion diagnosis tool."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

TOOL_PATH = (Path(__file__).resolve().parent.parent
             / "backend" / "tools.d" / "devnest_news_diagnose.py")


def _load():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tools_d.devnest_news_diagnose_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_returns_board_diagnosis_without_publishing():
    tool = _load()
    response = {
        "ok": True,
        "ready": False,
        "diagnosis_id": "diag-123",
        "checks": [{"id": "card_authorization", "status": "blocked"}],
    }
    with patch.object(tool.board_client, "post",
                      new=AsyncMock(return_value=(200, response))) as post:
        result = await tool.run(
            SimpleNamespace(),
            private_wiki_title="PrivateNews:Countries/Thailand/Test",
            public_wiki_title="News/Countries/Thailand/Test",
            review_revid=45,
        )
    assert result["ok"] is True
    assert result["ready"] is False
    post.assert_awaited_once()
    assert post.await_args.args[0] == "/wiki/news-diagnose"


@pytest.mark.asyncio
async def test_rejects_wrong_namespaces_before_write():
    tool = _load()
    with patch.object(tool.board_client, "post",
                      new=AsyncMock()) as post:
        result = await tool.run(
            SimpleNamespace(),
            private_wiki_title="News/Private/Test",
            public_wiki_title="News/Test",
        )
    assert result["ok"] is False
    assert "PrivateNews:" in result["error"]
    post.assert_not_awaited()

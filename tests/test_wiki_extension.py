"""Tests for the allowlisted, board-api-backed Wiki extension tool."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

TOOL_PATH = (Path(__file__).resolve().parent.parent
             / "backend" / "tools.d" / "wiki_extension.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "tools_d.wiki_extension_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_list_calls_board():
    tool = _load()
    with patch.object(tool.board_client, "post",
                      new=AsyncMock(return_value=(200, {"installed": []}))) as post:
        result = await tool.run(SimpleNamespace(), action="list")
    assert result["ok"] is True
    post.assert_awaited_once_with("/wiki/extension",
                                  {"action": "list", "from": "ada"},
                                  timeout=90)


@pytest.mark.asyncio
async def test_rejects_unapproved_name_without_write():
    tool = _load()
    with patch.object(tool.board_client, "post", new=AsyncMock()) as post:
        result = await tool.run(SimpleNamespace(), action="install",
                                name="ArbitraryRemoteExtension")
    assert result["ok"] is False
    assert "not approved" in result["error"]
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_install_receipt():
    tool = _load()
    with patch.object(
        tool.board_client, "post",
        new=AsyncMock(return_value=(
            200, {"action": "install", "name": "CategoryTree",
                  "verified": True, "after": ["CategoryTree"]}))
    ) as post:
        result = await tool.run(SimpleNamespace(), action="install",
                                name="CategoryTree")
    assert result["ok"] is True
    assert result["verified"] is True
    post.assert_awaited_once_with(
        "/wiki/extension",
        {"action": "install", "name": "CategoryTree", "from": "ada"},
        timeout=90)

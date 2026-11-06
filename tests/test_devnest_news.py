"""Tests for the confirm-gated DevNest News board-api adapter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

TOOL_PATH = (Path(__file__).resolve().parent.parent
             / "backend" / "tools.d" / "devnest_news.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "tools_d.devnest_news_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**overrides):
    args = {
        "title": "DevNest News path is live",
        "body": "The verified publication path now reaches MediaWiki.",
        "summary": "Verified DevNest to MediaWiki publishing.",
        "source_url": "https://wiki.h3.gizmo-thailand.com/",
        "source_published_at": "2026-11-04 00:00 UTC",
        "wiki_title": "News/Desks/DevNest/Projects/MediaWiki-Adapter",
        "desk": "DevNest",
        "geo_scope": "project",
        "topic": "Publishing",
        "parent": "News/Desks/DevNest",
    }
    args.update(overrides)
    return args


@pytest.mark.asyncio
async def test_publishes_only_after_board_receipt():
    tool = _load()
    response = {
        "ok": True,
        "event_id": "evt-real",
        "receipts": {
            "wiki": {
                "status": "delivered",
                "ref": "wiki:News/Desks/DevNest/Projects/MediaWiki-Adapter",
                "pageid": 22,
                "revid": 28,
            }
        },
    }
    with patch.object(tool.board_client, "post",
                      new=AsyncMock(return_value=(200, response))) as post:
        result = await tool.run(SimpleNamespace(), **_args())
    assert result["ok"] is True
    assert result["receipt"]["revid"] == 28
    post.assert_awaited_once()
    assert post.await_args.args[0] == "/publish"
    assert post.await_args.args[1]["sinks"] == ["wiki"]


@pytest.mark.asyncio
async def test_rejects_invalid_scope_before_write():
    tool = _load()
    with patch.object(tool.board_client, "post",
                      new=AsyncMock()) as post:
        result = await tool.run(SimpleNamespace(),
                                **_args(geo_scope="moon"))
    assert result["ok"] is False
    assert "geo_scope" in result["error"]
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_reports_delivery_failure_honestly():
    tool = _load()
    response = {
        "ok": True,
        "receipts": {"wiki": {"status": "queued", "reason": "offline"}},
    }
    with patch.object(tool.board_client, "post",
                      new=AsyncMock(return_value=(200, response))):
        result = await tool.run(SimpleNamespace(), **_args())
    assert result["ok"] is False
    assert "not delivered" in result["error"]


@pytest.mark.asyncio
async def test_promotion_carries_diagnosis_and_card_receipts():
    tool = _load()
    response = {
        "ok": True,
        "event_id": "evt-promote",
        "receipts": {
            "wiki": {
                "status": "delivered",
                "revid": 46,
            }
        },
    }
    with patch.object(tool.board_client, "post",
                      new=AsyncMock(return_value=(200, response))) as post:
        result = await tool.run(
            SimpleNamespace(),
            **_args(
                private_wiki_title="PrivateNews:Countries/Thailand/Test",
                review_revid=45,
                diagnosis_id="diag-123",
                card_ref="devnest-news-promote-test",
            ),
        )
    assert result["ok"] is True
    event = post.await_args.args[1]
    assert event["card"] == "devnest-news-promote-test"
    assert event["meta"]["diagnosis_id"] == "diag-123"
    assert event["meta"]["review_revid"] == 45

"""Unit tests for opencode/ada_mcp.py find_card natural-language resolution."""
import importlib.util
import sys
import types
from pathlib import Path


def _load_ada_mcp():
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    fastmcp_mod.FastMCP = FastMCP
    sys.modules.setdefault("mcp", mcp_mod)
    sys.modules.setdefault("mcp.server", server_mod)
    sys.modules.setdefault("mcp.server.fastmcp", fastmcp_mod)

    path = Path(__file__).resolve().parents[1] / "opencode" / "ada_mcp.py"
    spec = importlib.util.spec_from_file_location("ada_mcp_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ada_mcp = _load_ada_mcp()

CARDS = [
    {
        "id": "unified-media-pack-layout",
        "ref": "MED-1032",
        "title": "Define unified durable media pack layout and local-cache contract",
        "brief": "Generalize the Nest artifact structure",
        "spec": "Verify a Hibiscus round-trip; Gaussian, image, video, audio, document jobs",
    },
    {
        "id": "kornia-mcp-gs-evaluator",
        "ref": "OPS-1055",
        "title": "Add read-only Kornia MCP tools for Gaussian capture evaluation",
        "brief": "Kornia evaluator for capture sets",
        "spec": "Start offline/read-only on the hibiscus fixture",
    },
    {
        "id": "gaussian-splat-edu-smoke",
        "ref": "OPS-0866",
        "title": "T4 Gaussian splat education smoke pack",
        "brief": "Smallest real Gaussian splat renderer proof",
        "spec": "Synthetic colored shell orbit on gsplat CUDA",
    },
]


def _with_cards(monkeypatch):
    monkeypatch.setattr(ada_mcp, "_cards", lambda: CARDS)


def test_find_card_exact_ref_and_id(monkeypatch):
    _with_cards(monkeypatch)
    assert ada_mcp.find_card("MED-1032")["id"] == "unified-media-pack-layout"
    assert ada_mcp.find_card("unified-media-pack-layout")["ref"] == "MED-1032"


def test_find_card_title_substring(monkeypatch):
    _with_cards(monkeypatch)
    card = ada_mcp.find_card("media pack layout")
    assert card["ref"] == "MED-1032"


def test_find_card_unique_token_match(monkeypatch):
    _with_cards(monkeypatch)
    assert ada_mcp.find_card("kornia")["ref"] == "OPS-1055"
    assert ada_mcp.find_card("gaussian splat smoke")["ref"] == "OPS-0866"


def test_find_card_ambiguous_lists_candidates(monkeypatch):
    _with_cards(monkeypatch)
    try:
        ada_mcp.find_card("hibiscus")
    except LookupError as exc:
        assert "matched 2 cards" in str(exc)
        assert "MED-1032" in str(exc)
        assert "OPS-1055" in str(exc)
    else:
        raise AssertionError("expected LookupError for ambiguous reference")


def test_find_card_no_match(monkeypatch):
    _with_cards(monkeypatch)
    try:
        ada_mcp.find_card("nonexistent-thing")
    except LookupError as exc:
        assert "no board card matched" in str(exc)
    else:
        raise AssertionError("expected LookupError for unknown reference")


def test_find_card_alias(monkeypatch):
    cards = [dict(CARDS[0], aliases=["media pack", "nest pack contract"])]
    monkeypatch.setattr(ada_mcp, "_cards", lambda: cards)
    assert ada_mcp.find_card("nest pack contract")["ref"] == "MED-1032"

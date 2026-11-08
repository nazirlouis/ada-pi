#!/usr/bin/env python3
"""Read-only Ada control-plane MCP bridge for the DevNest OpenCode project."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

BOARD_API = os.environ.get(
    "ADA_BOARD_API",
    "https://idc03.taila0626a.ts.net:10000",
).rstrip("/")
MDDB_CMS = os.environ.get("ADA_MDDB_CMS_URL", "http://100.102.134.91:11027/v1").rstrip("/")
MDDB_MEMORY = os.environ.get(
    "ADA_MEMORY_MDDB_URL",
    "http://100.102.134.91:11023/v1",
).rstrip("/")
WORKSPACE = Path.cwd()
ADOBE_STOCK_INDEX_FILES = (
    "campaigns.json",
    "assets.json",
    "rejection-patterns.json",
    "prompt-patterns.json",
)

mcp = FastMCP("ada-control")
PENDING_NEXT_SELECTIONS: dict[str, tuple[str, float]] = {}

RISK_TERMS = (
    "deploy", "deployment", "production", "credential", "secret", "delete", "destructive",
    "database", "install", "systemd", "service", "origin", "restore",
    "timer", "rollout", "video", "render", "gpu", "mlx", "media",
        "rerender", "prune", "compact", "wp", "wordpress", "upsert",
)
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|do\s+not|does\s+not|must\s+not|"
    r"blocked|prohibited|forbidden)\b[^.!?]{0,80}$",
    re.I,
)


def _positive_risk_terms(text: str) -> list[str]:
    """Find affirmative risk terms while ignoring nearby prohibitions."""
    found = []
    lower = text.lower()
    for term in RISK_TERMS:
        for match in re.finditer(rf"\b{re.escape(term)}\b", lower):
            prefix = lower[max(0, match.start() - 100):match.start()]
            if not NEGATION_RE.search(prefix):
                found.append(term)
                break
    return found


def _selection_gate(card_reference: str, selection_token: str) -> str | None:
    """Bind a next-card plan to a card that was actually offered in the menu."""
    now = time.monotonic()
    for ref, (_, expires) in list(PENDING_NEXT_SELECTIONS.items()):
        if expires <= now:
            PENDING_NEXT_SELECTIONS.pop(ref, None)
    ref = card_reference.strip().lower()
    pending = PENDING_NEXT_SELECTIONS.get(ref)
    if pending and selection_token != pending[0]:
        return "selection_token_required"
    if PENDING_NEXT_SELECTIONS and not pending:
        return "selection_not_in_current_menu"
    if pending:
        PENDING_NEXT_SELECTIONS.pop(ref, None)
    return None


def _get_json(path: str) -> Any:
    req = urllib.request.Request(f"{BOARD_API}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def _post_json(path: str, payload: dict[str, Any], timeout: float = 3) -> Any:
    req = urllib.request.Request(
        f"{BOARD_API}{path}",
        data=json.dumps(payload).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _mddb_post(path: str, payload: dict[str, Any], timeout: float = 5) -> Any:
    req = urllib.request.Request(
        f"{MDDB_CMS}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _memory_mddb_post(path: str, payload: dict[str, Any], timeout: float = 5) -> Any:
    req = urllib.request.Request(
        f"{MDDB_MEMORY}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _cards() -> list[dict[str, Any]]:
    payload = _get_json("/cards")
    if not isinstance(payload, dict) or not isinstance(payload.get("cards"), list):
        raise RuntimeError("board response did not contain a cards list")
    return [c for c in payload["cards"] if isinstance(c, dict)]


def _terms(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
    terms = set(raw)
    for value in raw:
        terms.update(part for part in re.split(r"[-_]", value) if len(part) >= 3)
    return terms


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("&nbsp;", " ")


def _lesson_memory(query: str, deadline: float, limit: int = 3) -> list[dict[str, Any]]:
    remaining = max(0.25, min(1.5, deadline - time.monotonic()))
    payload = _post_json("/lesson", {"op": "list", "from": "devin"}, timeout=remaining)
    rows = payload.get("lessons", []) if isinstance(payload, dict) else []
    q = _terms(query)
    ranked = []
    for row in rows:
        text = " ".join(str(row.get(k) or "") for k in ("topic", "slug", "level"))
        score = len(q & _terms(text))
        if score:
            ranked.append((score, row))
    hits = []
    for score, row in sorted(ranked, key=lambda item: (-item[0], item[1].get("slug", "")))[:limit]:
        if time.monotonic() >= deadline:
            break
        try:
            remaining = max(0.25, min(0.75, deadline - time.monotonic()))
            detail = _post_json(
                "/lesson",
                {"op": "get", "slug": row["slug"], "from": "devin"},
                timeout=remaining,
            )
            body = _strip_html(detail.get("content", ""))
        except Exception as exc:
            body = f"lesson read failed: {exc}"
        hits.append({"source": "education", "slug": row.get("slug"), "topic": row.get("topic"), "score": score, "excerpt": body[:1200]})
    return hits


def _devin_memory(query: str, deadline: float, limit: int = 3) -> list[dict[str, Any]]:
    roots = [Path(os.environ.get("ADA_DEVIN_SUMMARIES", Path.home() / ".local/share/devin/summaries")), Path.home() / ".local/share/devin/cli/summaries"]
    q = _terms(query)
    ranked = []
    for root in roots:
        if not root.exists():
            continue
        paths = sorted(
            root.glob("*.md"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )[:30]
        for path in paths:
            if time.monotonic() >= deadline:
                break
            try:
                body = path.read_text(errors="replace")[:12000]
            except OSError:
                continue
            score = len(q & _terms(body[:12000]))
            if score:
                ranked.append((score, path, body))
    return [{"source": "devin-session", "path": str(path), "score": score, "excerpt": body[:1200]}
            for score, path, body in sorted(ranked, key=lambda item: (-item[0], str(item[1])))[:limit]]


def _devin_mddb_memory(query: str, deadline: float, limit: int = 3) -> list[dict[str, Any]]:
    """Read normalized Devin summaries from the canonical MDDB collection."""
    remaining = max(0.25, min(2.0, deadline - time.monotonic()))
    payload = _memory_mddb_post(
        "search",
        {
            "collection": "ada-ha-bank-devin-tony",
            "query": query,
            "limit": limit,
        },
        timeout=remaining,
    )
    if not isinstance(payload, list):
        return []
    hits = []
    for row in payload[:limit]:
        if not isinstance(row, dict):
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        hits.append({
            "source": "devin-session",
            "storage": "mddb",
            "key": row.get("key"),
            "score": row.get("score"),
            "card_id": (meta.get("card_id") or [None])[0]
            if isinstance(meta.get("card_id"), list) else meta.get("card_id"),
            "workflow_id": (meta.get("workflow_id") or [None])[0]
            if isinstance(meta.get("workflow_id"), list) else meta.get("workflow_id"),
            "job_id": (meta.get("job_id") or [None])[0]
            if isinstance(meta.get("job_id"), list) else meta.get("job_id"),
            "freshness": meta.get("date") or meta.get("updated_at"),
            "excerpt": str(row.get("contentMd") or row.get("content") or "")[:1200],
        })
    return hits


def recall_memory(query: str) -> dict[str, Any]:
    """Read Education and canonical Devin summaries with local fallback."""
    errors = []
    deadline = time.monotonic() + 4.5
    try:
        lessons = _lesson_memory(query, deadline)
    except Exception as exc:
        lessons = []
        errors.append(f"education:{exc}")
    try:
        sessions = _devin_mddb_memory(query, deadline)
        if len(sessions) < 3 and time.monotonic() < deadline:
            sessions.extend(_devin_memory(query, deadline, limit=3 - len(sessions)))
    except Exception as exc:
        errors.append(f"devin-mddb:{exc}")
        try:
            sessions = _devin_memory(query, deadline)
        except Exception as local_exc:
            sessions = []
            errors.append(f"devin-session:{local_exc}")
    return {"query": query, "lessons": lessons, "devin_sessions": sessions, "errors": errors}


def _adobe_stock_index_root() -> Path:
    """Return the bounded workspace or read-only GDrive index directory."""
    default_gdrive = Path("/data/gdrive/ada-nest-adobe-stock/indexes")
    default_workspace = WORKSPACE / ".adobe-stock" / "indexes"
    configured = os.environ.get(
        "ADA_ADOBE_STOCK_INDEX_ROOT",
        str(default_gdrive if default_gdrive.is_dir() else default_workspace),
    )
    root = Path(configured).expanduser().resolve()
    workspace = WORKSPACE.resolve()
    approved_gdrive_root = Path(
        "/data/gdrive/ada-nest-adobe-stock/indexes"
    ).resolve()
    approved = (
        root == workspace
        or workspace in root.parents
        or root == approved_gdrive_root
        or approved_gdrive_root in root.parents
    )
    if not approved:
        raise ValueError(
            "Adobe Stock index root must be inside the workspace or "
            "the read-only ada-nest GDrive mount"
        )
    return root


def _read_adobe_stock_indexes() -> tuple[dict[str, list[Any]], list[str]]:
    """Read only the fixed JSON index files; never searches arbitrary paths."""
    root = _adobe_stock_index_root()
    indexes: dict[str, list[Any]] = {}
    missing: list[str] = []
    for filename in ADOBE_STOCK_INDEX_FILES:
        path = root / filename
        key = filename.removesuffix(".json")
        if not path.is_file():
            missing.append(str(path))
            indexes[key] = []
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list or an items list")
        indexes[key] = payload
    return indexes, missing


def _rank_adobe_items(items: list[Any], query: str, limit: int = 10) -> list[dict[str, Any]]:
    terms = _terms(query)
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(value) for value in item.values())
        score = len(terms & _terms(text))
        if score:
            ranked.append((score, item))
    return [
        {"score": score, **item}
        for score, item in sorted(ranked, key=lambda value: (-value[0], str(value[1].get("id", ""))))[:limit]
    ]


def _asset_preview(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize only explicitly exported, browser-safe preview fields."""
    preview_url = str(item.get("preview_url") or "").strip()
    thumbnail_url = str(item.get("thumbnail_url") or "").strip()
    valid_urls: dict[str, str | None] = {
        "preview_url": None,
        "thumbnail_url": None,
    }
    invalid_fields: list[str] = []
    for field, value in (("preview_url", preview_url), ("thumbnail_url", thumbnail_url)):
        if not value:
            continue
        try:
            parsed = urllib.parse.urlparse(value)
        except ValueError:
            parsed = None
        if (
            parsed
            and parsed.scheme in {"http", "https"}
            and parsed.netloc
            and not parsed.username
            and not parsed.password
        ):
            valid_urls[field] = value
        else:
            invalid_fields.append(field)

    warnings = [
        f"{field} is not a browser-safe http(s) URL"
        for field in invalid_fields
    ]
    if valid_urls["preview_url"] or valid_urls["thumbnail_url"]:
        status = "available" if valid_urls["preview_url"] else "thumbnail-only"
        reason = None
    elif invalid_fields:
        status = "unavailable"
        reason = "preview URL must be an http(s) URL without embedded credentials"
    else:
        status = "unavailable"
        reason = "no exported preview URL"

    return {
        "status": status,
        "preview_url": valid_urls["preview_url"],
        "thumbnail_url": valid_urls["thumbnail_url"],
        "source_ref": item.get("source_ref"),
        "reason": reason,
        "warnings": warnings,
    }


def _markdown_text(value: Any) -> str:
    """Keep exported asset labels safe when inserted into Markdown."""
    return str(value or "Untitled asset").replace("\\", "\\\\").replace(
        "[", "\\[").replace("]", "\\]")


def _preview_markdown(assets: list[dict[str, Any]]) -> str:
    """Build an OpenChamber-friendly preview block from normalized assets."""
    lines = ["### Adobe Stock previews", ""]
    if not assets:
        return "### Adobe Stock previews\n\nNo matching assets."
    for asset in assets:
        title = _markdown_text(asset.get("title") or asset.get("id"))
        preview = asset["preview"]
        lines.append(f"**{title}** (`{asset.get('id', 'unknown')}`)")
        if preview["thumbnail_url"]:
            lines.append(
                f"![{title}](<{preview['thumbnail_url']}>)"
            )
        if preview["preview_url"]:
            lines.append(
                f"[Open full preview](<{preview['preview_url']}>)"
            )
        if preview["status"] == "unavailable":
            lines.append(f"> Preview unavailable: {preview['reason']}.")
        elif preview["warnings"]:
            lines.append(f"> Preview warning: {'; '.join(preview['warnings'])}.")
        if preview["source_ref"]:
            lines.append(f"Source: `{preview['source_ref']}`")
        lines.append("")
    return "\n".join(lines).rstrip()


def _preview_html(assets: list[dict[str, Any]], campaign_idea: str) -> str:
    """Build a self-contained HTML preview page for OpenChamber Preview/browser."""
    import html as _html

    cards = []
    if not assets:
        cards.append('<p class="empty">No matching assets.</p>')
    for asset in assets:
        title = _html.escape(str(asset.get("title") or asset.get("id") or "untitled"))
        aid = _html.escape(str(asset.get("id") or "unknown"))
        preview = asset["preview"]
        thumb = preview.get("thumbnail_url") or preview.get("preview_url")
        img = (
            f'<a href="{_html.escape(preview["preview_url"] or preview["thumbnail_url"] or "#", quote=True)}" target="_blank" rel="noopener">'
            f'<img class="thumb" src="{_html.escape(thumb, quote=True)}" alt="{title}" loading="lazy"></a>'
            if thumb
            else '<div class="noimg">preview unavailable</div>'
        )
        links = ""
        if preview.get("preview_url"):
            links += (
                f'<a class="full" href="{_html.escape(preview["preview_url"], quote=True)}" '
                'target="_blank" rel="noopener">Open full preview</a>'
            )
        status = _html.escape(str(preview.get("status") or ""))
        warnings = "".join(
            f'<p class="warn">⚠ {_html.escape(w)}</p>'
            for w in preview.get("warnings") or []
        )
        if preview.get("status") == "unavailable":
            warnings += f'<p class="warn">Preview unavailable: {_html.escape(str(preview.get("reason") or ""))}</p>'
        source = _html.escape(str(preview.get("source_ref") or ""))
        tags = ", ".join(_html.escape(str(t)) for t in (asset.get("tags") or [])[:8])
        cards.append(
            f'<section class="card">{img}'
            f'<h3>{title}</h3><p class="meta"><code>{aid}</code> · {status}'
            f'{(" · " + tags) if tags else ""}</p>{links}{warnings}'
            f'<p class="src">Source: <code>{source}</code></p></section>'
        )
    idea = _html.escape(campaign_idea)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Adobe Stock previews — {idea}</title><style>"
        "body{font-family:system-ui,sans-serif;margin:24px;background:#0f1115;color:#e8e8e8}"
        "h1{font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}"
        ".card{background:#1a1d24;border:1px solid #2a2f3a;border-radius:10px;padding:12px}"
        ".thumb{width:100%;height:140px;object-fit:cover;border-radius:6px;background:#000;display:block}"
        ".noimg{height:140px;display:flex;align-items:center;justify-content:center;background:#000;border-radius:6px;color:#777}"
        "h3{font-size:14px;margin:10px 0 4px}.meta{font-size:12px;color:#9aa}.full{font-size:13px}"
        ".warn{color:#f0b429;font-size:12px}.src{font-size:11px;color:#667}.empty{color:#888}"
        "a{color:#7db4ff;text-decoration:none}code{background:#000;padding:1px 4px;border-radius:3px}"
        "</style></head><body>"
        f"<h1>Adobe Stock previews — {_html.escape(campaign_idea)}</h1>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )


def _write_preview_html(html: str, campaign_idea: str) -> Path:
    """Persist the preview page under the bounded workspace previews dir."""
    import re as _re

    slug = _re.sub(r"[^a-z0-9]+", "-", campaign_idea.lower()).strip("-")[:48] or "preview"
    out_dir = WORKSPACE / ".adobe-stock" / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_adobe_stock_reuse_report(
    campaign_idea: str,
    *,
    cards: list[dict[str, Any]] | None = None,
    memory: dict[str, Any] | None = None,
    indexes: dict[str, list[Any]] | None = None,
    missing_indexes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a read-only, source-bound Adobe Stock reuse report."""
    idea = campaign_idea.strip()
    if not idea:
        raise ValueError("campaign_idea is required")
    indexes = indexes if indexes is not None else _read_adobe_stock_indexes()[0]
    if missing_indexes is None:
        _, missing_indexes = _read_adobe_stock_indexes()
    cards = cards if cards is not None else _cards()
    memory = memory if memory is not None else recall_memory(idea)
    related_cards = _rank_adobe_items(cards, idea, limit=10)
    lessons = memory.get("lessons", []) if isinstance(memory, dict) else []
    sessions = memory.get("devin_sessions", []) if isinstance(memory, dict) else []
    reusable_assets = _rank_adobe_items(indexes.get("assets", []), idea)
    for asset in reusable_assets:
        asset["preview"] = _asset_preview(asset)
    preview_statuses = [asset["preview"]["status"] for asset in reusable_assets]
    return {
        "artifact_type": "adobe_stock_reuse_report",
        "schema_version": 1,
        "status": "draft",
        "campaign_idea": idea,
        "related_cards": related_cards,
        "related_lessons": lessons[:10],
        "related_dev_sessions": sessions[:10],
        "related_campaigns": _rank_adobe_items(indexes.get("campaigns", []), idea),
        "reusable_assets": reusable_assets,
        "preview_summary": {
            "assets_returned": len(reusable_assets),
            "available": preview_statuses.count("available"),
            "thumbnail_only": preview_statuses.count("thumbnail-only"),
            "unavailable": preview_statuses.count("unavailable"),
        },
        "openchamber_preview_markdown": _preview_markdown(reusable_assets),
        "reusable_prompt_patterns": _rank_adobe_items(indexes.get("prompt-patterns", []), idea),
        "prior_rejection_patterns": _rank_adobe_items(indexes.get("rejection-patterns", []), idea),
        "facts": ["Every related item includes a source record from the bounded inputs."],
        "assumptions": [],
        "open_questions": [
            "Which generator and target batch size should this campaign use?",
            "Which related items should be reused?",
        ],
        "index_warnings": [f"missing index: {path}" for path in (missing_indexes or [])],
        "mutations_performed": [],
        "requires_native_followup": True,
        "next_menu": ["Create draft", "Revise", "Stop"],
    }


def verification_code(evidence: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:12].upper()
    return f"ADA-VFY-{digest}"


def _backlog_cards() -> list[dict[str, Any]]:
    fields = "id,title,brief,spec,column,priority,ref,updated,action,requests"
    query = urllib.parse.urlencode({
        "column": "backlog",
        "fields": fields,
        "limit": "1000",
    })
    payload = _get_json(f"/cards?{query}")
    if not isinstance(payload, dict) or not isinstance(payload.get("cards"), list):
        raise RuntimeError("board response did not contain projected cards")
    return [c for c in payload["cards"] if isinstance(c, dict)]


def _rank_next_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank safe, actionable backlog cards without mutating board state."""
    priority_points = {"high": 30, "medium": 20, "low": 10}
    safe_terms = ("test", "docs", "schema", "readonly", "read-only", "report", "opencode", "ada")
    ranked = []
    for card in cards:
        action = card.get("action") or {}
        status = str(action.get("status") or "").lower()
        if status in {"done", "running", "claimed", "idle", "held", "failed"}:
            continue
        if str(card.get("column") or "").lower() != "backlog":
            continue
        if not (card.get("brief") or card.get("spec")):
            continue
        card_id = str(card.get("id") or "")
        card_ref = card.get("ref")
        if not card_ref:
            continue
        title = str(card.get("title") or "")
        if title.lower().startswith("scenario fixture"):
            continue
        # Automated health/CMS cards need triage, not direct Ada/OpenCode
        # execution. Require a durable human-facing ref before recommending.
        if not card_ref and (
            card_id.startswith("cms-auto-") or card_id.startswith("audit-auto-")
        ):
            continue
        text = " ".join(str(card.get(k) or "") for k in ("title", "brief", "spec")).lower()
        score = priority_points.get(str(card.get("priority") or "").lower(), 0)
        reasons = []
        blockers = []
        if score:
            reasons.append(f"priority={card.get('priority')}")
        if len(str(card.get("brief") or card.get("spec") or "")) >= 80:
            score += 15
            reasons.append("self-contained brief")
        if any(term in text for term in ("test", "verify", "schema", "docs")):
            score += 15
            reasons.append("checkable artifact/test shape")
        if any(term in text for term in safe_terms):
            score += 5
            reasons.append("fits bounded Ada/OpenCode lane")
        if _positive_risk_terms(text):
            score -= 35
            blockers.append("risk keyword requires human/policy gate")
        if any(term in text for term in ("fixed", "completed", "closed", "already resolved")):
            score -= 50
            blockers.append("title suggests the work may already be resolved; verify before selecting")
        if card.get("requests"):
            score -= 25
            blockers.append("open decision/request")
        if str(action.get("type") or "").lower() == "manual":
            score -= 15
            blockers.append("manual action, not dispatchable")
        if not card.get("brief"):
            score -= 10
            blockers.append("thin brief")
        ranked.append({
            "id": card_id,
            "ref": card_ref,
            "selection_value": card_ref or card_id,
            "title": card.get("title"),
            "priority": card.get("priority"),
            "score": score,
            "reasons": reasons,
            "blockers": blockers,
            "recommended_command": f"/ada preflight {card_ref or card_id}",
        })
    ordered = sorted(ranked, key=lambda x: (-x["score"], str(x.get("ref") or x["id"])))
    safe = [candidate for candidate in ordered if not candidate["blockers"]]
    return (safe or ordered)[:5]


_CARD_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "no", "not", "of", "on", "or", "our", "please", "should",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "too", "up", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "about", "after", "again", "all", "also", "any", "because",
    "before", "being", "between", "both", "each", "few", "get", "got",
    "just", "like", "more", "most", "now", "off", "one", "only",
    "other", "out", "over", "own", "same", "some", "such", "than",
    "that", "card", "cards", "status", "project",
})


def _card_aliases(card: dict[str, Any]) -> set[str]:
    raw = card.get("aliases") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(a).strip().lower() for a in raw if str(a).strip()}


def _card_text(card: dict[str, Any]) -> str:
    return " ".join(
        str(card.get(k) or "")
        for k in ("id", "ref", "title", "brief", "spec")
    ).lower()


def _reference_tokens(reference: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", reference.lower())
        if len(t) > 1 and t not in _CARD_STOPWORDS
    ]


def _ambiguity_error(reference: str, matches: list[dict[str, Any]]) -> LookupError:
    options = "; ".join(
        f"{c.get('ref') or c.get('id')} ({c.get('title')})"
        for c in matches[:5]
    )
    return LookupError(
        f"card reference {reference!r} matched {len(matches)} cards: {options}"
    )


def find_card(reference: str) -> dict[str, Any]:
    """Resolve a card by exact id/ref/alias, title substring, then scored
    natural-language token overlap. Ambiguous matches list candidates so the
    caller can retry with an exact ref."""
    needle = reference.strip().lower()
    if not needle:
        raise ValueError("card reference is required")
    cards = _cards()

    exact = [
        c for c in cards
        if needle in {str(c.get("id", "")).lower(), str(c.get("ref", "")).lower()}
        or needle in _card_aliases(c)
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise _ambiguity_error(reference, exact)

    titled = [
        c for c in cards
        if needle in str(c.get("title", "")).lower()
    ]
    if len(titled) == 1:
        return titled[0]
    if len(titled) > 1:
        raise _ambiguity_error(reference, titled)

    tokens = _reference_tokens(reference)
    if tokens:
        scored = []
        for c in cards:
            haystack = _card_text(c)
            hits = sum(
                1 for t in tokens
                if re.search(rf"\b{re.escape(t)}\b", haystack)
            )
            if hits:
                scored.append((hits, c))
        if scored:
            best = max(hits for hits, _ in scored)
            tops = [c for hits, c in scored if hits == best]
            if len(tops) == 1:
                return tops[0]
            raise _ambiguity_error(reference, tops)

    raise LookupError(f"no board card matched {reference!r}")


def _safe_card(card: dict[str, Any]) -> dict[str, Any]:
    """Keep receipts useful without returning arbitrary card internals."""
    return {
        "id": card.get("id"),
        "ref": card.get("ref"),
        "title": card.get("title"),
        "column": card.get("column"),
        "priority": card.get("priority"),
        "updated": card.get("updated"),
        "brief": card.get("brief"),
        "spec": card.get("spec"),
        "action": card.get("action") or {},
        "requests": card.get("requests") or [],
        "comms_tail": (card.get("comms") or [])[-5:],
    }


def build_plan(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "operation": "plan",
        "card": _safe_card(card),
        "recommended_stages": [
            "read current board, repo, executor, and artifact contracts",
            "define canonical job-envelope schema and lifecycle transitions",
            "define result bundle and independent verifier contract",
            "write tests and examples without changing live services",
            "review with an independent executor before implementation",
        ],
        "allowed_scope": ["planning", "schema", "documentation", "tests", "examples"],
        "blocked_without_approval": ["live service changes", "credentials", "merge", "deploy", "self-modifying Ada runtime"],
        "next_safe_action": "produce a design artifact and acceptance checklist",
    }


def build_handoff_preview(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": "handoff-preview",
        "card": {"id": card.get("id"), "ref": card.get("ref"), "title": card.get("title")},
        "envelope": {
            "card_id": card.get("id"),
            "task_class": "bounded development",
            "objective": card.get("brief"),
            "requirements": ["preserve board truth", "use an isolated worktree", "produce durable artifacts"],
            "executor": "devin-or-opencode-after-review",
            "allowed_paths": ["design docs", "schema", "tests", "examples"],
            "verify": ["schema validation", "unit tests", "independent review"],
            "approval_required": ["dispatch", "merge", "deploy", "self-modification"],
            "timeout_s": 900,
        },
        "mutations_performed": [],
        "next_safe_action": "review this preview, then explicitly dispatch through the board workflow",
    }


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True, timeout=10)
        return result.returncode, (result.stdout + result.stderr).strip()[:1200]
    except Exception as exc:
        return 1, str(exc)


def _plan(card_reference: str, include_memory: bool = False) -> dict[str, Any]:
    card = find_card(card_reference)
    plan = build_plan(card)
    if include_memory:
        plan["memory"] = recall_memory(
            " ".join(str(card.get(k) or "") for k in ("title", "brief", "spec"))
        )
    else:
        plan["memory"] = {
            "deferred": True,
            "next_tool": "ada_memory_search",
            "reason": "fast plan path; memory is opt-in",
        }
    plan["requires_native_followup"] = True
    plan["next_menu"] = ["Handoff preview", "Choose another card", "Stop"]
    return plan


@mcp.tool()
def ada_plan(card_reference: str, selection_confirmation: str = "", selection_token: str = "") -> str:
    """Read a card after exact native-menu confirmation; memory is deferred."""
    try:
        if selection_confirmation.strip().lower() != card_reference.strip().lower():
            return json.dumps({
                "ok": False,
                "operation": "plan",
                "error": "selection_confirmation_required",
                "expected": card_reference,
                "next_safe_action": "show a native confirmation menu and retry with the exact selected ref",
            }, ensure_ascii=False, indent=2)
        selection_error = _selection_gate(card_reference, selection_token)
        if selection_error:
            return json.dumps({
                "ok": False,
                "operation": "plan",
                "error": selection_error,
                "expected": card_reference,
                "next_safe_action": "use only a card and token from the current native candidate menu",
            }, ensure_ascii=False, indent=2)
        return json.dumps(_plan(card_reference), ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "plan", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_plan_with_context(card_reference: str, selection_confirmation: str = "", selection_token: str = "") -> str:
    """Read a confirmed card and explicitly include bounded context."""
    try:
        if selection_confirmation.strip().lower() != card_reference.strip().lower():
            return json.dumps({
                "ok": False,
                "operation": "plan-with-context",
                "error": "selection_confirmation_required",
                "expected": card_reference,
                "next_safe_action": "show a native confirmation menu and retry with the exact selected ref",
            }, ensure_ascii=False, indent=2)
        selection_error = _selection_gate(card_reference, selection_token)
        if selection_error:
            return json.dumps({
                "ok": False,
                "operation": "plan-with-context",
                "error": selection_error,
                "expected": card_reference,
                "next_safe_action": "use only a card and token from the current native candidate menu",
            }, ensure_ascii=False, indent=2)
        return json.dumps(_plan(card_reference, include_memory=True), ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "plan-with-context", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_memory_search(query: str) -> str:
    """Search Education lessons and local Devin summaries without mutating them."""
    try:
        return json.dumps({"ok": True, "operation": "memory-search", "memory": recall_memory(query)}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "memory-search", "error": str(exc)}, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_adobe_stock_reuse(campaign_idea: str, compact: bool = False) -> str:
    """Build a read-only Adobe Stock reuse report from bounded indexes and memory.

    compact=True returns only the preview summary and OpenChamber markdown —
    the full report exceeds tool-output display limits in OpenCode/OpenChamber.
    """
    try:
        indexes, missing = _read_adobe_stock_indexes()
        report = build_adobe_stock_reuse_report(
            campaign_idea,
            indexes=indexes,
            missing_indexes=missing,
        )
        if compact:
            preview_html_path = None
            try:
                preview_html_path = _write_preview_html(
                    _preview_html(
                        report.get("reusable_assets") or [],
                        str(report.get("campaign_idea") or campaign_idea),
                    ),
                    str(report.get("campaign_idea") or campaign_idea),
                )
            except Exception:
                preview_html_path = None
            return json.dumps(
                {
                    "ok": True,
                    "operation": "adobe-stock-reuse",
                    "compact": True,
                    "campaign_idea": report.get("campaign_idea"),
                    "preview_summary": report.get("preview_summary"),
                    "openchamber_preview_markdown": report.get(
                        "openchamber_preview_markdown"
                    ),
                    "preview_html_path": (
                        str(preview_html_path) if preview_html_path else None
                    ),
                    "mutations_performed": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {"ok": True, "operation": "adobe-stock-reuse", "report": report},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "operation": "adobe-stock-reuse",
            "error": str(exc),
            "mutations_performed": [],
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_self_verify() -> str:
    """Generate an evidence-bound verification code for the read-only bridge."""
    checks: dict[str, Any] = {
        "command_file": (WORKSPACE / ".opencode/commands/ada.md").exists(),
        "config_file": (WORKSPACE / ".opencode/opencode.json").exists(),
        "bridge_file": (WORKSPACE / ".opencode/ada_mcp.py").exists(),
    }
    try:
        card = find_card("ADA-1010")
        checks["card_id"] = card.get("id")
        checks["card_state"] = card.get("column")
        checks["board_reachable"] = True
    except Exception as exc:
        checks["board_reachable"] = False
        checks["board_error"] = str(exc)
    memory = recall_memory("ADA-1010 canonical Ada development control plane")
    checks["memory_lesson_hits"] = len(memory.get("lessons", []))
    checks["memory_devin_hits"] = len(memory.get("devin_sessions", []))
    checks["fixed_test_command"] = "python3 test_ada_mcp.py"
    checks["mutations_performed"] = []
    code = verification_code(checks)
    return json.dumps({
        "ok": all(v is True for k, v in checks.items() if k.endswith("_file") or k == "board_reachable"),
        "operation": "self-verify",
        "verification_code": code,
        "checks": checks,
        "independent_verification_required": True,
        "warning": "This code proves the read-only bridge checks completed; it is not approval of Ada's own work.",
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_status(card_reference: str) -> str:
    """Read verified board state for a card or reference; never mutates it."""
    try:
        card = find_card(card_reference)
        # An explicit status target establishes active-card context and
        # invalidates any older candidate menu from this MCP process.
        PENDING_NEXT_SELECTIONS.clear()
        return json.dumps({"ok": True, "operation": "status", "card": _safe_card(card)}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "status", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_next_card() -> str:
    """Rank the safest valuable backlog cards for Tony to review next."""
    try:
        cards = _backlog_cards()
        candidates = _rank_next_cards(cards)
        PENDING_NEXT_SELECTIONS.clear()
        for candidate in candidates:
            ref = str(candidate.get("ref") or candidate.get("id")).lower()
            token = secrets.token_urlsafe(12)
            candidate["selection_token"] = token
            PENDING_NEXT_SELECTIONS[ref] = (token, time.monotonic() + 900)
        return json.dumps({
            "ok": True,
            "operation": "next-card",
            "selection_policy": "priority + self-contained brief + checkable verifier - risk/open-decision/manual penalties",
            "candidates": candidates,
            "mutations_performed": [],
            "next_safe_action": "choose one candidate, confirm it, then call ada_preflight once",
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "next-card", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_handoff_preview(card_reference: str, plan_confirmation: str = "") -> str:
    """Build a handoff preview after exact plan-stage confirmation."""
    try:
        if plan_confirmation.strip().lower() != card_reference.strip().lower():
            return json.dumps({
                "ok": False,
                "operation": "handoff-preview",
                "error": "plan_confirmation_required",
                "expected": card_reference,
                "next_safe_action": "show the plan follow-up menu before retrying",
            }, ensure_ascii=False, indent=2)
        result = {"ok": True, **build_handoff_preview(find_card(card_reference))}
        result["requires_native_followup"] = True
        result["next_menu"] = ["Start supervised", "Revise", "Stop"]
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "handoff-preview", "error": str(exc), "board_api": BOARD_API})


READ_ONLY_PROFILES = {"ops-readonly", "ssot-research", "mddb-sequence", "storage-dryrun"}


def _resolved_verifier(action: dict[str, Any]) -> str | None:
    """Resolve a verifier from an explicit card field or a trusted profile."""
    explicit = action.get("verify") or action.get("verifier")
    if explicit:
        return str(explicit)
    profile = str(action.get("agent_profile") or action.get("profile") or "").strip()
    if action.get("executor") == "opencode" and profile in READ_ONLY_PROFILES:
        return f"profile:{profile}"
    return None


@mcp.tool()
def ada_preflight(
    card_reference: str,
    selection_confirmation: str = "",
    selection_token: str = "",
) -> str:
    """Run all read-only gates once and return one workflow receipt."""
    try:
        if selection_confirmation.strip().lower() != card_reference.strip().lower():
            return json.dumps({
                "ok": False,
                "operation": "preflight",
                "error": "selection_confirmation_required",
                "expected": card_reference,
            }, ensure_ascii=False, indent=2)
        selection_error = _selection_gate(card_reference, selection_token)
        if selection_error:
            return json.dumps({
                "ok": False,
                "operation": "preflight",
                "error": selection_error,
                "expected": card_reference,
                "next_safe_action": "use only a card and token from the current native candidate menu",
            }, ensure_ascii=False, indent=2)

        card = find_card(card_reference)
        action = card.get("action") or {}
        text = " ".join(str(card.get(k) or "") for k in ("title", "brief", "spec")).lower()
        risky = _positive_risk_terms(text)
        start_gates = {
            "card_found": True,
            "backlog": card.get("column") == "backlog",
            "no_open_requests": not bool(card.get("requests")),
            "safe_scope": not bool(risky),
            "typed_dispatch_available": False,
        }
        required = ("type", "repo", "executor", "model", "profile", "target_host", "artifact")
        aliases = {"profile": ("profile", "agent_profile"), "artifact": ("artifact", "output_artifact")}
        missing = [
            field for field in required
            if not any(action.get(name) for name in aliases.get(field, (field,)))
        ]
        verifier = _resolved_verifier(action)
        if not verifier:
            missing.append("verifier")
        dispatch = {
            "ok": not missing,
            "required_fields": list(required) + ["verifier"],
            "missing_fields": sorted(set(missing)),
            "resolved_verifier": verifier,
            "typed_dispatch_available": False,
            "approval_required": True,
        }
        workflow_id = f"ada-preflight-{secrets.token_urlsafe(12)}"
        ready = all(start_gates.values()) and dispatch["ok"]
        return json.dumps({
            "ok": True,
            "operation": "preflight",
            "workflow_id": workflow_id,
            "card": _safe_card(card),
            "plan": _plan(card_reference),
            "handoff": build_handoff_preview(card),
            "start_check": {
                "ok": all(start_gates.values()),
                "gates": start_gates,
                "risk_terms": risky,
                "executor": action.get("executor") or "not declared",
            },
            "dispatch_check": dispatch,
            "ready_for_dispatch": ready,
            "mutations_performed": [],
            "requires_native_followup": True,
            "next_menu": (
                ["Approve dispatch", "Revise", "Stop"]
                if ready else ["Review preflight", "Revise", "Stop"]
            ),
            "next_safe_action": (
                "typed dispatch is not wired; review the receipt and stop"
                if not ready else "require explicit approval before dispatch"
            ),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "preflight", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_start_check(card_reference: str, handoff_confirmation: str = "") -> str:
    """Validate start after exact handoff-stage confirmation."""
    try:
        if handoff_confirmation.strip().lower() != card_reference.strip().lower():
            return json.dumps({
                "ok": False,
                "operation": "start-check",
                "error": "handoff_confirmation_required",
                "expected": card_reference,
                "next_safe_action": "show the handoff decision menu before retrying",
            }, ensure_ascii=False, indent=2)
        card = find_card(card_reference)
        action = card.get("action") or {}
        text = " ".join(str(card.get(k) or "") for k in ("title", "brief", "spec")).lower()
        risky = _positive_risk_terms(text)
        gates = {
            "card_found": True,
            "backlog": card.get("column") == "backlog",
            "no_open_requests": not bool(card.get("requests")),
            "safe_scope": not bool(risky),
            "typed_dispatch_available": False,
        }
        return json.dumps({
            "ok": all(gates.values()) and not risky,
            "operation": "start-check",
            "card": _safe_card(card),
            "gates": gates,
            "risk_terms": risky,
            "executor": action.get("executor") or "not declared",
            "approval_required": True,
            "mutations_performed": [],
            "next_safe_action": (
                "typed dispatch is not wired; keep the card review-ready and "
                "do not dispatch from this tool"
            ),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "start-check", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_dispatch_check(card_reference: str) -> str:
    """Validate the typed dispatch envelope without creating a job."""
    try:
        card = find_card(card_reference)
        action = card.get("action") or {}
        required = ("type", "repo", "executor", "model", "profile", "target_host", "artifact")
        aliases = {"profile": ("profile", "agent_profile"), "artifact": ("artifact", "output_artifact")}
        missing = []
        for field in required:
            names = aliases.get(field, (field,))
            if not any(action.get(name) for name in names):
                missing.append(field)
        verifier = _resolved_verifier(action)
        if not verifier:
            missing.append("verifier")
        return json.dumps({
            "ok": not missing,
            "operation": "dispatch-check",
            "card": {"id": card.get("id"), "ref": card.get("ref"), "title": card.get("title")},
            "required_fields": list(required) + ["verifier"],
            "missing_fields": sorted(set(missing)),
            "typed_dispatch_available": False,
            "approval_required": True,
            "mutations_performed": [],
            "next_safe_action": (
                "complete the card envelope and obtain explicit approval; "
                "this check never creates or queues a job"
            ),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "dispatch-check", "error": str(exc), "board_api": BOARD_API})


@mcp.tool()
def ada_spend_summary() -> str:
    """Read the published LiteLLM spend ledger without provider credentials."""
    try:
        report = _mddb_post(
            "get",
            {"collection": "ada-cms-pages",
             "key": "llm-gateway-spend-ledger", "lang": "en"},
        )
        content = str(report.get("contentMd") or report.get("content") or "")
        updated_at = report.get("updatedAt")
        try:
            report_age_s = max(0, time.time() - float(updated_at))
        except (TypeError, ValueError):
            report_age_s = None
        total = re.search(r"\*\*Total 24h spend:\s*\$([0-9.]+)", content)
        rows = []
        for match in re.finditer(
            r"^\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([0-9]+)\s*\|"
            r"\s*([0-9]+)\s*\|\s*\$([0-9.]+)\s*\|$",
            content,
            re.M,
        ):
            if match.group(1).strip().lower() in {"key", "–", "-"}:
                continue
            rows.append({
                "key": match.group(1).strip(),
                "model": match.group(2).strip(),
                "calls": int(match.group(3)),
                "tokens": int(match.group(4)),
                "spend_usd": float(match.group(5)),
            })
        return json.dumps({
            "ok": True,
            "operation": "spend-summary",
            "source": "ada-cms-pages/llm-gateway-spend-ledger",
            "report_updated_at": updated_at,
            "report_age_s": report_age_s,
            "stale": report_age_s is None or report_age_s > 86400,
            "total_24h_usd": float(total.group(1)) if total else None,
            "rows": rows,
            "secrets_read": False,
            "mutations_performed": [],
            "next_safe_action": "keep paid jobs held if the budget guard is exceeded",
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "operation": "spend-summary",
            "error": str(exc),
            "source": "ada-cms-pages/llm-gateway-spend-ledger",
            "secrets_read": False,
            "mutations_performed": [],
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_self_test() -> str:
    """Run the fixed local Ada bridge contract test without external inputs.

    This is intentionally narrower than a general test runner: it cannot
    accept a command, mutate the board, or select a repository path.
    """
    command = ["python3", "test_ada_mcp.py"]
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            command,
            cwd=WORKSPACE,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()[-2000:]
        return json.dumps({
            "ok": result.returncode == 0,
            "operation": "self-test",
            "command": command,
            "workspace": str(WORKSPACE),
            "exit_code": result.returncode,
            "output": output,
            "mutations_performed": [],
            "next_safe_action": "review the test receipt; no dispatch or mutation was performed",
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "operation": "self-test",
            "command": command,
            "workspace": str(WORKSPACE),
            "error": str(exc),
            "mutations_performed": [],
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_debug_opencode() -> str:
    """Diagnose workspace, command files, MCP bridge, board reachability, and git state."""
    checks: dict[str, Any] = {
        "workspace": str(WORKSPACE),
        "command_file": (WORKSPACE / ".opencode/commands/ada.md").exists(),
        "config_file": (WORKSPACE / ".opencode/opencode.json").exists(),
        "bridge_file": (WORKSPACE / ".opencode/ada_mcp.py").exists(),
        "board_api": BOARD_API,
    }
    try:
        cards = _cards()
        checks["board_reachable"] = True
        checks["ada_1010_present"] = any(c.get("id") == "ada-dev-control-plane-envelope" or c.get("ref") == "ADA-1010" for c in cards)
    except Exception as exc:
        checks["board_reachable"] = False
        checks["board_error"] = str(exc)
    code, output = _run(["git", "status", "--short"])
    checks["git_status_exit"] = code
    checks["git_status"] = output
    return json.dumps({"ok": checks.get("command_file") and checks.get("config_file") and checks.get("bridge_file"), "operation": "debug", "checks": checks}, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_wiki_extensions() -> str:
    """List the approved MediaWiki extension state through board-api."""
    try:
        result = _post_json(
            "/wiki/extension",
            {"action": "list", "from": "ada"},
            timeout=15,
        )
        return json.dumps(
            {"ok": True, "operation": "wiki_extensions", "result": result,
             "mutations": []},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return json.dumps(
            {"ok": False, "operation": "wiki_extensions", "error": str(exc),
             "mutations": []},
            ensure_ascii=False,
            indent=2,
        )


WIKI_EXTENSION_ALLOWLIST = {
    "CategoryTree", "Mermaid", "MsUpload", "MultimediaViewer",
    "Cargo", "PageForms", "Lockdown",
}


@mcp.tool()
def ada_wiki_extension(
    action: str,
    name: str = "",
    confirmed: bool = False,
) -> str:
    """List or confirm-gated install/enable/disable of Wiki extensions."""
    action = action.strip().lower()
    name = name.strip()
    actions = {"list", "install", "enable", "disable"}
    if action not in actions:
        return json.dumps({
            "ok": False,
            "operation": "wiki_extension",
            "error": "action must be list|install|enable|disable",
        }, ensure_ascii=False, indent=2)
    if action != "list" and name not in WIKI_EXTENSION_ALLOWLIST:
        return json.dumps({
            "ok": False,
            "operation": "wiki_extension",
            "error": f"name is not approved: {sorted(WIKI_EXTENSION_ALLOWLIST)}",
            "mutations": [],
        }, ensure_ascii=False, indent=2)
    if action != "list" and not confirmed:
        return json.dumps({
            "ok": False,
            "operation": "wiki_extension",
            "needs_confirmation": True,
            "action": action,
            "name": name,
            "warning": (
                "This changes the live MediaWiki installation. "
                "Ask Tony for explicit confirmation, then retry the "
                "identical call with confirmed=true."
            ),
            "mutations": [],
        }, ensure_ascii=False, indent=2)
    try:
        payload = {"action": action, "from": "ada"}
        if name:
            payload["name"] = name
        result = _post_json("/wiki/extension", payload, timeout=90)
        return json.dumps({
            "ok": True,
            "operation": "wiki_extension",
            "result": result,
            "mutations": [] if action == "list" else [action],
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "operation": "wiki_extension",
            "error": str(exc),
            "mutations": [],
        }, ensure_ascii=False, indent=2)


def _open_requests(card: dict) -> list:
    return [r for r in (card.get("requests") or []) if r.get("status") != "answered"]


@mcp.tool()
def ada_attention(oldest: int = 5) -> str:
    """List what needs Tony: cards with unanswered requests, plus the N oldest backlog cards for sweep triage."""
    cards = _backlog_cards()
    pending = []
    for c in cards:
        open_reqs = _open_requests(c)
        if open_reqs:
            pending.append({
                "id": c.get("id"), "ref": c.get("ref"), "title": c.get("title"),
                "column": c.get("column"), "priority": c.get("priority"),
                "requests": [{"id": r.get("id"), "ask": r.get("ask"),
                              "options": r.get("options") or [],
                              "suggested": r.get("suggested")} for r in open_reqs],
            })
    backlog = [c for c in cards if c.get("column") == "backlog"]
    backlog.sort(key=lambda c: c.get("updated") or "")
    oldest_cards = [{
        "id": c.get("id"), "ref": c.get("ref"), "title": c.get("title"),
        "priority": c.get("priority"), "updated": c.get("updated"),
        "action": c.get("action"),
    } for c in backlog[:max(0, int(oldest))]]
    return json.dumps({
        "ok": True, "operation": "attention",
        "pending_requests": pending,
        "oldest_backlog": oldest_cards,
        "next": "present each pending request as a native question; answers go through ada_respond",
        "mutations": [],
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def ada_respond(card_reference: str, request_id: str, answer: str, confirmed: bool = False) -> str:
    """Answer an open card request on the board -- mutation, two-call confirmation gate."""
    try:
        card = find_card(card_reference)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "respond", "error": str(exc), "mutations": []})
    open_reqs = _open_requests(card)
    target = next((r for r in open_reqs if r.get("id") == request_id), None)
    if target is None:
        return json.dumps({"ok": False, "operation": "respond",
                           "error": "no open request with that id on this card",
                           "open_request_ids": [r.get("id") for r in open_reqs],
                           "mutations": []})
    if not confirmed:
        return json.dumps({"ok": False, "operation": "respond", "preview": {
            "card": card.get("id"), "request_id": request_id,
            "ask": target.get("ask"), "answer": answer},
            "warning": "This writes the answer to the board. Retry the identical call with confirmed=true.",
            "mutations": []})
    try:
        result = _post_json("/respond", {"id": card.get("id"), "request_id": request_id,
                                         "answer": answer, "from": "ada"}, timeout=30)
        return json.dumps({"ok": True, "operation": "respond", "result": result,
                           "mutations": ["respond"]})
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "respond", "error": str(exc), "mutations": []})


_CARD_ACTIONS = {"queue", "hold"}


@mcp.tool()
def ada_card_action(card_reference: str, do: str, confirmed: bool = False) -> str:
    """Queue or hold a board card -- mutation, two-call confirmation gate."""
    if do not in _CARD_ACTIONS:
        return json.dumps({"ok": False, "operation": "card_action",
                           "error": f"action {do!r} not allowed -- permitted: {sorted(_CARD_ACTIONS)}",
                           "mutations": []})
    try:
        card = find_card(card_reference)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "card_action", "error": str(exc), "mutations": []})
    if not confirmed:
        return json.dumps({"ok": False, "operation": "card_action", "preview": {
            "card": card.get("id"), "do": do, "current_action": card.get("action")},
            "warning": "This mutates board action state. Retry the identical call with confirmed=true.",
            "mutations": []})
    try:
        result = _post_json("/action", {"id": card.get("id"), "do": do}, timeout=30)
        return json.dumps({"ok": True, "operation": "card_action", "result": result,
                           "mutations": [do]})
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "card_action", "error": str(exc), "mutations": []})


ADOBE_PICKER = os.environ.get(
    "ADA_ADOBE_PICKER", "http://100.68.142.13:8777").rstrip("/")
_QC_DECISIONS = ("keep", "reject", "revise")
_ADOBE_CAMPAIGN = "AS-2026-11-05-REMOTE-WORK-001"


def _picker_get(path: str, timeout: float = 30) -> Any:
    req = urllib.request.Request(
        f"{ADOBE_PICKER}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _picker_post(path: str, payload: dict[str, Any], timeout: float = 30) -> Any:
    req = urllib.request.Request(
        f"{ADOBE_PICKER}{path}",
        data=json.dumps(payload).encode(),
        headers={"Accept": "application/json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


@mcp.tool()
def ada_adobe_stock_qc() -> str:
    """Adobe Stock campaign QC — pending candidate image + QC menu spec.

    Read-only. Show the candidate `markdown` (renders the image inline),
    then present a native question menu with the `menu` options. Tony's
    pick becomes the asset+decision for ada_adobe_stock_qc_decide
    (keep|reject|revise); a 'repair' pick means reply with `picker_url`
    instead — no decide call.
    """
    try:
        camp = _picker_get("/api/campaign", 60)
        sess = _picker_get("/api/session", 60)
        assets = camp.get("generated_files") or []
        focus = ((sess.get("comparison") or {})
                 .get("last_landed") or {}).get("asset_id") \
            or (assets[-1] if assets else None)
        out: dict[str, Any] = {
            "ok": True, "operation": "adobe-stock-qc",
            "campaign": _ADOBE_CAMPAIGN,
            "status": camp.get("status"), "focus_asset": focus,
            "pending_decisions": camp.get("pending_decisions") or [],
            "adobe_upload": False,
        }
        if focus:
            out["image_url"] = f"{ADOBE_PICKER}/img/{focus}"
            out["picker_url"] = f"{ADOBE_PICKER}/?img=/img/{focus}"
            out["markdown"] = (
                f"**Candidate: {focus}**\n\n"
                f"![{focus}]({ADOBE_PICKER}/img/{focus})\n")
            out["menu"] = [
                {"id": f"adobe_qc:{d}:{focus}",
                 "label": f"{d.capitalize()} {focus}"}
                for d in _QC_DECISIONS
            ] + [{"id": f"adobe_qc:repair:{focus}",
                  "label": f"Repair region… ({focus})"}]
        out["next_safe_action"] = (
            "show the candidate image markdown, then present a native "
            "question menu with the `menu` options; on Tony's pick call "
            "ada_adobe_stock_qc_decide with that asset + decision and "
            "selection_confirmation=asset — for 'repair' reply with "
            "picker_url instead of deciding")
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "operation": "adobe-stock-qc",
                           "error": str(exc)})


@mcp.tool()
def ada_adobe_stock_qc_decide(
        asset: str, decision: str, note: str = "",
        selection_confirmation: str = "") -> str:
    """Record Tony's QC decision for a candidate — mutation, native-menu gate.

    Writes SESSION.json + appends 40-qc-review.md on Drive via the
    picker-server. Requires selection_confirmation == asset after the
    native confirmation menu pick; keep|reject|revise only.
    """
    if selection_confirmation.strip().lower() != asset.strip().lower():
        return json.dumps({
            "ok": False, "operation": "adobe-stock-qc-decide",
            "error": "selection_confirmation_required",
            "expected": asset,
            "next_safe_action": "show a native confirmation menu and "
                                "retry with the exact selected asset"})
    dec = decision.strip().lower()
    if dec not in _QC_DECISIONS:
        return json.dumps({"ok": False, "operation": "adobe-stock-qc-decide",
                           "error": f"decision must be one of {_QC_DECISIONS}"})
    try:
        res = _picker_post("/api/qc-decision",
                           {"asset": asset, "decision": dec,
                            "note": (note or "")[:400]}, timeout=120)
        return json.dumps({"ok": bool(res.get("ok")),
                           "operation": "adobe-stock-qc-decide",
                           "result": res,
                           "recorded": {"asset": asset, "decision": dec}},
                          ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False,
                           "operation": "adobe-stock-qc-decide",
                           "error": str(exc)})


if __name__ == "__main__":
    mcp.run(transport="stdio")

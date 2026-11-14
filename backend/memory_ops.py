"""Curated memory-bank operations — search, remember (create/correct/
supersede), and retract.

These take (mddb, registry, instance) explicitly instead of living on
ToolRunner, so the write path is testable in isolation and the runner stays
focused on dispatch + Home Assistant tooling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.mddb_client import MddbClient
from backend import write_outbox
from backend import memory_write_guard
from backend.memory_banks import (
    MemoryBank,
    MemoryBankRegistry,
    SENSITIVE_MEMORY_RE,
    _meta_first,
    _slug,
    doc_effective_status,
)

logger = logging.getLogger("tools")

# Minimum vector-search score for a bank hit to count as confident. Below
# this, recall escalates to the NotebookLM deep tier. ~0.45-0.55 is the
# observed "real match" band on this MDDB's embedding model.
ADA_BANK_SEARCH_THRESHOLD = float(os.environ.get("ADA_BANK_SEARCH_THRESHOLD", "0.45"))

# Higher bar for "this is the same fact restated" — when ada_remember finds
# no exact key/subject match, a hit above this score is corrected in place
# instead of creating a near-duplicate.
ADA_BANK_UPDATE_THRESHOLD = float(os.environ.get("ADA_BANK_UPDATE_THRESHOLD", "0.85"))

# Below this stored confidence, a hit is tagged unverified so the model must
# hedge or escalate rather than state it as fact. record_outcome() is what
# moves a doc's confidence over time — this gate makes that matter.
# See docs/ada-memory/tony-projects/knowledge-circle.md (chaba repo).
ADA_BANK_CONFIDENCE_MIN = float(os.environ.get("ADA_BANK_CONFIDENCE_MIN", "0.4"))

# Banks whose auto-extracted draft notes may surface in recall before human
# promotion. Restricted to same-instance docs — extraction noise stays
# private to the instance that produced it.
DRAFT_VISIBLE_BANKS = {
    s.strip()
    for s in os.environ.get("ADA_DRAFT_VISIBLE_BANKS", "personal,general").split(",")
    if s.strip()
}
# Drafts must clear a higher score bar than active docs and age out of
# recall after this many days, so stale extraction noise can't linger.
ADA_DRAFT_MIN_SCORE = float(os.environ.get("ADA_DRAFT_MIN_SCORE", "0.55"))
ADA_DRAFT_MAX_AGE_DAYS = int(os.environ.get("ADA_DRAFT_MAX_AGE_DAYS", "7"))

# Read-only banks (cms, chaba-*, kb-*, infrastructure-ssot) are reference
# docs, not memories — in degraded keyword mode they need a higher match
# bar so they can't tie-crowd out the curated memory banks.
ADA_DEGRADED_KB_MIN_SCORE = float(
    os.environ.get("ADA_DEGRADED_KB_MIN_SCORE", "0.5"))

# Ops-routed banks are degraded BY DESIGN (no embedding provider — they
# will never vector-search), and their candidate set is already bounded
# by the recency window, so a much lower keyword bar is safe — the 0.5
# bar made scenario-report recall unreachable (2026-10-05).
ADA_OPS_LISTING_MIN_SCORE = float(
    os.environ.get("ADA_OPS_LISTING_MIN_SCORE", "0.15"))

# Max chars of a surfaced doc injected into a live turn. Raw session dumps
# and KB pages are much bigger; Ada gets an excerpt + key, not the blob.
ADA_HIT_MAX_CHARS = int(os.environ.get("ADA_HIT_MAX_CHARS", "2000"))

# Bulk-archive doc-key prefixes demoted in bank='all' merges. Devin
# session dumps (devin/session-summary/*, devin/report/* — synced
# transcripts) match nearly any topical query by sheer size and once
# filled every hit slot (liam-e2e, session ece770ccb0). They still count
# — they just sort below real memory hits, and an explicit bank='devin'
# (or 'reports') search returns them unfiltered.
ADA_ALL_DEMOTE_KEY_PREFIXES = tuple(
    p.strip()
    for p in os.environ.get("ADA_MEMORY_DEMOTE_PREFIXES", "devin/").split(",")
    if p.strip()
)


def _demoted_in_all(hit: dict[str, Any]) -> bool:
    """True when a hit is bulk-archive exhaust that must not crowd out
    real memory in the bank='all' merge."""
    key = str(hit.get("key") or "")
    return any(key.startswith(p) for p in ADA_ALL_DEMOTE_KEY_PREFIXES)

# How each recorded outcome nudges a doc's confidence (clamped 0.05..1.0).
# Good outcomes also bump last_verified — the verify stage of the knowledge
# circle. "skipped" is neutral: the check ran but was never exercised.
OUTCOME_CONFIDENCE_DELTA = {
    "good": 0.1,
    "worked": 0.1,
    "bought_good": 0.1,
    "partial": -0.05,
    "skipped": 0.0,
    "bad": -0.2,
    "failed": -0.2,
    "bought_bad": -0.2,
}

# Reconnect priming: when a websocket session opens, the gap since the
# previous session ended is classified so session_prime_text can tell the
# model how to greet the user. Boundaries are deliberately coarse — the
# greeting should feel natural, not like a stopwatch. Scenario coverage:
# tests/scenarios/reconnect_continuity.yaml.
AWAY_TIER_RESUME = "resume"  # brief disconnect — pick up mid-thought
AWAY_TIER_RETURN = "return"  # same-day hours — welcome back, offer to resume
AWAY_TIER_RECAP = "recap"    # most of a day — offer a one-line recap
AWAY_TIER_DAYS = "days"      # several days — greet, recap, don't assume active
AWAY_TIER_LONG = "long"      # a week or more — treat as a fresh start

_AWAY_TIERS: list[tuple[float, str]] = [
    (15 * 60, AWAY_TIER_RESUME),
    (4 * 3600, AWAY_TIER_RETURN),
    (24 * 3600, AWAY_TIER_RECAP),
    (7 * 86400, AWAY_TIER_DAYS),
]


def away_tier(away_seconds: float | None) -> str | None:
    """Reconnect tier for a gap, or None for a first-ever session."""
    if away_seconds is None or away_seconds < 0:
        return None
    for limit, tier in _AWAY_TIERS:
        if away_seconds < limit:
            return tier
    return AWAY_TIER_LONG


def _format_away(away_seconds: float) -> str:
    if away_seconds < 120:
        return "a minute or two"
    if away_seconds < 3600:
        return f"about {max(1, round(away_seconds / 60))} minutes"
    if away_seconds < 86400:
        return f"about {round(away_seconds / 3600)} hours"
    return f"about {round(away_seconds / 86400)} days"


def reconnect_directive(away_seconds: float | None, last_tail: str = "",
                        context_reset: bool = False) -> str | None:
    """Session note telling the model how to greet a returning user, or
    None for a first-ever session (no prior disconnect on record).

    context_reset=True is the mid-conversation provider reconnect that
    could NOT resume (no resumption handle): the model-side context was
    wiped and only last_tail survives. The directive makes her say the
    reset aloud instead of quietly continuing on partial context (the
    23:19 ICT 2026-10-13 kk incident — stream ended 1000, resumed=False,
    she contradicted the coffee-plug answer without ever flagging the
    reset)."""
    if context_reset:
        note = (
            "Session note: your model connection just reset and could NOT "
            "be resumed — the previous context is gone and only the "
            "transcript tail below survived; the last exchange may be "
            "incomplete. Acknowledge it aloud in one clause (the "
            "connection hiccupped, the last moment may be missing), then "
            "pick the conversation back up from the tail — do not "
            "contradict what it shows."
        )
        if last_tail.strip():
            note += ("\nConversation just before the reset:\n"
                     + last_tail.strip())
        return note
    tier = away_tier(away_seconds)
    if tier is None:
        return None
    gap = _format_away(float(away_seconds))
    # 2026-10-01: every tier must say the CURRENT message wins — without it
    # a reconnect next to a pending topic made Ada ignore the user's actual
    # first request and go back to pending work (cam_to_screen flake).
    _current_first = (
        " If the user's first message is a new request, answer that directly"
        " — previous topics only if they ask.")
    if tier == AWAY_TIER_RESUME:
        note = (
            f"Session note: the user's connection dropped {gap} ago — this is "
            "the same conversation resuming, not a new one. Greet them very "
            "briefly and pick up right where you left off." + _current_first
        )
        if last_tail.strip():
            note += (
                "\nConversation just before the disconnect:\n"
                + last_tail.strip()
            )
        return note
    if tier == AWAY_TIER_RETURN:
        return (
            f"Session note: the user is back after {gap}. Welcome them back "
            "and offer to continue what you were discussing."
            + _current_first
        )
    if tier == AWAY_TIER_RECAP:
        return (
            f"Session note: the user has been away for {gap}. Welcome them "
            "back and offer a one-line recap of what you were working on, "
            "then ask if they want to continue." + _current_first
        )
    if tier == AWAY_TIER_DAYS:
        return (
            f"Session note: the user has been away for {gap}. Greet them "
            "warmly, mention briefly what you last discussed, and ask what "
            "they'd like to do — don't assume the old task is still active."
        )
    return (
        f"Session note: the user has been away for {gap}. Treat this as a "
        "fresh start — greet them and ask what they'd like to do. Use the "
        "context below only if it's clearly relevant."
    )


def memory_meta(
    bank: MemoryBank,
    scope: str,
    today: str,
    kind: str | None,
    subject: str | None,
    attribute: str | None,
    valid_until: str | None,
    applies_to: list[str],
    session_id: str | None = None,
    prime: bool | None = None,
) -> dict[str, list[str]]:
    meta: dict[str, list[str]] = {
        "bank": [bank.name],
        "kind": [str(kind or "note")],
        "scope": [scope],
        "status": ["active"],
        "valid_from": [today],
        "last_verified": [today],
        "source": ["voice"],
        "written_by": ["ada_remember"],
    }
    if subject:
        meta["subject"] = [str(subject)]
    if attribute:
        meta["attribute"] = [str(attribute)]
    if valid_until:
        meta["valid_until"] = [str(valid_until)]
    if applies_to:
        meta["applies_to"] = [str(a) for a in applies_to]
    if session_id and session_id != "unknown":
        meta["session_id"] = [session_id]
    if prime:
        meta["prime"] = ["true"]
    if bank.env:
        meta["env"] = [bank.env]
    return meta


def _warn_unknown_meta(registry: MemoryBankRegistry, meta: dict[str, list[str]]) -> None:
    unknown = registry.validate_meta(meta)
    if unknown:
        logger.warning(
            "memory meta fields not in SSOT meta_schema: %s "
            "(extend docs/ssot/apps/ssot.apps.ada-memory-schema.yml)", unknown,
        )


def _must(res: Any, what: str) -> None:
    # mddb write helpers return None on failure; surface it instead of
    # reporting a successful write.
    if res is None:
        raise RuntimeError(f"mddb write failed: {what}")


def _queued_note(out: dict[str, Any], res: Any) -> dict[str, Any]:
    """Annotate a result when its write landed in the retry outbox instead
    of mddb — honest 'not yet saved' without raising (unlike _must)."""
    if write_outbox.is_queued(res):
        out["queued_for_retry"] = True
        out["note"] = ("mddb unreachable — parked in the local write "
                       "outbox; it will land automatically within the "
                       "retry window. Do NOT describe it as saved yet.")
    return out


def _draft_bank_allowed(bank: MemoryBank) -> bool:
    """Draft visibility at the bank level. Person-scoped banks
    (personal-kk, personal-testo) inherit the 'personal' allowlist entry —
    bank.name is the resolved name, not the alias the user configured."""
    if bank.name in DRAFT_VISIBLE_BANKS:
        return True
    return bank.scope == "person" and "personal" in DRAFT_VISIBLE_BANKS


def _draft_visible(bank: MemoryBank, doc: dict[str, Any], instance: str) -> bool:
    """True when a status=draft doc may surface in recall: only for
    allowlisted banks, only the extracting instance's own docs, only
    recent, and only above the draft score bar."""
    if not _draft_bank_allowed(bank):
        return False
    meta = doc.get("meta") or {}
    if _meta_first(meta, "scope") != instance:
        return False
    score = doc.get("score")
    if score is not None and float(score) < ADA_DRAFT_MIN_SCORE:
        return False
    try:
        created = datetime.fromisoformat(str(_meta_first(meta, "valid_from") or ""))
    except ValueError:
        return False
    age = datetime.now(timezone.utc).date() - created.date()
    return 0 <= age.days <= ADA_DRAFT_MAX_AGE_DAYS


# Transcript-shaped content (card ada-memory-hit-shape-guard,
# context-growth debug run 2026-10-07 ~21:39): the 'devin' bank indexes
# Devin session dumps — literal '=== MESSAGE 1 - System ===' chat
# exports. When that shape reaches the live model as ordinary context
# it pattern-matches the roles as the conversation it is IN and starts
# executing the transcript's tool calls (the tool storms). Detection is
# shape-based, not bank-based: a pasted transcript in any bank carries
# the same hazard. Two headers minimum — a doc quoting the format once
# is not a transcript.
TRANSCRIPT_HEADER_RE = re.compile(
    r"^[ \t]*={2,}\s*(?:MESSAGE|MSG|USER|ASSISTANT|SYSTEM|HUMAN|AI|"
    r"TOOL|MODEL|AGENT)\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE)

# Meta kinds / key segments that mark archived transcripts even when a
# truncated excerpt shows fewer than two message headers.
ARCHIVAL_KINDS = {
    "session-dump", "session-transcript", "transcript", "chat-log",
    "conversation-dump",
}
ARCHIVAL_KEY_SEGMENTS = ("/session-dump/", "/transcript/", "/chat-log/")


def is_transcript_shaped(
    content: Any, kind: Any = None, key: Any = None
) -> bool:
    """True when content/meta looks like a raw chat or session dump
    rather than a curated memory."""
    if str(kind or "").strip().lower() in ARCHIVAL_KINDS:
        return True
    if any(seg in str(key or "").lower() for seg in ARCHIVAL_KEY_SEGMENTS):
        return True
    return len(TRANSCRIPT_HEADER_RE.findall(str(content or ""))) >= 2


def _doc_to_hit(
    bank: MemoryBank,
    doc: dict[str, Any],
    instance: str,
    include_inactive: bool,
    with_bank: bool = False,
) -> dict[str, Any] | None:
    """Shape one doc into a search hit, or None when filtered out."""
    status = doc_effective_status(doc)
    is_draft = False
    if bank.status_outcome:
        # meta.status is a run outcome (pass/fail/flaky), not a lifecycle
        # state — pass it through to the hit, never gate on it.
        pass
    elif status != "active":
        if include_inactive:
            # Audit reads surface drafts too — still flag them so the
            # tool boundary frames the content like any draft hit.
            if status == "draft":
                is_draft = True
        elif status == "draft" and _draft_visible(bank, doc, instance):
            is_draft = True
        else:
            return None
    meta = doc.get("meta") or {}
    content = doc.get("contentMd") or doc.get("content_md") or ""
    if not str(content).strip():
        # Content-wiped doc (meta-only write emptied the body). It still
        # scores on meta but carries nothing Ada can use — and it crowds
        # out real hits in the top-k. Skip it.
        return None
    # Reference banks hold large docs (devin session dumps can be ~50KB).
    # Injecting one verbatim explodes the live-turn context and slows
    # first-audio — surface an excerpt; the key is there to fetch the rest.
    if len(content) > ADA_HIT_MAX_CHARS:
        content = (
            content[:ADA_HIT_MAX_CHARS].rstrip()
            + f"\n… [truncated — {len(content)} chars; fetch key for full doc]"
        )
    hit: dict[str, Any] = {
        "key": doc.get("key"),
        "status": status,
        "score": doc.get("score"),
        "content": content,
        "kind": _meta_first(meta, "kind"),
        "subject": _meta_first(meta, "subject"),
        "attribute": _meta_first(meta, "attribute"),
        "last_verified": _meta_first(meta, "last_verified"),
        "valid_until": _meta_first(meta, "valid_until"),
    }
    if with_bank:
        hit["bank"] = bank.name
    if is_draft:
        hit["draft"] = True
        hit["unverified"] = True
    else:
        try:
            confidence = float(_meta_first(meta, "confidence"))
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            hit["confidence"] = confidence
            if confidence < ADA_BANK_CONFIDENCE_MIN:
                hit["unverified"] = True
    return hit


_KW_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "who", "what", "where", "when", "how", "why", "which", "tell", "me",
    "about", "in", "on", "at", "to", "of", "for", "and", "or", "this",
    "that", "you", "your", "we", "my", "it", "its", "be", "been", "know",
    "can", "could", "would", "should", "refer", "now", "any",
}


def _keyword_rank(docs: list[dict[str, Any]], q: str) -> list[dict[str, Any]]:
    """Score listed docs by query-token presence when vector search is
    down: key/subject/content substring hits. English stopwords don't
    count — they match nearly everything and drown the real hits. Docs
    with no token overlap get score 0 and sort last."""
    tokens = [
        t for t in re.split(r"[^\wก-๙]+", q.lower())
        if len(t) >= 2 and t not in _KW_STOP
    ]
    if not tokens:
        return docs
    for doc in docs:
        meta = doc.get("meta") or {}
        hay = " ".join(filter(None, [
            str(doc.get("key") or ""),
            str(doc.get("contentMd") or doc.get("content_md") or ""),
            str(_meta_first(meta, "subject") or ""),
            str(_meta_first(meta, "attribute") or ""),
        ])).lower()
        hits = sum(1 for t in tokens if t in hay)
        doc["score"] = hits / len(tokens) if hits else 0.0
    docs.sort(key=lambda d: d.get("score") or 0.0, reverse=True)
    # Degraded listing order is arbitrary — a zero-token-overlap doc is
    # noise, not a weak hit. Drop them rather than fill the response
    # with random bank entries (they also crowd out matching docs from
    # other banks in the bank='all' merge).
    return [d for d in docs if (d.get("score") or 0.0) > 0]


async def _bank_docs(
    mddb: MddbClient,
    bank: MemoryBank,
    q: str,
    limit: int,
    include_inactive: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch candidate docs for one bank: vector search on a real query,
    meta-filtered listing otherwise; falls back to listing when the
    embedding path is down. Returns (docs, degraded)."""
    # Read-only banks (kb imports, devin summaries) have no lifecycle meta —
    # filtering on status would exclude every doc. Post-filter still runs
    # (missing status is treated as active). Draft-visible banks widen the
    # filter so recent same-instance drafts can surface as unverified hits.
    filter_meta = None
    if not include_inactive and bank.writable:
        statuses = ["active", "draft"] if _draft_bank_allowed(bank) else ["active"]
        filter_meta = {"status": statuses}
    if q and q != "*":
        docs = (
            await mddb.vector_search(
                collection=bank.mddb_collection,
                query=q,
                limit=int(limit) * 3,
                filter_meta=filter_meta,
                threshold=ADA_BANK_SEARCH_THRESHOLD,
            )
            # The ops store has no embedding provider by design — a
            # vector call there is a guaranteed 400, so go straight to
            # the degraded listing path.
            if not mddb.is_ops_routed(bank.mddb_collection) else None
        )
        if docs is None:
            # Degraded: widen the listing well past the caller's limit so
            # keyword ranking has candidates to work with — an unordered
            # top-N listing would silently miss the right doc.
            listing_filter = filter_meta
            if mddb.is_ops_routed(bank.mddb_collection):
                # Ops listings are insertion-ordered (oldest first) with
                # no offset/sort — unfiltered candidates can never reach
                # fresh docs in a large collection (2026-10-04:
                # scenario_report_recall failed because the newest
                # reports never entered the 50-doc window). Bound the
                # window to recent last_verified days instead.
                days = [
                    (datetime.now(timezone.utc).date() - timedelta(days=i)).isoformat()
                    for i in range(7)
                ]
                listing_filter = {
                    **(filter_meta or {}), "last_verified": days}
            listed = await mddb.search_documents(
                collection=bank.mddb_collection,
                filter_meta=listing_filter,
                limit=max(int(limit) * 10, 50),
            )
            ranked = _keyword_rank(listed, q)
            if not bank.writable:
                min_score = (
                    ADA_OPS_LISTING_MIN_SCORE
                    if mddb.is_ops_routed(bank.mddb_collection)
                    else ADA_DEGRADED_KB_MIN_SCORE
                )
                ranked = [d for d in ranked
                          if (d.get("score") or 0.0) >= min_score]
            return ranked[: int(limit) * 3], True
        return docs or [], False
    return (
        await mddb.search_documents(
            collection=bank.mddb_collection,
            filter_meta=filter_meta,
            limit=int(limit) * 3,
        ),
        False,
    )


def _check_bank_allowed(
    registry: MemoryBankRegistry, bank: str, person_entity: str | None
) -> None:
    """Per-speaker bank ACL — raises PermissionError when the registry's
    person_policies exclude this bank for this speaker."""
    if not registry.bank_allowed(str(bank).strip(), person_entity):
        logger.warning("denied bank %r for speaker %r", bank, person_entity)
        allowed = ", ".join(registry.banks_for_person(person_entity))
        raise PermissionError(
            f"memory bank '{bank}' is not available for this speaker"
            + (f" — allowed banks: {allowed}" if allowed else "")
        )


def _check_person_scope_write(bank: MemoryBank, person_entity: str | None) -> None:
    """Person-scoped banks are private to their owner for writes —
    person_scope and key_scope declare who may write; the 'personal'
    alias already routes callers to their own bank before this runs."""
    if bank.scope != "person":
        return
    owners = {bank.person_scope, *bank.key_scope}
    owners.discard(None)
    if person_entity not in owners:
        logger.warning(
            "denied write to person-scoped bank %r for identity %r",
            bank.name, person_entity,
        )
        raise PermissionError(
            f"memory bank '{bank.name}' is private to its owner — "
            "use bank 'personal' for the speaker's own notes instead"
        )


async def _timed_bank_docs(
    mddb: MddbClient,
    bank: MemoryBank,
    q: str,
    limit: int,
    include_inactive: bool,
) -> tuple[list[dict[str, Any]], bool, float]:
    """_bank_docs + wall time — feeds the memory_search timing log."""
    t0 = time.perf_counter()
    docs, degraded = await _bank_docs(mddb, bank, q, limit, include_inactive)
    return docs, degraded, time.perf_counter() - t0


def _log_search_timing(
    bank_arg: str,
    n_banks: int,
    per_bank_ms: dict[str, int],
    total_s: float,
    hits: list[dict[str, Any]],
    degraded: bool,
) -> None:
    """One greppable line per memory_search — the Phase-0 feed for the
    bank-router card (nest-bank-router): mine `memory_search timing`
    lines from the journal to split per-bank wall time vs total. The
    remote query-embed happens inside mddb, so a per-bank ms that
    saturates near total_ms is embed-bound; near-zero means the mddb
    embed cache (or an ops-routed keyword path) served it."""
    logger.info(
        "memory_search timing bank=%s banks=%d total_ms=%.0f "
        "per_bank_ms=%s hits=%d top_bank=%s degraded=%s",
        bank_arg, n_banks, total_s * 1000,
        json.dumps(per_bank_ms, sort_keys=True, separators=(",", ":")),
        len(hits), hits[0].get("bank") if hits else None, degraded,
    )


def _schedule_bge_shadow(
    query: str,
    banks: list[MemoryBank],
    hits: list[dict[str, Any]],
    *,
    instance: str,
    degraded: bool,
) -> None:
    """Start optional BGE shadow work without changing the baseline result."""
    try:
        from backend.bge_memory_shadow import schedule
        schedule(
            query, banks, hits, instance=instance,
            baseline_degraded=degraded)
    except Exception as exc:  # shadow setup must not break memory recall
        logger.info(
            "bge_memory_search shadow status=error error_type=%s",
            type(exc).__name__,
        )


async def memory_search(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    bank: str,
    query: str,
    limit: int = 5,
    include_inactive: bool = False,
    person_entity: str | None = None,
) -> dict[str, Any]:
    """Search a curated memory bank's MDDB collection.

    A real query uses semantic (vector) search; '*' or empty lists by
    metadata filter. Vector search failures fall back to the listing so
    recall degrades gracefully when embeddings are down.
    bank='all' fans out across every bank assigned to this instance and
    merges hits by score — the model doesn't have to guess which bank.

    When *person_entity* is set (speaker ID active), the default 'personal'
    bank is swapped for the speaker's person-scoped bank (e.g. personal-kk)
    in both single-bank and bank='all' modes, so personal memory never
    crosses person boundaries.

    A comma-separated bank list ('people,general') fans out over just
    that subset — the routed-hint path for the bank-router specialist
    (card nest-bank-router): same parallel search, same merge, but only
    over the banks the router predicted."""
    q = str(query or "").strip()
    t_start = time.perf_counter()
    bank_spec = str(bank).strip()
    wanted = [w.strip() for w in bank_spec.split(",") if w.strip()]
    if bank_spec.lower() in ("all", "*") or len(wanted) > 1:
        fanout_all = bank_spec.lower() in ("all", "*")
        banks = list(registry.banks_for_person(person_entity).values())
        if not fanout_all:
            # Routed subset: resolve 'personal' to the speaker's scoped
            # bank the same way the single-bank path does, then keep only
            # banks the caller asked for. Unknown names are a caller bug —
            # say so rather than silently widening to 'all'.
            names = {b.name for b in banks}
            resolved = {
                registry.personal_bank_name(person_entity)
                if w == "personal" and person_entity else w
                for w in wanted
            }
            unknown = resolved - names
            if unknown:
                raise ValueError(
                    f"unknown memory bank(s): {sorted(unknown)}")
            banks = [b for b in banks if b.name in resolved]
            for b in banks:
                _check_bank_allowed(registry, b.name, person_entity)
        results = await asyncio.gather(
            *(
                _timed_bank_docs(mddb, b, q, limit, include_inactive)
                for b in banks
            )
        )
        hits: list[dict[str, Any]] = []
        degraded = False
        per_bank_ms = {
            b.name: round(elapsed * 1000)
            for b, (_docs, _deg, elapsed) in zip(banks, results)
        }
        for b, (docs, deg, _elapsed) in zip(banks, results):
            degraded = degraded or deg
            used: list[dict[str, Any]] = []
            for doc in docs:
                hit = _doc_to_hit(b, doc, registry.instance, include_inactive, with_bank=True)
                if hit is None:
                    continue
                hits.append(hit)
                used.append(doc)
            if used and not include_inactive and q and q != "*":
                record_use_bg(mddb, b.mddb_collection, used)
        # Writable memory banks win score ties over read-only KB banks —
        # a same-scoring cms doc must not bury a people/ general fact.
        # Bulk-archive hits (devin/* session dumps) sort below everything
        # real regardless of score — tagged so the model can say so.
        writable_banks = {b.name for b in banks if b.writable}
        for h in hits:
            if _demoted_in_all(h):
                h["archive"] = True
        hits.sort(key=lambda h: (not h.get("archive"),
                                 float(h.get("score") or 0),
                                 h.get("bank") in writable_banks),
                  reverse=True)
        hits = hits[: int(limit)]
        _log_search_timing(
            bank_spec if not fanout_all else "all", len(banks), per_bank_ms,
            time.perf_counter() - t_start, hits, degraded)
        if q and q != "*" and not include_inactive:
            _schedule_bge_shadow(
                q, banks, hits, instance=registry.instance,
                degraded=degraded)
        return {
            "bank": "all" if fanout_all else bank_spec,
            "count": len(hits),
            "hits": hits,
            "degraded": degraded,
        }

    # Resolve personal → person-scoped bank when speaker is identified
    if str(bank) == "personal" and person_entity:
        bank = registry.personal_bank_name(person_entity)
    b = registry.bank(str(bank))
    _check_bank_allowed(registry, b.name, person_entity)
    docs, degraded, elapsed = await _timed_bank_docs(
        mddb, b, q, limit, include_inactive)
    hits = []
    used_docs = []
    for doc in docs:
        hit = _doc_to_hit(b, doc, registry.instance, include_inactive)
        if hit is None:
            continue
        hits.append(hit)
        used_docs.append(doc)
        if len(hits) >= int(limit):
            break
    # Knowledge-circle bookkeeping: a real query that surfaced docs counts as
    # "applied" — the observable proxy for the doc being used in the answer.
    # Listing/audit queries (empty, '*', include_inactive) don't count.
    if used_docs and not include_inactive and q and q != "*":
        record_use_bg(mddb, b.mddb_collection, used_docs)
    _log_search_timing(
        b.name, 1, {b.name: round(elapsed * 1000)},
        time.perf_counter() - t_start, hits, degraded)
    if q and q != "*" and not include_inactive:
        _schedule_bge_shadow(
            q, [b], hits, instance=registry.instance, degraded=degraded)
    return {
        "bank": b.name,
        "count": len(hits),
        "hits": hits,
        "degraded": degraded,
    }


async def record_use(
    mddb: MddbClient, collection: str, doc: dict[str, Any]
) -> None:
    """Bump use_count/last_used on a surfaced doc. Best-effort bookkeeping —
    recall never depends on it succeeding."""
    key = doc.get("key")
    if not key:
        return
    meta = dict(doc.get("meta") or {})
    today = datetime.now(timezone.utc).date().isoformat()
    if _meta_first(meta, "last_used") == today:
        # Already bumped today — a recall fan-out would otherwise rewrite
        # this doc on every search, which stampedes MDDB's inline embed.
        return
    try:
        count = int(_meta_first(meta, "use_count") or 0)
    except ValueError:
        count = 0
    meta["use_count"] = [str(count + 1)]
    meta["last_used"] = [today]
    await mddb.update_document(collection, str(key), meta=meta)


async def _record_use_safe(
    mddb: MddbClient, collection: str, doc: dict[str, Any]
) -> None:
    try:
        await record_use(mddb, collection, doc)
    except Exception as exc:
        logger.debug("record_use failed for %r: %s", doc.get("key"), exc)


def record_use_bg(
    mddb: MddbClient, collection: str, docs: list[dict[str, Any]]
) -> None:
    """Fire-and-forget use accounting — never blocks or breaks recall.
    Silently skipped without a running loop (unit tests, sync callers)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for doc in docs:
        loop.create_task(_record_use_safe(mddb, collection, doc))


async def record_outcome(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    bank: str,
    key: str,
    outcome: str,
    note: str | None = None,
    session_id: str | None = None,
    person_entity: str | None = None,
) -> dict[str, Any]:
    """Record how applied knowledge turned out — the verify stage that
    closes the learn->apply loop. Sets meta outcome, nudges confidence per
    OUTCOME_CONFIDENCE_DELTA, and bumps last_verified on good outcomes.

    When *person_entity* is set and bank is 'personal', the outcome is
    recorded on the speaker's person-scoped bank."""
    if str(bank) == "personal" and person_entity:
        bank = registry.personal_bank_name(person_entity)
    b = registry.bank(str(bank))
    _check_bank_allowed(registry, b.name, person_entity)
    _check_person_scope_write(b, person_entity)
    outcome = str(outcome)
    delta = OUTCOME_CONFIDENCE_DELTA.get(outcome)
    if delta is None:
        raise ValueError(
            f"unknown outcome {outcome!r} "
            f"(allowed: {', '.join(sorted(OUTCOME_CONFIDENCE_DELTA))})"
        )
    doc = await mddb.get_document(b.mddb_collection, str(key))
    if doc is None:
        raise ValueError(f"no such document {key!r} in bank '{b.name}'")
    meta = dict(doc.get("meta") or {})
    today = datetime.now(timezone.utc).date().isoformat()
    meta["outcome"] = [outcome]
    meta["last_used"] = [today]
    try:
        conf = float(_meta_first(meta, "confidence") or 0.6)
    except ValueError:
        conf = 0.6
    meta["confidence"] = [str(round(min(1.0, max(0.05, conf + delta)), 2))]
    if delta > 0:
        meta["last_verified"] = [today]
    if note:
        meta["outcome_note"] = [str(note)]
    if session_id and session_id != "unknown":
        meta["outcome_session"] = [session_id]
    _warn_unknown_meta(registry, meta)
    res = await mddb.update_document(
        b.mddb_collection, str(key), meta=meta,
        durable=True, tool="ada_ops.outcome", session_id=session_id)
    _must(res, f"update {b.mddb_collection}/{key}")
    return _queued_note({
        "verb": "outcome",
        "bank": b.name,
        "key": str(key),
        "outcome": outcome,
        "confidence": meta["confidence"][0],
    }, res)


async def remember(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    instance: str,
    bank: str,
    text: str,
    key: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
    kind: str | None = None,
    valid_until: str | None = None,
    applies_to: list[str] | str | None = None,
    supersedes: str | None = None,
    session_id: str | None = None,
    person_entity: str | None = None,
    prime: bool | None = None,
) -> dict[str, Any]:
    """Write a memory to a bank: create, correct-in-place, or supersede.

    *prime*: when true the fact is injected into every new session's context
    (see session_prime_text); when None an existing doc's flag is preserved.

    When *person_entity* is set and bank is 'personal', the write is routed
    to the speaker's person-scoped bank (e.g. personal-kk) so personal
    memories never cross person boundaries.

    Sensitive content (documents, IDs, passports, named persons' private
    details) targeting a shared bank is rerouted to the writer's personal
    bank — shared banks are visible to every household speaker."""
    rerouted_from: str | None = None
    if str(bank) == "personal" and person_entity:
        bank = registry.personal_bank_name(person_entity)
    b = registry.bank(str(bank))
    if (b.scope == "shared" and person_entity
            and SENSITIVE_MEMORY_RE.search(f"{text} {subject or ''}")):
        rerouted_from = b.name
        b = registry.bank(registry.personal_bank_name(person_entity))
    _check_bank_allowed(registry, b.name, person_entity)
    _check_person_scope_write(b, person_entity)
    if str(kind or "note") not in b.kinds:
        raise ValueError(
            f"kind {kind!r} not allowed in bank '{b.name}' (allowed: {', '.join(b.kinds)})"
        )
    # Size caps (card ada-memory-write-caps): text is the FINAL stored
    # body in every path below — create, correct-in-place, and supersede
    # all replace content_md wholesale — so len(text) is the real size.
    refusal = memory_write_guard.check_write(
        bank=b.name, text=text,
        fields={"key": key, "subject": subject, "attribute": attribute,
                "supersedes": supersedes})
    if refusal is not None:
        return refusal
    today = datetime.now(timezone.utc).date().isoformat()
    scope = "shared" if b.scope == "shared" else instance
    applies = [applies_to] if isinstance(applies_to, str) else list(applies_to or [])

    if supersedes:
        old = await mddb.get_document(b.mddb_collection, str(supersedes))
        if old is None:
            raise ValueError(
                f"cannot supersede {supersedes!r}: no such document in bank '{b.name}'"
            )
        new_key = str(key) if key else f"{b.name}/{_slug(str(subject or text))}"
        if new_key == str(supersedes):
            # The derived key collides with the doc being superseded — writing
            # there would resurrect-then-supersede the same doc. Bump the key.
            base = new_key
            n = 2
            while await mddb.get_document(b.mddb_collection, new_key) is not None:
                new_key = f"{base}-{n}"
                n += 1
        meta = memory_meta(b, scope, today, kind, subject, attribute, valid_until, applies, session_id, prime=prime)
        _warn_unknown_meta(registry, meta)
        meta["supersedes"] = [str(supersedes)]
        res = await mddb.add_document(
            b.mddb_collection, new_key, "en", str(text), meta,
            durable=True, tool="ada_remember", session_id=session_id)
        _must(res, f"add {b.mddb_collection}/{new_key}")
        old_meta = dict(old.get("meta") or {})
        old_meta["status"] = ["superseded"]
        old_meta["superseded_by"] = [new_key]
        res2 = await mddb.update_document(
            b.mddb_collection, str(supersedes), meta=old_meta,
            durable=True, tool="ada_remember", session_id=session_id)
        _must(res2, f"supersede-mark {b.mddb_collection}/{supersedes}")
        await memory_write_guard.warn_if_bank_large(
            mddb, b.mddb_collection, b.name)
        return _queued_note({
            "verb": "supersede",
            "bank": b.name,
            "key": new_key,
            "superseded": str(supersedes),
        }, res if write_outbox.is_queued(res) else res2)

    # Find-then-update: an existing active doc about the same
    # subject/attribute (or at the requested key) is corrected in place.
    target_key = str(key) if key else None
    existing = None
    if target_key:
        existing = await mddb.get_document(b.mddb_collection, target_key)
    elif subject:
        filt: dict[str, list[str]] = {"subject": [str(subject)], "status": ["active"]}
        if attribute:
            filt["attribute"] = [str(attribute)]
        docs = await mddb.search_documents(
            b.mddb_collection, "*", filter_meta=filt, limit=1
        )
        docs = [d for d in docs if doc_effective_status(d) == "active"]
        if docs:
            target_key = docs[0].get("key")
            existing = docs[0]
    if not target_key:
        target_key = f"{b.name}/{_slug(str(subject or text))}"
        existing = await mddb.get_document(b.mddb_collection, target_key)

    if existing is None:
        # Dedupe: a near-identical active memory counts as the same fact —
        # correct it in place rather than stacking a duplicate.
        sims = await mddb.vector_search(
            collection=b.mddb_collection,
            query=str(text),
            limit=1,
            filter_meta={"status": ["active"]},
            threshold=ADA_BANK_UPDATE_THRESHOLD,
        )
        if sims and doc_effective_status(sims[0]) == "active":
            target_key = sims[0].get("key")
            existing = sims[0]
            logger.info(
                "ada_remember dedupe: %r matched %r (score %.3f)",
                str(text)[:60], target_key, sims[0].get("score") or 0,
            )

    meta = memory_meta(b, scope, today, kind, subject, attribute, valid_until, applies, session_id, prime=prime)
    _warn_unknown_meta(registry, meta)
    if existing is not None:
        old_meta = dict(existing.get("meta") or {})
        for keep in ("valid_from", "supersedes", "superseded_by"):
            if keep in old_meta:
                meta[keep] = old_meta[keep]
        if prime is None and "prime" in old_meta:
            meta["prime"] = old_meta["prime"]
        if kind is None and "kind" in old_meta:
            meta["kind"] = old_meta["kind"]
        res = await mddb.update_document(
            b.mddb_collection, target_key, content_md=str(text), meta=meta,
            durable=True, tool="ada_remember", session_id=session_id)
        _must(res, f"correct {b.mddb_collection}/{target_key}")
        out = {"verb": "correct", "bank": b.name, "key": target_key}
    else:
        res = await mddb.add_document(
            b.mddb_collection, target_key, "en", str(text), meta,
            durable=True, tool="ada_remember", session_id=session_id)
        _must(res, f"add {b.mddb_collection}/{target_key}")
        out = {"verb": "create", "bank": b.name, "key": target_key}
    await memory_write_guard.warn_if_bank_large(
        mddb, b.mddb_collection, b.name)
    out = _queued_note(out, res)
    if rerouted_from:
        out["rerouted_from"] = rerouted_from
        out["note"] = (f"sensitive content kept in {b.name} "
                       f"(your personal bank) instead of shared '{rerouted_from}'")
    return out


VOCAB_DOC_KEY = "vocab/log"


async def vocab_append(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    person_entity: str | None,
    text: Any,
    note: Any = None,
) -> dict[str, Any]:
    """Append a term-coaching entry to the speaker's personal vocab log
    (vocab/log in their own personal bank — KK's notes land in
    personal-kk, Tony's in personal-tony). Not confirmation-gated:
    append-only, scoped to the caller's own bank. Shared by the
    ada_remember kind='vocab' path and pending-lane approval
    (memory_pending route='vocab')."""
    bank = persona_bank_for(registry, person_entity)
    if bank is None:
        raise PermissionError(
            "vocab notes need a personal bank — this identity has none"
        )
    entry = str(text or "").strip()
    note_s = str(note).strip() if note else ""
    if not entry:
        raise ValueError("vocab memory requires text ('term → correction')")
    today = datetime.now(timezone.utc).date().isoformat()
    line = f"- {entry} — {today}" + (f" ({note_s})" if note_s else "")
    doc = await mddb.get_document(bank.mddb_collection, VOCAB_DOC_KEY)
    body = ((doc or {}).get("contentMd") or doc and doc.get("content_md") or "")
    if not body.strip():
        body = "# Vocabulary — terms I heard, gently corrected\n"
    if line not in body:
        body = body.rstrip("\n") + "\n" + line + "\n"
    # Cap the FINAL merged doc, not the appended line — a small
    # addition to a near-full vocab log still refuses (card
    # ada-memory-write-caps edge case).
    refusal = memory_write_guard.check_write(bank=bank.name, text=body)
    if refusal is not None:
        return refusal
    meta = {
        "kind": ["vocab"], "subject": ["persona"],
        "status": ["active"], "scope": ["instance"],
        "last_verified": [today],
    }
    if doc is None:
        ok = await mddb.add_document(
            bank.mddb_collection, VOCAB_DOC_KEY, "en", body, meta)
    else:
        ok = await mddb.update_document(
            bank.mddb_collection, VOCAB_DOC_KEY, content_md=body, meta=meta)
    if not ok:
        return {"status": "error", "error": "mddb write failed"}
    return {"status": "noted", "bank": bank.name, "key": VOCAB_DOC_KEY,
            "entry": entry}


# Keys shaped like speaker-profile paths (speaker/kk, voiceprint/พรศิริ)
# never name memory docs — voiceprints live in the speaker profile store
# (backend/speaker_id.py), which ada_forget cannot reach. Redirecting
# beats a confusing "no such document" or, worse, a wrong guess.
_SPEAKER_PROFILE_KEY_RE = re.compile(
    r"^\s*(?:speaker_?profiles?|speakers?|voice_?prints?|voices?)\s*[/_.:\-]",
    re.IGNORECASE)


def _enrolled_profile_named(key: str) -> str | None:
    """Resolve a memory key to an enrolled speaker profile name, or None.
    Only consulted when no doc exists for the key — a real memory doc
    always wins over the lookalike check."""
    try:
        from backend.speaker_id import SpeakerIdentifier
        return SpeakerIdentifier.get().resolve_name(key)
    except Exception:
        return None


async def forget(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    bank: str,
    key: str,
    reason: str | None = None,
    session_id: str | None = None,
    person_entity: str | None = None,
) -> dict[str, Any]:
    """Retract a memory: status becomes retracted; the doc stays auditable.

    When *person_entity* is set and bank is 'personal', the retraction
    targets the speaker's person-scoped bank."""
    if _SPEAKER_PROFILE_KEY_RE.match(str(key or "")):
        raise ValueError(
            f"'{key}' is a speaker-profile path, not a memory key — "
            "voiceprints live in the speaker profile store, not memory "
            "banks. Use the speaker_profiles tool (action='list' to see "
            "profiles, action='remove' name=... to delete one) instead "
            "of ada_forget.")
    if str(bank) == "personal" and person_entity:
        bank = registry.personal_bank_name(person_entity)
    b = registry.bank(str(bank))
    _check_bank_allowed(registry, b.name, person_entity)
    _check_person_scope_write(b, person_entity)
    doc = await mddb.get_document(b.mddb_collection, str(key))
    if doc is None:
        profile = _enrolled_profile_named(str(key))
        if profile is not None:
            raise ValueError(
                f"no memory doc {key!r} in bank '{b.name}' — but '{key}' "
                f"resolves to enrolled voiceprint '{profile}'. "
                "Voiceprints are not memory docs; use the "
                "speaker_profiles tool (action='remove' name=...) "
                "instead of ada_forget.")
        raise ValueError(f"no such document {key!r} in bank '{b.name}'")
    meta = dict(doc.get("meta") or {})
    meta["status"] = ["retracted"]
    meta["last_verified"] = [datetime.now(timezone.utc).date().isoformat()]
    if reason:
        meta["retracted_reason"] = [str(reason)]
    if session_id and session_id != "unknown":
        meta["retracted_by_session"] = [session_id]
    res = await mddb.update_document(
        b.mddb_collection, str(key), meta=meta,
        durable=True, tool="ada_forget", session_id=session_id)
    _must(res, f"retract {b.mddb_collection}/{key}")
    return _queued_note(
        {"verb": "retract", "bank": b.name, "key": str(key)}, res)


# ---------- conversational persona ----------

PERSONA_DOC_KEY = "persona/self"


def persona_bank_for(
    registry: MemoryBankRegistry, identity: str | None
) -> "MemoryBank | None":
    """The writable personal-ish bank that stores this identity's persona
    doc: their person-scoped bank, else the default 'personal' — but only
    when the identity's policy actually allows writing there."""
    name = registry.personal_bank_name(identity)
    try:
        b = registry.bank(name)
        if b.writable and registry.bank_allowed(name, identity):
            return b
    except KeyError:
        pass
    for b in registry.banks_for_person(identity).values():
        if b.scope == "person" and b.writable:
            return b
    return None


def _persona_defaults(registry: MemoryBankRegistry) -> dict[str, Any]:
    return {
        k: v.get("default")
        for k, v in (registry.persona.get("knobs") or {}).items()
    }


def _validate_persona_knob(registry: MemoryBankRegistry, knob: str, value: Any) -> Any:
    defs = registry.persona.get("knobs") or {}
    spec = defs.get(knob)
    if spec is None:
        raise ValueError(
            f"unknown persona knob {knob!r} (allowed: {', '.join(defs) or 'none'})"
        )
    if spec.get("values") and str(value) not in spec["values"]:
        raise ValueError(f"{knob} must be one of {', '.join(spec['values'])}")
    if spec.get("type") == "bool":
        return str(value).lower() in ("true", "1", "yes", "on")
    if spec.get("type") == "string":
        value = str(value).strip()
        if len(value) > int(spec.get("max", 200)):
            raise ValueError(f"{knob} exceeds {spec.get('max')} chars")
        return value
    return str(value)


async def get_persona(
    mddb: MddbClient, registry: MemoryBankRegistry, identity: str | None
) -> dict[str, Any]:
    """Effective persona: SSOT defaults overlaid with the speaker's stored
    knobs (doc 'persona/self' in their personal bank)."""
    knobs = _persona_defaults(registry)
    bank = persona_bank_for(registry, identity)
    result: dict[str, Any] = {
        "knobs": knobs, "custom": [], "bank": bank.name if bank else None,
    }
    if bank is None:
        return result
    doc = await mddb.get_document(bank.mddb_collection, PERSONA_DOC_KEY)
    if not doc:
        return result
    for line in str(doc.get("content_md") or "").splitlines():
        k, sep, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if sep and k in knobs and v != str(_persona_defaults(registry).get(k)):
            knobs[k] = v
            result["custom"].append(k)
    return result


async def set_persona(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    identity: str | None,
    knob: str,
    value: Any,
) -> dict[str, Any]:
    """Set one persona knob for this identity; persists to 'persona/self'."""
    value = _validate_persona_knob(registry, knob, value)
    bank = persona_bank_for(registry, identity)
    if bank is None:
        raise PermissionError(
            "persona needs a writable personal bank — this identity has none"
        )
    defs = registry.persona.get("knobs") or {}
    cur = await get_persona(mddb, registry, identity)
    cur["knobs"][knob] = value
    lines = [
        f"{k}: {v}"
        for k, v in cur["knobs"].items()
        if str(v) != str(defs.get(k, {}).get("default"))
    ]
    meta = {
        "kind": ["preference"], "subject": ["persona"],
        "status": ["active"], "scope": ["instance"],
        "last_verified": [datetime.now(timezone.utc).date().isoformat()],
    }
    content = "\n".join(lines) or "# defaults"
    if await mddb.get_document(bank.mddb_collection, PERSONA_DOC_KEY) is None:
        res = await mddb.add_document(
            bank.mddb_collection, PERSONA_DOC_KEY, "en", content, meta,
            durable=True, tool="ada_persona")
        _must(res, f"add {bank.mddb_collection}/{PERSONA_DOC_KEY}")
        verb = "create"
    else:
        res = await mddb.update_document(
            bank.mddb_collection, PERSONA_DOC_KEY, content_md=content,
            meta=meta, durable=True, tool="ada_persona")
        _must(res, f"update {bank.mddb_collection}/{PERSONA_DOC_KEY}")
        verb = "correct"
    return _queued_note({"verb": verb, "bank": bank.name, "key": PERSONA_DOC_KEY,
                         "knob": knob, "value": value, "knobs": cur["knobs"]},
                        res)


async def reset_persona(
    mddb: MddbClient, registry: MemoryBankRegistry, identity: str | None
) -> dict[str, Any]:
    """Clear stored knobs back to SSOT defaults."""
    bank = persona_bank_for(registry, identity)
    if bank is None:
        raise PermissionError(
            "persona needs a writable personal bank — this identity has none"
        )
    meta = {
        "kind": ["preference"], "subject": ["persona"],
        "status": ["active"], "scope": ["instance"],
        "last_verified": [datetime.now(timezone.utc).date().isoformat()],
    }
    res: Any = None
    if await mddb.get_document(bank.mddb_collection, PERSONA_DOC_KEY) is not None:
        res = await mddb.update_document(
            bank.mddb_collection, PERSONA_DOC_KEY, content_md="# defaults",
            meta=meta, durable=True, tool="ada_persona")
        _must(res, f"reset {bank.mddb_collection}/{PERSONA_DOC_KEY}")
    return _queued_note({"verb": "reset", "bank": bank.name,
                         "knobs": _persona_defaults(registry)}, res)


def persona_instruction(knobs: dict[str, Any], registry: MemoryBankRegistry) -> str | None:
    """One-line prime text describing the speaker's non-default knobs."""
    defs = registry.persona.get("knobs") or {}
    custom = {
        k: v for k, v in knobs.items()
        if str(v) != str(defs.get(k, {}).get("default"))
    }
    if not custom:
        return None
    pairs = ", ".join(f"{k}={v}" for k, v in custom.items())
    return (
        f"(system) This speaker's saved style preferences: {pairs}. "
        "Apply them to tone and format from the first reply; "
        "use ada_persona to change them."
    )


# --- L0 hot headlines -------------------------------------------------------
# Recall-ladder tier 0: fresh report/digest one-liners injected at session
# open so Ada can say "what's new" immediately — detail is fetched lazily
# via cms_read action='get' only when the user asks. A headline is announced once
# (announced_at stamped when injected) then cools to the warm tier.
HEADLINE_COLLECTION = os.environ.get("ADA_CMS_COLLECTION", "ada-cms-pages")
HEADLINE_MAX_AGE_S = float(os.environ.get("ADA_HEADLINE_MAX_AGE_H", "72")) * 3600
HEADLINE_MAX = int(os.environ.get("ADA_HEADLINE_MAX", "5"))


def _headline_meta_first(doc: dict, name: str) -> str:
    v = (doc.get("meta") or {}).get(name)
    if isinstance(v, list) and v:
        return str(v[0])
    return str(v) if isinstance(v, str) else ""


async def fetch_headlines(mddb: MddbClient) -> list[dict[str, str]]:
    """Return recent CMS pages (kind page/report) that carry a speakable
    one-liner and have not been announced yet. Stamps announced_at on each
    returned doc — announcement is a once-ever event, not a ticker."""
    docs = await mddb.search_documents(
        collection=HEADLINE_COLLECTION,
        filter_meta={"kind": ["page", "report"], "status": ["active"]},
        limit=60,
    )
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - HEADLINE_MAX_AGE_S
    fresh: list[tuple[float, dict]] = []
    for d in docs or []:
        if _headline_meta_first(d, "announced_at"):
            continue
        text = (_headline_meta_first(d, "headline")
                or _headline_meta_first(d, "summary")).strip()
        if not text:
            continue
        # Indexes and digests are navigation/rollups — the individual
        # reports beneath them are the news; aggregates stay warm-tier.
        if (_headline_meta_first(d, "report_role") in ("index", "digest")
                or str(d.get("key") or "").endswith("-digest")):
            continue
        raw_updated = _headline_meta_first(d, "updated")
        try:
            ts = datetime.fromisoformat(raw_updated.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if ts < cutoff:
            continue
        fresh.append((ts, {
            "key": str(d.get("key") or ""),
            "title": _headline_meta_first(d, "title") or str(d.get("key") or ""),
            "text": text[:160],
        }))
    fresh.sort(key=lambda t: t[0], reverse=True)
    picks = [p for _, p in fresh[:HEADLINE_MAX]]
    # Stamp announced_at — the doc cools to the warm tier after one
    # injection regardless of whether Ada actually spoke it. Failures are
    # non-fatal: worst case the headline is offered again next session.
    stamp = now.isoformat(timespec="seconds")
    for p in picks:
        try:
            await mddb.update_document(
                HEADLINE_COLLECTION, p["key"],
                meta={"announced_at": [stamp]},
            )
        except Exception as exc:
            logger.debug("headline announce stamp failed for %s: %s", p["key"], exc)
    return picks


CMS_CATALOG_PRIME = os.environ.get(
    "ADA_CMS_CATALOG_PRIME", "").strip().lower() in ("1", "true", "yes")
CMS_CATALOG_MAX = int(os.environ.get("ADA_CMS_CATALOG_MAX", "40"))


async def fetch_cms_catalog(mddb: MddbClient) -> list[str]:
    """Compact always-on catalog of the lane CMS collection — opt-in via
    ADA_CMS_CATALOG_PRIME. Member lanes where CMS is the content memory:
    the model cannot answer "do you have X" honestly if the catalog is
    not in front of it. Deterministic — no tool call needed to know a
    page exists; cms_read still fetches detail."""
    docs = await mddb.search_documents(
        collection=HEADLINE_COLLECTION,
        filter_meta={"status": ["active"]},
        limit=CMS_CATALOG_MAX,
    )
    lines: list[str] = []
    for d in docs or []:
        key = str(d.get("key") or "")
        title = _headline_meta_first(d, "title") or key
        summary = _headline_meta_first(d, "summary")
        n_img = (d.get("content_md") or d.get("contentMd") or "").count("![")
        bits = []
        if summary:
            bits.append(summary[:80])
        if n_img:
            bits.append(f"{n_img} image" + ("s" if n_img != 1 else ""))
        line = f"- {key}: {title}"
        if bits:
            line += " — " + "; ".join(bits)
        lines.append(line)
    return lines


async def session_prime_text(
    mddb: MddbClient,
    registry: MemoryBankRegistry,
    summary: str | None = None,
    max_facts: int = 6,
    max_chars: int = 160,
    away_seconds: float | None = None,
    last_tail: str = "",
    person_entity: str | None = None,
    channel_context: str = "",
    context_reset: bool = False,
) -> str | None:
    """Build the session-start context injection: the rolling recent-sessions
    summary plus a few facts from the personal/general banks, so Ada starts
    aware of general info instead of blank. When away_seconds is set, a
    reconnect directive is prepended so the greeting matches how long the
    user was gone. Returns None when there is nothing worth injecting."""
    if os.environ.get("ADA_SESSION_PRIME", "1") in ("0", "false", "no"):
        return None
    parts: list[str] = []
    anon_note: str | None = None
    directive = reconnect_directive(
        away_seconds, last_tail, context_reset=context_reset)
    if directive:
        parts.append(directive)
    # Short reconnect (<1h): skip the recent-sessions summary too — like
    # bank facts it injects stale threads that drown the live tail.
    # Authoritative session identity — without it the model reconstructs
    # "who am I talking to" from memory/archive content and can mistake
    # the speaker for whoever the last summary mentioned (the KK bug).
    ident_note: str | None = None
    if person_entity:
        who = registry.identity_label(person_entity) or person_entity
        ident_note = (
            f"(system) You are talking to {who} — this device is registered "
            f"to {person_entity}. Greet and address them as {who}. Any other "
            "names in this context are other people or past topics, never "
            "the speaker. Only a speaker-identification event naming a "
            "different person overrides this."
        )
    else:
        # Anonymous device — only worth noting when other injected content
        # (summary, facts, tail) could carry names the model might mistake
        # for the current speaker; an empty prime stays None.
        anon_note = (
            "(system) No confirmed speaker identity — names in memory and "
            "session archives may refer to other people; greet neutrally "
            "and do not address anyone by name unless a speaker-"
            "identification event confirms who is speaking."
        )
    if summary and not (directive and (away_seconds or 0) < 3600):
        parts.append(f"Recent sessions: {summary.strip()}")
    # Unified voice+chat history (card ada-reads-text-chat): recent turns
    # from this owner's other channels — live sibling sessions and/or the
    # persisted shared tail. Always allowed through: it is capped by age,
    # so it is never stale thread content like bank facts can be.
    if str(channel_context or "").strip():
        parts.append(
            "Your other chat channels (the voice session, text chat and "
            "relays share one history — you can see these messages and "
            "the user may refer to them):\n"
            + channel_context.strip())
    # L0 hot headlines — fresh report/digest one-liners. Skipped on short
    # reconnects like facts; announced_at stamping means they surface once.
    if not (directive and (away_seconds or 0) < 3600):
        try:
            heads = await asyncio.wait_for(fetch_headlines(mddb), timeout=8.0)
        except Exception as exc:
            logger.debug("headlines prefetch failed: %s", exc)
            heads = []
        if heads:
            lines = ["New since the last session — mention briefly if "
                     "the user asks what's new or it fits the moment; "
                     "fetch the page for detail only if asked:"]
            lines += [f"- {h['title']}: {h['text']}" for h in heads]
            parts.append("\n".join(lines))
    # CMS catalog prime — opt-in lane flag; the lane's page inventory is
    # reference state (like persona style), not staleable news, so it
    # primes on every session including short reconnects.
    if CMS_CATALOG_PRIME:
        try:
            catalog = await asyncio.wait_for(
                fetch_cms_catalog(mddb), timeout=8.0)
        except Exception as exc:
            logger.debug("cms catalog prime failed: %s", exc)
            catalog = []
        if catalog:
            parts.append(
                "Your CMS pages — read one for detail before saying you "
                "do not have the content or asking the user for it. "
                "To share a page or its images in chat, send its "
                "URLs with chat_send (image_url for images); to put "
                "it on screen, cast it:\n"
                + "\n".join(catalog))
    # Speaker's saved style preferences — applies even on short reconnects
    # (it's a style contract, not stale content).
    if person_entity:
        try:
            persona = await get_persona(mddb, registry, person_entity)
            line = persona_instruction(persona["knobs"], registry)
            if line:
                parts.append(line)
        except Exception as exc:
            logger.info("persona prime skipped: %s", exc)
    # Short reconnect (<1h): the conversation tail already carries the live
    # threads — bank facts would inject stale context and drown them.
    # First session or long gap: facts still prime cold-start awareness.
    facts: list[str] = []
    rules: list[str] = []
    if not (directive and (away_seconds or 0) < 3600):
        # Opt-in facts: docs flagged prime:true surface at every session
        # start regardless of bank fill order — device-name mappings and
        # other high-traffic facts that would otherwise need a tool call.
        # They render under a directive header ("apply these"), not the
        # passive "Known facts" list — background framing makes the model
        # treat mappings as trivia and still run entity searches.
        prime_seen: set[str] = set()
        prime_banks = list(dict.fromkeys(
            [registry.personal_bank_name(person_entity), "general", "home"]))
        for name in prime_banks:
            try:
                b = registry.bank(name)
            except KeyError:
                continue
            try:
                docs = await mddb.search_documents(
                    collection=b.mddb_collection,
                    filter_meta={"status": ["active"], "prime": ["true"]},
                    limit=4,
                )
            except Exception as exc:
                logger.debug("prime-flag fetch failed for %r: %s", name, exc)
                continue
            for doc in docs or []:
                body = str(doc.get("contentMd") or doc.get("content_md") or "").strip()
                key = str(doc.get("key") or "")
                meta = doc.get("meta") or {}
                # Transcript dumps must never be injected as "apply
                # these directly" guidance — same steer-the-model
                # hazard as the search-hit path.
                if is_transcript_shaped(
                        body, kind=_meta_first(meta, "kind"), key=key):
                    continue
                if body and key not in prime_seen:
                    prime_seen.add(key)
                    rules.append(body[:max_chars])
                if len(rules) >= 4:
                    break
            if len(rules) >= 4:
                break
        for name in ("personal", "general"):
            try:
                b = registry.bank(name)
            except KeyError:
                continue
            try:
                docs = await mddb.search_documents(
                    collection=b.mddb_collection,
                    filter_meta={"status": ["active"]},
                    limit=max_facts,
                )
            except Exception as exc:
                logger.debug("session prime listing failed for %r: %s", name, exc)
                continue
            for doc in docs or []:
                body = str(doc.get("contentMd") or doc.get("content_md") or "").strip()
                key = str(doc.get("key") or "")
                meta = doc.get("meta") or {}
                if is_transcript_shaped(
                        body, kind=_meta_first(meta, "kind"), key=key):
                    continue
                if body and key not in prime_seen:
                    prime_seen.add(key)
                    facts.append(body[:max_chars])
                if len(facts) >= max_facts:
                    break
            if len(facts) >= max_facts:
                break
    if facts:
        parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in facts))
    if rules:
        parts.append(
            "Standing guidance — apply these directly when the user mentions "
            "the topic; do not look the entities up again:\n"
            + "\n".join(f"- {r}" for r in rules))
    # Identity goes LAST — the freshest directive when the greeting is
    # generated; the rolling summary often names other people (the
    # "Hi KK" bug: a KK-heavy summary outcompeted the device binding).
    if ident_note:
        parts.append(ident_note)
    if anon_note and parts:
        parts.insert(1 if directive else 0, anon_note)
    if not parts:
        return None
    header = (
        "(system) Reconnect context — act on the session note below, treat "
        "the rest as background information:"
        if directive
        else "(system) Session context — background information only, do not "
        "announce or answer it:"
    )
    return header + "\n" + "\n\n".join(parts)

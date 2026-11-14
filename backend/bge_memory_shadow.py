"""Opt-in, non-blocking BGE retrieval shadow for Ada memory searches.

The shadow is isolated from the user-visible result: it reads the local BGE
plane, reranks a bounded candidate set, and logs aggregate comparison metrics
only. It never writes to MDDB or logs query/document content. It is disabled
unless ``ADA_BGE_SHADOW_ENABLED=1`` and both endpoint URLs are configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from backend.mddb_client import is_ops_collection

logger = logging.getLogger("tools")

TOP_K = 10
MAX_INFLIGHT = 2
MAX_PARALLEL_COLLECTIONS = 4
MAX_COLLECTIONS = 32
_tasks: set[asyncio.Task] = set()


def _enabled() -> bool:
    return os.environ.get("ADA_BGE_SHADOW_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def schedule(
    query: str,
    banks: list[Any],
    baseline_hits: list[dict[str, Any]],
    *,
    instance: str,
    baseline_degraded: bool = False,
) -> bool:
    """Queue shadow work without awaiting it or changing the baseline hits."""
    if not _enabled() or not query.strip() or not banks:
        return False
    if len(banks) > MAX_COLLECTIONS:
        logger.info(
            "bge_memory_search shadow status=dropped reason=collection_limit")
        return False
    if not os.environ.get("ADA_BGE_SHADOW_MDDB_URL", "").strip():
        return False
    if not os.environ.get("ADA_BGE_SHADOW_RERANK_URL", "").strip():
        return False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    if len(_tasks) >= MAX_INFLIGHT:
        logger.info("bge_memory_search shadow status=dropped reason=busy")
        return False

    baseline_ids = [
        (str(hit.get("bank") or (banks[0].name if len(banks) == 1 else "")),
         str(hit.get("key") or ""))
        for hit in baseline_hits
        if hit.get("key")
    ]
    task = loop.create_task(_run_safely(
        query=query,
        banks=list(banks),
        baseline_ids=baseline_ids,
        instance=instance,
        baseline_degraded=baseline_degraded,
    ))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


async def _run_safely(**kwargs: Any) -> None:
    try:
        await run_shadow(**kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # shadow errors must never affect memory_search
        logger.info(
            "bge_memory_search shadow status=error error_type=%s",
            type(exc).__name__,
        )


async def run_shadow(
    *,
    query: str,
    banks: list[Any],
    baseline_ids: list[tuple[str, str]],
    instance: str,
    baseline_degraded: bool = False,
    client_factory: Any = httpx.AsyncClient,
) -> dict[str, Any]:
    """Run one bounded shadow query and return only non-content metrics."""
    started = time.perf_counter()
    base_url = os.environ["ADA_BGE_SHADOW_MDDB_URL"].rstrip("/")
    rerank_url = os.environ["ADA_BGE_SHADOW_RERANK_URL"]
    timeout_s = min(10.0, max(0.1, float(os.environ.get(
        "ADA_BGE_SHADOW_TIMEOUT_S", "3.0"))))
    deadline = started + timeout_s
    semaphore = asyncio.Semaphore(MAX_PARALLEL_COLLECTIONS)

    async with client_factory(timeout=httpx.Timeout(timeout_s)) as client:
        async def fetch(bank: Any) -> tuple[list[dict[str, Any]], bool]:
            try:
                async with semaphore:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return [], False
                    filter_meta = None
                    if bank.writable:
                        from backend.memory_ops import _draft_bank_allowed
                        statuses = (
                            ["active", "draft"]
                            if _draft_bank_allowed(bank) else ["active"]
                        )
                        filter_meta = {"status": statuses}
                    payload: dict[str, Any] = {
                        "collection": bank.mddb_collection,
                        "query": query,
                        "topK": TOP_K,
                        "includeContent": True,
                    }
                    if filter_meta:
                        payload["filterMeta"] = filter_meta
                    response = await client.post(
                        f"{base_url}/vector-search",
                        json=payload,
                        timeout=max(0.1, remaining),
                    )
                    response.raise_for_status()
                    results = response.json().get("results") or []
                    from backend.memory_ops import _doc_to_hit
                    candidates = []
                    for row in results:
                        document = dict(row.get("document") or {})
                        document["score"] = row.get("score")
                        hit = _doc_to_hit(
                            bank, document, instance, include_inactive=False)
                        if hit is None:
                            continue
                        candidates.append({
                            "bank": bank.name,
                            "key": str(hit.get("key") or ""),
                            "text": str(document.get("contentMd")
                                        or document.get("content_md") or ""),
                            "score": float(row.get("score") or 0.0),
                        })
                    return candidates, False
            except Exception:
                return [], True

        eligible_banks = [
            bank for bank in banks
            if not is_ops_collection(bank.mddb_collection)
        ]
        batches = await asyncio.gather(*(fetch(bank) for bank in eligible_banks))
        collection_errors = sum(1 for _docs, failed in batches if failed)
        candidates_by_id: dict[tuple[str, str], dict[str, Any]] = {}
        for candidates, _failed in batches:
            for candidate in candidates:
                identity = (candidate["bank"], candidate["key"])
                previous = candidates_by_id.get(identity)
                if previous is None or candidate["score"] > previous["score"]:
                    candidates_by_id[identity] = candidate

        dense = sorted(
            candidates_by_id.values(),
            key=lambda item: (-item["score"], item["bank"], item["key"]),
        )[:TOP_K]
        dense_ms = round((time.perf_counter() - started) * 1000)
        rerank_ms = 0
        if not dense:
            metrics = {
                "status": "empty", "candidate_count": 0,
                "baseline_count": len(baseline_ids),
                "baseline_degraded": baseline_degraded,
            }
        else:
            rerank_started = time.perf_counter()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("shadow deadline exceeded before rerank")
            payload = {
                "query": query,
                "docs": [{"id": str(i), "text": item["text"]}
                         for i, item in enumerate(dense)],
            }
            response = await client.post(
                rerank_url, json=payload, timeout=max(0.1, remaining))
            response.raise_for_status()
            scores = response.json().get("scores") or []
            if len(scores) != len(dense):
                raise ValueError("reranker returned an unexpected score count")
            reranked = sorted(
                zip(dense, (float(score) for score in scores)),
                key=lambda pair: pair[1],
                reverse=True,
            )
            reranked_ids = [
                (item["bank"], item["key"])
                for item, _score in reranked
            ]
            overlap_at_5 = len(
                set(reranked_ids[:5]) & set(baseline_ids[:5]))
            metrics = {
                "status": "ok",
                "candidate_count": len(candidates_by_id),
                "reranked_count": len(reranked),
                "baseline_count": len(baseline_ids),
                "overlap_at_5": overlap_at_5,
                "baseline_degraded": baseline_degraded,
            }
            rerank_ms = round((time.perf_counter() - rerank_started) * 1000)

    metrics["collections_scanned"] = len(eligible_banks)
    metrics["collection_errors"] = collection_errors
    metrics["dense_ms"] = dense_ms
    metrics["rerank_ms"] = rerank_ms
    metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
    logger.info(
        "bge_memory_search shadow status=%s candidates=%d reranked=%d "
        "baseline_hits=%d overlap_at_5=%d collections=%d collection_errors=%d "
        "dense_ms=%d rerank_ms=%d total_ms=%d "
        "baseline_degraded=%s",
        metrics["status"], metrics.get("candidate_count", 0),
        metrics.get("reranked_count", 0), metrics.get("baseline_count", 0),
        metrics.get("overlap_at_5", 0), metrics["collections_scanned"],
        metrics["collection_errors"], metrics["dense_ms"],
        metrics["rerank_ms"], metrics["total_ms"],
        metrics["baseline_degraded"],
    )
    return metrics

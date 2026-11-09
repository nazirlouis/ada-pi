"""nest_status — one-call DevNest/KK-Mac queue status for Ada.

Reads the published nest-queue-health snapshot (written every 2 min on
tony-dell by scripts/ops/nest-queue-health.py and mirrored to the
web-served snapshots dir) and returns a compact, speech-friendly status:
per-lane verdicts, KK-Mac worker heartbeat freshness, pack counts, and
how stale the snapshot itself is.

Use when Tony asks what any host/queue is doing — "is kk-mac working?",
"did the video queue drain?", "is anything stuck?" — instead of a
Devin/ssh lookup. Read-only; never mutates queues or hosts.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

STATUS_URL = os.environ.get(
    "ADA_NEST_STATUS_URL",
    "https://tony-dell.taila0626a.ts.net/snapshots/"
    "nest-queue-status.json",
)
TIMEOUT = 15.0
# Snapshot is regenerated every ~2 min; older than this means the
# producer stopped — report that rather than trusting stale numbers.
SNAP_STALE_S = 15 * 60

DECLARATION = {
    "name": "nest_status",
    "description": (
        "Live DevNest queue + KK-Mac worker status: per-lane verdicts, "
        "heartbeat, pack counts. Read-only; ~2 min old."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _age(now: float, epoch: float | None) -> int | None:
    return int(now - epoch) if epoch else None


def _fmt_age(secs: float | int | None) -> str:
    if secs is None:
        return "unknown"
    secs = int(secs)
    if secs < 120:
        return f"{secs}s"
    if secs < 7200:
        return f"{secs // 60}m"
    return f"{secs // 3600}h"


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(STATUS_URL)
    except (httpx.HTTPError, TimeoutError) as exc:
        return {
            "ok": False,
            "error": (
                f"I couldn't reach the nest status page "
                f"({exc.__class__.__name__}) — the snapshot itself may "
                "be down, so treat any status claim as unverified"),
        }
    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": f"the nest status page returned HTTP {resp.status_code}",
        }
    try:
        snap = resp.json()
    except Exception:
        return {"ok": False,
                "error": "the nest status page returned an unreadable body"}
    if not isinstance(snap, dict):
        return {"ok": False,
                "error": "the nest status page returned an unexpected body"}

    now = time.time()
    snap_age = _age(now, snap.get("checked_at"))

    kk = snap.get("kk_llm") or {}
    video = snap.get("video") or {}
    worker = snap.get("kk_worker") or {}
    gen = snap.get("packs") or {}

    return {
        "ok": True,
        "snapshot_age": _fmt_age(snap_age),
        "snapshot_stale": snap_age is None or snap_age > SNAP_STALE_S,
        "verdicts": snap.get("verdicts") or {},
        "kk_mac_worker": {
            "host": worker.get("host"),
            "heartbeat_age": _fmt_age(worker.get("age_s")),
            "stale": worker.get("stale"),
            "ollama_ok": worker.get("ollama_ok"),
            "load_1m": worker.get("load_1m"),
            "disk_free_gb": worker.get("disk_free_gb"),
            "cpu_temp_c": worker.get("cpu_temp_c"),
            "node_busy": worker.get("node_busy"),
            "error": worker.get("error"),
        },
        "kk_llm_queue": {
            "packs_total": kk.get("total"),
            "done": kk.get("done"),
            "failed": kk.get("failed"),
            "pending": kk.get("pending"),
            "active_claims": kk.get("active_claims"),
            "specs_received": kk.get("inbox_specs"),
        },
        "video_queue": {
            "packs_total": video.get("total"),
            "done": video.get("done"),
            "failed": video.get("failed"),
            "pending": video.get("pending"),
            "active_claims": video.get("active_claims"),
            "heartbeat_age": _fmt_age(
                (video.get("heartbeat") or {}).get("age_s")),
        },
        "hosts": {
            h: {k: v for k, v in (info or {}).items()
                if k in ("role", "age_s", "stale", "cpu_temp_c",
                         "load_1m", "disk_free_gb", "ollama_ok")}
            for h, info in ((snap.get("hosts") or {})
                            .get("by_host") or {}).items()
        },
        "general_queue": {
            "packs_total": gen.get("total"),
            "done": gen.get("done"),
            "pending": gen.get("pending"),
            "verdict": snap.get("verdict"),
        },
        "errors": snap.get("errors") or [],
    }

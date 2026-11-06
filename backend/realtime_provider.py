"""Gemini Live provider for the browser audio bridge."""

from __future__ import annotations

import abc
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types

from backend import chaba_memory, lane_budget, tools_loader, voice_config, vms_camera, vision_describe
from backend import command_lane, write_outbox
from backend.instance import ada_instance_id
from backend.conversation_memory import ConversationMemory
from backend.speech_sanitize import sanitize_speech, split_artifact_tail
from backend.tool_runner import (
    ToolRunner,
    CALENDAR_WRITE_TOOLS,
    CAPTURE_CONFIRMED_TOOLS,
    CMS_WRITE_TOOLS,
    CONTROL_TOOLS,
    DEVIN_CONFIRMED_TOOLS,
    DOC_CONFIRMED_TOOLS,
    MEMORY_WRITE_TOOLS,
    _resolve_alias,
    _alias_call_args,
    _PERSONA_VOICE_ACTIONS,
    normalize_tool_result,
)

# Tools whose call requires an explicit user confirmation. The Jev advisory
# probe fires on every one of these calls — not only when the model asserted
# confirmed=true — so divergence data covers the common path too.
_CONFIRM_GATED_TOOLS = (
    CONTROL_TOOLS | MEMORY_WRITE_TOOLS | CALENDAR_WRITE_TOOLS
    | CMS_WRITE_TOOLS | DEVIN_CONFIRMED_TOOLS | CAPTURE_CONFIRMED_TOOLS
    | DOC_CONFIRMED_TOOLS | {"ada_enroll_speaker"}
)
from backend.usage_tracker import usage_ledger

logger = logging.getLogger("voice.provider")

LIST_HOME_DEVICES_MAX = int(os.environ.get("ADA_LIST_HOME_DEVICES_MAX", "60"))

# Display name for user/model-facing result text (OPS-0734): non-Ada
# lanes set ADA_PWA_NAME so tool results say e.g. Kraut, not Ada.
_ADA_DISPLAY_NAME = os.environ.get("ADA_PWA_NAME") or "Ada"

EXPRESSION_NAMES = (
    "neutral",
    "sassy",
    "amused",
    "skeptical",
    "annoyed",
    "mad",
    "concerned",
    "surprised",
    "mischievous",
    "serious",
    "alert",
)

# Calendar/task tool names and the instruction paragraph that describes them.
# Kept as constants so ADA_EXCLUDED_TOOLS can strip both declarations and
# instructions together (dark-launching the tools without confusing the model).
CALENDAR_TOOLS = {
    # tools-merge-calendar-plan (2026-10-05): 9 -> 3. The absorbed names
    # (calendar_list_calendars/calendar_list_events/calendar_freebusy/
    # calendar_create_event/calendar_delete_event/calendar_shift_overdue/
    # ada_daily_summary/ada_weekly_comparison) live on as
    # tool_runner._ALIASES.
    "calendar_read", "calendar_write", "plan_day",
    # tools-merge-tasks-status (2026-10-05): tasks_add/tasks_list/
    # tasks_complete/tasks_move collapsed into tasks(action=add|list|done|
    # move); the absorbed names live on as tool_runner._ALIASES.
    "tasks",
    "ada_resolve_action",
}

# Same constant pattern as CALENDAR_TOOLS: lets ADA_EXCLUDED_TOOLS strip the
# CMS declarations and their instruction paragraph together.
CMS_TOOLS = {
    # tools-merge-cms (2026-10-05): 7 -> 3. The absorbed names
    # (cms_list_pages/cms_get_page/cms_verify_page/cms_note_update/
    # cms_delete_page/cms_automation) live on as tool_runner._ALIASES.
    "cms_publish_page", "cms_read", "cms_edit",
}

CALENDAR_INSTRUCTIONS = (
    " You have calendar and task tools backed by the user's configured providers: "
    "calendar_read action='calendars' shows which calendars exist, "
    "calendar_read action='events' and action='freebusy' read the schedule, "
    "plan_day returns one merged view of a day's events plus open tasks and that "
    "day's session digest (period='today'/'tomorrow', or day='yesterday'/a "
    "YYYY-MM-DD date for a recap of that day), period='week' compares the last 7 "
    "days of daily digests into a weekly trend view — use it when the user asks "
    "for a recap of a day or how the week went, "
    "calendar_write action='create'/'delete'/'shift' modifies the calendar ('shift' "
    "moves every overdue task plus already-ended event to a new day in one call, "
    "to='tomorrow' default), and "
    "tasks manages the task list — action='list' reads open tasks (free read), "
    "action='add'/'done'/'move' add, complete, or reschedule and need "
    "confirmed=true like other calendar writes. "
    "For any schedule question call calendar_read action='events' or plan_day first "
    "and answer from the result; never recite a schedule from memory. "
    "Interpret relative dates ('tomorrow', 'Friday') in the user's local timezone and "
    "echo the resolved day+date in your reply (see the date-echo rule). "
    "Before creating or deleting an event, shifting overdue items, or adding or "
    "completing a task, restate the "
    "exact details (title, resolved date with weekday, time — never a bare 'tomorrow'/"
    "'พรุ่งนี้') and get an explicit yes, then call the tool with "
    "confirmed=true — writes are enforced server-side. "
    "If a calendar tool result has ok=false or an 'error' field, the provider is unreachable "
    "or unauthenticated — say that plainly (it needs re-authentication when the error mentions "
    "auth/token) and stop retrying; never present a missing or empty events/tasks list as the "
    "real schedule. A non-empty 'errors' list means a provider didn't answer — mention the view "
    "is partial. Event and task ids are provider-qualified (e.g. 'google:primary/abc') — "
    "pass them back exactly as returned. "
    "When the session context lists pending suggestions from earlier conversations, offer each "
    "once, briefly and early in the conversation; if the user accepts, restate the details, call "
    "the matching calendar/task tool with confirmed=true, then call ada_resolve_action with the "
    "shown key and resolution 'applied'. If declined, call ada_resolve_action with 'dismissed'. "
    "When the user states a plan, appointment, or reminder intention unprompted, proactively "
    "offer to put it on the calendar or task list instead of waiting to be asked. "
    "When a day has no sessions the plan_day digest says so rather than guessing."
)

CMS_INSTRUCTIONS = (
    " You maintain the user's miniapp — a small multi-page site whose pages you own. "
    "cms_read action='list' lists existing pages and reports (kind per row, most "
    "recently updated first) with their language variants, "
    "action='search' finds a page by free text (query=) when you don't know its slug, and "
    "action='get' reads one by key (its page slug) — get reaches any slug you "
    "know even when list truncates, "
    "cms_publish_page creates or fully replaces a page (slugs are lowercase, e.g. 'pool-notes'; "
    "en/th variants coexist — publish the user's language plus the other when asked), "
    "cms_edit action='note' appends a timeline note to an existing page without replacing content, and "
    "cms_edit action='delete' removes one. "
    "Every publish must carry the report meta contract — summary (a one-line brief of the gist, "
    "~240 chars max; it is the reports-index row), domain (grouping tag like 'flood'/'health'/"
    "'bench'/'news'), fresh_for (staleness hint '30m'/'1h'/'6h'/'1d'), and confidence "
    "(high|medium|low|unverified) — cms_publish_page REJECTS calls missing them; updated and "
    "timeline are stamped automatically. "
    "Page content is written as markdown, html, yaml, or slides markdown. "
    "PUBLIC LANE: public=yes on cms_publish_page flags the page for the public website — the result returns public_url (surf-thailand.com/<tenant>/<slug>.html); it goes live within ~15 min via the static publisher. Share public_url for anyone outside the tailnet — view/cast URLs do not reach them. Only flag public when the owner asks for a shareable link. If a read result has no public_url, the page is NOT public — never guess or construct a surf-thailand.com URL yourself; flag public=yes (when the owner asked to share) and use the returned public_url, or tell the owner it is not published. "
    "For 'what's new', 'status', or inventory/library questions — 'what videos do we have cached', "
    "'what pages do we have', 'which documents are in the archive' — read the 'reports-index' page first — it lists every "
    "report with a one-line summary and staleness flag; only cms_read action='get' the linked page when the "
    "summary isn't enough. "
    "Video, dub, and cached-clip questions route to two pages you own: 'cached-videos-report' "
    "lists every playable cached clip and 'voice-dub-demos' holds the dub demos with the exact "
    "cast recipe — for 'the dubbed version', 'what videos do we have cached', or a dub-demo ask, "
    "cms_read action='get' those slugs and cast the local "
    "/apps/yt-live/*.mp4 URL with cast_to_screen, never a YouTube watch URL. "
    "READ ORDER — most distilled first, outside last: reports-index (or cms_read action='list' / "
    "ada_memory_search bank='cms'), then the linked page, then memory banks, and only then an "
    "outside source. Before web_search or ada_ops action='research' on any topic a page might already "
    "cover — status, incidents, news, research, anything previously reported — check your own "
    "pages FIRST; outside search is the fallback for what pages don't cover or what must be "
    "fresh, never the first read. "
    "The report-first ritual: before answering from a report, state its "
    "last-update summary and when it was written ('the flood report from 09:12 says…'). Then "
    "decide whether to drill deeper — your knobs are the report's fresh_for hint (stale → "
    "refresh via cms_edit action='note' after querying) and its confidence field (low/unverified → "
    "don't state it as settled). When a tool call gives you new information tied to a report, "
    "call cms_edit action='note' on that report and tell the user the diff — what changed and at what "
    "time ('added: flood moved to yellow at 14:05'). "
    "REPORT TRIAGE — when a report's state needs attention (reports-index shows stale/error/delta, "
    "confidence is low/unverified, or a report-watch inbox item flags it), do not wait for a question: "
    "lead with a crisp situation brief — what the report says, how fresh, and what it means in one "
    "breath. Then offer two or three concrete options (refresh the data, fix the producer, delegate "
    "to Devin, drop it) and ask which the user wants — never dump a finding without a proposed "
    "direction. Once the user picks, act: refresh via cms_edit note, fix directly if it is a content "
    "edit, or hand to Devin — devin action='dispatch' with a self-contained spec (repo, task, the "
    "chosen direction), restate it, get an explicit yes, then call with confirmed=true. The loop "
    "ends when the report is green again or the user dismisses it. "
    "SAVE-BACK — when an outside source (web_search, deep research, news headlines) supplies "
    "the answer, write the finding back to CMS the same turn: cms_edit action='note' on the matching "
    "page needs no confirmation; when no page covers it and the finding is worth keeping, "
    "offer a new page via cms_publish_page with the usual confirm. A repeat question must be "
    "answerable from your pages — never re-search for what you already found. "
    "LOOP-CLOSE — a promised write is a debt: when you tell the user you will update, add, or fix a page, complete that cms_edit/cms_publish_page call in the same turn — the promise is the consent; pass confirmed=true and do not defer it to a later turn. Before a conversation closes (goodbyes, 'anything else?', winding down), sweep what you said you would write — pending page edits, notes, corrections — and write them first; a session must not end with promised writes unsent. "
    "BOARD — kanban questions go through the 'kanban-brief' page first: its "
    "<!-- board-index --> block carries one compact row per open card (id | col | pri | "
    "updated | reqs | title). Only call kanban action='list'/'read' for what the index "
    "doesn't carry, and always kanban action='read' a card's current state before acting "
    "on it. On any card write (comment/move/respond) pass expected_updated=<the stamp "
    "you last read> — the tool refuses a write when the card changed underneath you; "
    "when it does, re-read, reconcile, and retry once if it still applies, otherwise "
    "tell the user the card moved. "
    "Inside markdown pages you can embed rich blocks as fenced code blocks: "
    "```chart <yaml echarts option> for 2D charts (line/bar/pie/scatter), "
    "```chart3d <yaml echarts-gl option> for 3D (surface3d/bar3d/scatter3d — set "
    "xAxis3D/yAxis3D/zAxis3D/grid3D), ```mermaid for diagrams (flowchart, sequence, "
    "timeline), and ```media <yaml {url, type, caption}> for images, video, or audio. "
    "chart/chart3d bodies are YAML echarts options (or {height: N, option: {...}}). "
    "When the user asks you to prepare a document or update a page, draft the content, "
    "restate the slug and title, get an explicit yes, then call the write tool with "
    "confirmed=true — writes are enforced server-side. "
    "You cannot see the rendered site: after publishing or updating a page, call "
    "cms_read action='verify' to check the content parses and confirm the structure, then "
    "tell the user the page is live (or fix it if verification failed)."
    " When the user reports an ongoing incident — a flood, outage, emergency, "
    "or similar — create or update a cms report page for it (a short status "
    "log, slug like 'flood-report'), check the live conditions with the "
    "home/weather tools, and keep the page current as new details arrive; "
    "the user's report is the consent — pass confirmed=true directly and do "
    "not ask for confirmation. "
    "Generated pages (news digests, flood reports) have an automation registry "
    "you control with cms_edit action='automate': op='list'/'get' are free reads; "
    "op='set', 'enable', 'disable', and 'run' adjust a page's feeds, refresh "
    "interval, relevance filter, language variants, or queue a regeneration — "
    "writes need confirmed=true like other CMS writes. When the user asks to "
    "refresh a generated page, prefer cms_edit action='automate' op='run' over "
    "republishing."
    "REPORT OPINION — when the user asks what you think of a report or CMS "
    "page: cms_read action='get' the page first (never opine on a page you "
    "have not read), then kanban action='read' report=<slug> to "
    "find the kanban card linked to it, then kanban "
    "action='comment' on that card with text starting '[opinion]' that "
    "cites the slug and the updated timestamp you just read. Answer aloud "
    "too — the comment is the durable record, the spoken take is for the "
    "moment. If the read reports several open cards it picked the most "
    "recently updated — say which card got the comment; if no card links "
    "the report, answer aloud and offer to file one (action='file')."
    " NEXT — when the user says 'next' without a specific card, call kanban action='next' first. Do not answer with a raw card list. The result checks current doing/review/backlog work, blockers, and safe candidates, then presents the returned next_menu choices. Wait for Tony's choice; choosing a menu item selects the next plan/review step only and never queues or dispatches a job automatically. "
    " KANBAN — you review and triage the board with the kanban tool: "
    "list/read/comment/file are always fine; 'move' is yours for "
    "backlog→doing, doing→backlog, and doing→review only; closing a "
    "review card (review→done) needs evidence= — a fact you verified "
    "with a tool this turn — and is never yours on priority:high, "
    "decide, or prod/security cards: for those, or any other move, get "
    "Tony's explicit yes first or leave him a question with action='ask'."
)

# Same constant pattern as CALENDAR_TOOLS/CMS_TOOLS: lets ADA_EXCLUDED_TOOLS
# strip the devin declarations and their instruction paragraph together.
# tools-merge-devin-mcp (2026-10-05): 8 -> 2. The absorbed names
# (devin_dispatch/devin_status/devin_followup/devin_job_report/
# devin_pending/devin_jobs/devin_answer/ada_devteam_review) live on as
# tool_runner._ALIASES.
DEVIN_TOOLS = {
    "devin", "devin_read",
}

# Tools with real-world side effects — they share a tighter per-turn cap
# (ADA_ACTUATION_BUDGET) than the generic tool-call budget so a runaway
# planning/status turn can't mass-actuate devices or spawn work.
# Phantom-write detector: assistant claims something was
# registered/recorded/noted but made ZERO tool calls in the turn —
# observed failure mode when tool results fail silently upstream.
_PHANTOM_CLAIM_RE = re.compile(
    r"(?i)\b(registered|registration|recorded|queued|written down|noted down"
    # 2026-10-07 confirm-gate flake: Ada narrated "saved" over six
    # NOT-EXECUTED ada_remember denials — the write-claim wordset was
    # missing the exact verb the model reaches for.
    r"|saved|stored|memori[sz]ed|jotted down|kept (?:a )?note|filed)"
    # board write claims (card ada-phantom-card-claims, session
    # 67b02417a8 — two "card filed" narrations, zero file calls):
    # "posted a card", "moved it to doing", "put that on the board".
    # ('opened' is left out — "opened the card" is a read, not a write.)
    r"|\b(?:posted|dropped|added|put|logged|raised|moved|created|made)\b"
    r"[^.\n]{0,40}?\b(?:card|kanban|board)\b"
    r"|\b(?:is|are|now)\s+(?:showing|displayed|playing|up)\s+on\s+(?:the\s+)?screen\b"
    r"|\bon screen (?:now|\d)\b"
    r"|จดไว้|บันทึกไว้|บันทึกแล้ว|จำไว้|เก็บไว้|รับทราบ"
    # board write claims in Thai — the phantom lines were
    # "เปิดการ์ดให้ Devin จัดการ…เรียบร้อยแล้ว อยู่ใน Doing" and
    # "บันทึกไว้บนบอร์ดแล้ว": write-verb + การ์ด/บอร์ด carrying a done
    # marker (bare "อยู่ใน doing" stays out — a read-backed state report
    # is honest).
    r"|(?:เปิด|สร้าง|เพิ่ม|ย้าย|ปิด|ลง)\s*การ์ด[^.\n]{0,60}?(?:แล้ว|เรียบร้อย|เสร็จ)"
    r"|การ์ด[^.\n]{0,30}?(?:เรียบร้อย|เสร็จแล้ว)"
    r"|(?:ลง|ขึ้น)บอร์ด[^.\n]{0,20}?(?:แล้ว|เรียบร้อย|ไว้)"
    r"|บันทึก[^.\n]{0,15}?บอร์ด"
    # display/cast completion claims in Thai — "แสดงผล...แล้ว", "ขึ้นจอ 3 แล้ว"
    r"|(?:แสดงผล|ขึ้น(?:ที่)?จอ|บนจอ|ส่ง(?:ไป)?(?:ที่)?จอ)[^.\n]{0,40}(?:แล้ว|เรียบร้อย)"
    # send/delivery claims (card ada-phantom-claim-gate — kk-dev probes
    # 2026-10-12): chat_send photo:pick was policy-DENIED and she still
    # said it was being sent; another run claimed a catalog delivery
    # over ZERO chat_send calls. Completed/progressive send frames only —
    # promises ("I'll send it", "เดี๋ยวส่งให้") carry no done marker and
    # stay out, as do the user's own sends ("the photo you sent me").
    r"|\b(?:sent|delivered|forwarded|pushed)\s+"
    r"(?:it|that|this|them|those|the|an?|to|over|off|out|you(?:r)?)\b"
    r"|\b(?:it'?s|is)\s+being\s+sent\b"
    r"|\b(?:it'?s|is)\s+on\s+(?:its|the)\s+way\b"
    r"|\bsending\s+(?:it|that|this|them|over|off|out)\b"
    r"|กำลังส่ง"
    r"|ส่ง[^.\n]{0,30}?(?:แล้ว|เรียบร้อย)"
)
# Negated/inability statements aren't claims — "I can't put it on screen"
# fired phantom_write_claim as often as real phantoms (sampled 2026-09-29:
# 41 events in 72h, ~half inability reports). Guard scans the ~48 chars
# before the match for a negation word.
_PHANTOM_NEGATION_RE = re.compile(
    r"(?i)(?:can't|cannot|couldn't|won't|wouldn't|not|isn't|aren't"
    r"|unable|ไม่ได้|ยังไม่|ไม่สามารถ)\W*(?:\w+\W+){0,5}$")

# A done-verb sitting directly on a non-Ada subject isn't Ada's claim —
# "you sent it already" / "เขาส่งแล้ว" reports the OTHER side's action
# and must not arm the re-voice.
_CLAIM_SUBJECT_RE = re.compile(
    r"(?i)(?:you|he|she|they)\s*$|(?:เธอ|คุณ|เขา|หล่อน|พวกเขา|ท่าน)\s*$")


def _phantom_claim(text: str) -> bool:
    m = _PHANTOM_CLAIM_RE.search(text)
    while m:
        if not _PHANTOM_NEGATION_RE.search(text[: m.start()][-48:]):
            if not _CLAIM_SUBJECT_RE.search(text[: m.start()][-24:]):
                return True
        m = _PHANTOM_CLAIM_RE.search(text, m.end())
    return False


# Past-time frames make a write/send-shaped clause a RECALL report, not a
# this-turn claim — "yes, I sent it yesterday" over a zero-tool recall
# answer is honest. The turn-boundary re-voice checks this and stands
# down: forcing a correction onto recall speech would itself narrate a
# falsehood ("nothing was sent"). The phantom_write_claim ops event still
# fires — telemetry tolerates the noise, a spoken correction does not.
_RECALL_CLAIM_RE = re.compile(
    r"(?i)\b(?:earlier|yesterday|last\s+(?:week|night|time|month)|"
    r"previously|before|a\s+while\s+ago|already\s+did)\b"
    r"|เมื่อวาน|ก่อนหน้า|เมื่อก่อน|ไปแล้วตั้งแต่|ตั้งแต่(?:เมื่อ)?วาน")


# Failed result admonition (card ada-phantom-claim-gate): injected into a
# claim-capable call's result blob so the "nothing happened" contract
# travels WITH the result the model composes over — the NOT-EXECUTED
# prefix covers gate denials only, a tool that ran and came back
# ok:false had no such marker (probe 1's photo:pick denial was still
# narrated as 'being sent').
_CLAIM_FAILED_SPEECH_RULE = (
    "This action did NOT happen — the call was denied or failed. Do not "
    "describe it as sent, done, saved, queued, or in progress; say aloud "
    "that it did not go through and give the reason in one clause.")


# What can back a narrated write/done claim: a claim is honest only when
# a call that could produce it returned ok this turn — a passed READ
# must not whitewash it. Card ada-phantom-card-claims (session
# 67b02417a8): two "card filed" narrations with zero write calls went
# unflagged because the old gate counted ANY successful call — a kanban
# list or memory_search — as cover. Tools absent from both maps
# (searches, status/state reads) never back a claim.
_CLAIM_BACKING_ACTIONS = {
    # action-routed tools — only the write seats count. Read actions
    # (kanban list/read, tasks list, cast list/status, yt status/
    # transcript, docs search/get, drive search/get/show, ada_ops
    # usage/health/check/research) fall through to False.
    "kanban": frozenset({"file", "create", "new", "comment", "move",
                         "ask", "respond"}),
    "tasks": frozenset({"add", "done", "move"}),
    "ada_ops": frozenset({"outcome"}),
    "cms_edit": frozenset({"note", "delete", "automate", "edit"}),
    "docs": frozenset({"archive", "print"}),
    "drive": frozenset({"update"}),
    "yt": frozenset({"cast", "stop"}),
    "ada_persona": frozenset({"set", "reset", "set_voice"}),
    "cast_to_screen": frozenset({
        "nav", "play", "image", "audio", "cast", "shortcut", "stop",
        "layout", "zoom", "unzoom", "uplink", "uplink-stop", "say"}),
}
_CLAIM_BACKING_TOOLS = frozenset({
    # Whole-tool writes/actuations — every ok result can back a claim.
    "ada_remember", "ada_forget", "calendar_write", "cms_publish_page",
    "devin", "chat_send", "ada_enroll_speaker", "control_entity",
    "tv_action", "gev_command", "cctv_wall", "vcast_gesture",
    "ada_render_video", "ada_member_keys", "ada_device_acl",
    "speaker_profiles", "ada_track_device", "voice_fx",
    "ada_resolve_action", "report_habit_observation",
})


def _claim_backing_call(name: str, args: dict[str, Any]) -> bool:
    """True when an ok result from this call could honestly back a
    narrated write/done claim — mutating tools and the mutating seats of
    action-routed tools."""
    if name in _CLAIM_BACKING_TOOLS:
        return True
    acts = _CLAIM_BACKING_ACTIONS.get(name)
    if acts is not None:
        return str((args or {}).get("action") or "").strip().lower() in acts
    if name == "ada_camera_snapshot":
        # A frame pushed to a display is an actuation; a bare describe
        # is a read (same split as the actuation budget).
        args = args or {}
        return bool(args.get("screen")) or str(
            args.get("target") or "").strip().lower() in ("tv", "screen")
    return False

# Media narration guard (card ada-tv-action-hallucinated-input,
# 2026-10-09): Ada answered "what's playing" with 'Dancing In The
# Street' while the turn's only evidence was cast_verify=paused — the
# title existed in no tool result. A named title or a playing claim is
# honest only when that turn's tool results literally carry it.
_MEDIA_PLAY_RE = re.compile(
    r"(?i)\bplaying\b|กำลังเล่น|เล่นอยู่|กำลังฉาย|ฉายอยู่")
_MEDIA_TITLE_RES = (
    # 'Title' / "Title" quoted spans (only consulted when a play word is
    # present — checked by the caller). The lookbehind keeps mid-word
    # apostrophes ("It's") from opening a fake span.
    re.compile(r"(?<![\w’'])[\"'‘’“”«»]([^\"'‘’“”«»\n]{2,60})[\"'‘’“”«»]"),
    # "now playing <Title>"
    re.compile(r"(?i)\bnow\s+playing[\s:–—-]*"
               r"([A-Za-z0-9][A-Za-z0-9'’&.,\- ]{1,58})"),
    # "<Title Case words> is/was (now|still) playing"
    re.compile(r"\b([A-Z][\w'’&.,-]*(?:\s+[A-Za-z0-9'’&.,-]+){0,6})"
               r"\s+(?:is|was)\s+(?:now\s+|still\s+|currently\s+)?playing\b"),
    # Thai: เพลง/เพลิดเพลิน/วิดีโอ/คลิป followed by a latin-script title
    # ("กำลังเล่นเพลง Dancing In The Street อยู่ค่ะ").
    re.compile(r"(?:เพลง|เพลิดเพลิน|วิดีโอ|คลิป)\s*[\"'«»]?"
               r"([A-Za-z0-9][A-Za-z0-9'’&.,\- ]{1,58}[A-Za-z0-9])"),
)
# Pronouns/generic words the Title-Case extractor can pick up — never
# treated as named titles.
_MEDIA_TITLE_STOP = frozenset({
    "it", "this", "that", "the", "they", "he", "she", "we", "i", "you",
    "something", "nothing", "anything", "everything", "ada", "tv",
    "กำลัง", "เพลง",
})
# Filler words trimmed off capture edges — "now playing on the TV" must
# not read 'on the TV' as a title.
_MEDIA_TITLE_EDGE_RE = re.compile(
    r"(?i)^(?:on|the|a|an|in|at|to|of|for|by|and|is|was|now|still|"
    r"currently|right|your|my)\s+|\s+(?:on|the|a|an|in|at|to|of|for|"
    r"by|and|is|was|now|still|currently|right|your|my)$")
# Instructive result fields are our own admonitions ("do not claim it is
# playing") — not evidence a title/state was reported.
_MEDIA_EVIDENCE_SKIP_RE = re.compile(
    r"(?i)error|warn|note|usage|reminder|hint|instruct")


def _media_evidence_blob(results: list[tuple[str, Any]]) -> str:
    """Lowercased JSON of the turn's tool results minus instructive
    fields — the corpus a media claim must be found in."""
    def scrub(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items()
                    if not _MEDIA_EVIDENCE_SKIP_RE.search(str(k))}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v
    return json.dumps([scrub(r) for _, r in results],
                      default=str).lower()


def _media_grounding_problems(
        text: str, results: list[tuple[str, Any]]) -> list[str]:
    """Ungrounded media narration in one assistant turn: playing-state
    assertions and named titles no tool result carries. Empty = grounded
    or no media claim. Negated statements ("it's not playing") are not
    claims."""
    if not _MEDIA_PLAY_RE.search(text):
        return []
    blob = _media_evidence_blob(results)
    problems: list[str] = []
    for m in _MEDIA_PLAY_RE.finditer(text):
        if _PHANTOM_NEGATION_RE.search(text[: m.start()][-48:]):
            continue
        if "playing" not in blob:
            problems.append("asserted 'playing'")
        break  # one verdict covers all play-state mentions in the turn
    for raw in _MEDIA_TITLE_RES:
        for match in raw.finditer(text):
            title = match.group(1).strip(" .,!?'\"'’“”«»")
            while True:
                trimmed = _MEDIA_TITLE_EDGE_RE.sub("", title).strip(
                    " .,!?'\"'’“”«»")
                if trimmed == title:
                    break
                title = trimmed
            low = title.lower()
            if len(low) < 2 or low in _MEDIA_TITLE_STOP:
                continue
            if low not in blob:
                problems.append(f"named title {title!r}")
    return problems


# Write-fidelity guard (card ada-cms-write-fidelity, MEM-0498 finding 2):
# the text Ada narrates and the content she persists are separate
# generations that can diverge — the kk-cms-only bench caught her storing
# 'Tuesdays' while saying 'Mondays' (2026-10-12), and the write landed.
# Fact tokens (weekdays/clock times/dates) are extracted from BOTH the
# narration and the persisted content arg of each publish call that
# returned status=published this turn; only a CONTRADICTION flags —
# both sets non-empty and disjoint — so a vague narration ("saved the
# page") or a matching one stays silent.
_FIDELITY_WRITE_TOOLS = frozenset({"cms_publish_page"})

_FIDELITY_DAY_RES = (
    (re.compile(r"(?i)\bmondays?\b"), "mon"),
    (re.compile(r"(?i)\btuesdays?\b"), "tue"),
    (re.compile(r"(?i)\bwednesdays?\b"), "wed"),
    (re.compile(r"(?i)\bthursdays?\b"), "thu"),
    (re.compile(r"(?i)\bfridays?\b"), "fri"),
    (re.compile(r"(?i)\bsaturdays?\b"), "sat"),
    (re.compile(r"(?i)\bsundays?\b"), "sun"),
)
# Thai weekday names — พฤหัส matches inside พฤหัสบดี; both map to thu.
_FIDELITY_DAY_TH = (
    ("จันทร์", "mon"), ("อังคาร", "tue"), ("พุธ", "wed"),
    ("พฤหัส", "thu"), ("ศุกร์", "fri"), ("เสาร์", "sat"),
    ("อาทิตย์", "sun"),
)
# Clock times — '6am', '6:30pm', '18:00' normalize to 'HH:MM' 24h; Thai
# '<n> โมงเช้า/บ่าย/เย็น' and '<n> น.' land in the same space so an en
# narration and th content still compare.
_FIDELITY_TIME_EN_RE = re.compile(
    r"(?i)\b([01]?\d|2[0-3])(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b"
    r"|\b([01]?\d|2[0-3]):([0-5]\d)\b")
_FIDELITY_TIME_TH_RE = re.compile(
    r"([01]?\d|2[0-3])\s*(?:โมง)\s*(เช้า|บ่าย|เย็น|ค่ำ|ดึก)?"
    r"|([01]?\d|2[0-3])\s*น\.")
_FIDELITY_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b")


def _fidelity_tokens(text: str) -> dict[str, set]:
    """Fact-token sets a persisted text or a narration carries:
    {day: {mon..sun}, time: {HH:MM}, date: {literal}}."""
    t = str(text or "")
    days: set[str] = set()
    for rx, canon in _FIDELITY_DAY_RES:
        if rx.search(t):
            days.add(canon)
    for th, canon in _FIDELITY_DAY_TH:
        if th in t:
            days.add(canon)
    times: set[str] = set()
    for m in _FIDELITY_TIME_EN_RE.finditer(t):
        if m.group(3):  # am/pm form
            hour = int(m.group(1)) % 12
            if m.group(3).lower().startswith("p"):
                hour += 12
            times.add(f"{hour:02d}:{int(m.group(2) or 0):02d}")
        else:
            times.add(f"{int(m.group(4)):02d}:{m.group(5)}")
    for m in _FIDELITY_TIME_TH_RE.finditer(t):
        if m.group(1):  # '<n> โมง<period>'
            hour = int(m.group(1))
            period = m.group(2)
            if period in ("บ่าย", "เย็น", "ค่ำ", "ดึก") and hour < 12:
                hour += 12
            times.add(f"{hour:02d}:00")
        else:
            times.add(f"{int(m.group(3)):02d}:00")
    dates = set(_FIDELITY_DATE_RE.findall(t))
    return {"day": days, "time": times, "date": dates}


def _write_fidelity_problems(
        narration: str,
        user_text: str,
        calls: list[tuple[str, dict, dict]]) -> list[str]:
    """Narrated/seeded facts contradicted by persisted publish content.

    calls = [(name, args, result)] for the turn's fidelity-watched writes.
    Returns advisory strings; empty = consistent or nothing to compare.
    A contradiction needs BOTH sides to carry the same fact class with no
    common value — absence on either side is not a flag."""
    narr = _fidelity_tokens(narration)
    seed = _fidelity_tokens(user_text)
    if not any(narr.values()) and not any(seed.values()):
        return []
    problems: list[str] = []
    for name, args, result in calls:
        if name not in _FIDELITY_WRITE_TOOLS:
            continue
        if not isinstance(result, dict) or result.get("ok") is False:
            continue
        if str(result.get("status") or "") != "published":
            continue
        content = str((args or {}).get("content") or "")
        if not content.strip():
            continue
        stored = _fidelity_tokens(content)
        slug = str((args or {}).get("slug") or "?")
        for label, side in (("narration", narr), ("user", seed)):
            for cat in ("day", "time", "date"):
                if (side[cat] and stored[cat]
                        and side[cat].isdisjoint(stored[cat])):
                    problems.append(
                        f"{name} slug={slug}: stored {cat} "
                        f"{sorted(stored[cat])} contradicts {label} "
                        f"{sorted(side[cat])}")
    return problems


# Dead-turn detector (card ada-dead-turn-guard — session 56d2e4d167,
# 2026-10-07): the model twice ended a turn having emitted only the
# literal scaffold "response:" while the memory tool already held the
# answer — the user heard nothing and had to repeat the question.
# Degenerate = whitespace-only, a bare scaffold prefix, or lone
# structural tags/punctuation. A scaffold prefix followed by real words
# ("response: พบแล้วค่ะ…") is NOT dead — the lead strip only fires when
# nothing speakable remains.
_DEAD_SCAFFOLD_WORDS = (
    r"response|answer|reply|output|result|assistant|transcript|text"
    r"|คำตอบ|ตอบ|ผลลัพธ์"
)
_DEAD_SCAFFOLD_RE = re.compile(
    rf"^\s*(?:{_DEAD_SCAFFOLD_WORDS})\s*[:：]?\s*$", re.IGNORECASE)
_DEAD_LEAD_RE = re.compile(
    rf"^\s*(?:{_DEAD_SCAFFOLD_WORDS})\s*[:：]", re.IGNORECASE)
_DEAD_TAG_RE = re.compile(r"<[^>\n]{0,80}>")
_DEAD_JUNK_RE = re.compile(r"[^\wก-๙]+", re.UNICODE)


def _dead_turn_text(text: str) -> bool:
    """True when an assistant turn's accumulated text carries nothing
    speakable — empty, a bare 'response:' scaffold, or only structural
    tags/punctuation."""
    t = (text or "").strip()
    if not t or _DEAD_SCAFFOLD_RE.match(t):
        return True
    t = _DEAD_LEAD_RE.sub("", t, count=1)
    t = _DEAD_TAG_RE.sub(" ", t)
    return not _DEAD_JUNK_RE.sub("", t)


def _first_speakable_line(text: Any, limit: int = 160) -> str:
    """First non-empty line of `text`, de-marked-up and capped — the piece
    of a tool result a dead-turn fallback can say out loud."""
    for line in str(text or "").splitlines():
        line = line.strip().lstrip("#*>-• ").strip()
        if line:
            return line[:limit].rstrip()
    return ""


# Tool-call-shaped text the leak filter must catch even when the name is
# NOT one of the declared tools (hallucinated names, aliases that slipped
# the declaration set): `identifier{key:value`, `identifier{key=value`,
# `identifier {"key": …`. The braces never occur in spoken Thai/English
# transcription, so a match is always a leak.
_GENERIC_TOOL_LEAK_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]{1,39}\s*\{\s*[\"']?"
    r"[A-Za-z_][A-Za-z0-9_]{0,39}\s*[:=]")

# Role-marker leak (card ada-context-budget — session d6cbf6e6e7,
# 2026-10-07): when the context window saturates, the model degenerates
# into raw transcript continuation and emits the scaffold's role tags —
# 'ยืนยัน—user\nบันทึกเลยmodel' was spoken as one assistant turn, meaning
# the model fabricated the user's reply inside its own output. The tag
# must sit at a boundary AND terminate a chunk (em-dash/newline before,
# newline-or-end after): mid-sentence 'model'/'user' stays speakable.
_ROLE_LEAK_RE = re.compile(
    r"(?:^|[—–\n\r])\s*(?:user|model|system|assistant)\s*(?=\n|$)",
    re.IGNORECASE)


ACTUATING_TOOLS = frozenset({
    # tools-merge-ha (2026-10-05): control_cover/control_media_player/
    # press_button collapsed into control_entity — the canonical name
    # carries every actuating action now.
    # tools-merge-display: vcast_say folded into cast_to_screen
    # action='say'; the canonical name holds the seat.
    # tools-merge-docs-drive: ada_doc_archive/ada_doc_print collapsed into
    # docs — actuation is keyed per-action at the call site below.
    "control_entity", "tv_action",
    # tools-merge-yt: yt_cast/yt_cast_stop collapsed into yt
    # (action=cast|stop) — the canonical name holds the seat,
    # actuation keyed per-action at the call site below.
    "yt", "cast_to_screen",
    # tools-merge-meta-voice: ada_set_voice folded into
    # ada_persona(*_voice actions) — set_voice gated per-action
    # at the call site below.
    "devin_dispatch",
    "calendar_write",
    # tools-merge-tasks-status: the merged tasks tool holds the absorbed
    # writers' seat — a stray action='list' call draws on the same budget,
    # same name-keyed convention as the other merged actuators.
    "tasks",
})
# tools-merge-meta-voice: ada_set_voice's actuating seat (it drops the
# live session for a reconnect) moved to ada_persona action='set_voice' —
# counted per-call at the budget check since persona's other actions are
# reads.

# 'confirmed=true' in a tool arg is only honored when the user's own
# speech affirms — otherwise the model could self-certify past the
# dangerous-device and write-policy gates (observed during the
# 2026-09-25 tool storm: plug_tv switched on with zero user consent).
_CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|confirm(ed)?|go ahead|do it|sure|okay?|"
    r"approved?|proceed|absolutely|mhm|uh huh|sounds good)\b|"
    # STT often splits Thai syllables with spaces ("เริ่ม เลย") — \s* keeps
    # compounds matching.
    r"ใช่|ยืน\s*ยัน|ตก\s*ลง|เอา\s*เลย|ทำ\s*เลย|ได้\s*เลย|ทำ\s*ได้|โอเค|ออเค|เออ|อือ|"
    r"เริ่ม\s*เลย|จัดการ\s*เลย|ลอง\s*เลย|ต่อไป|จัด\s*ไป|เอา\s*สิ|ไป\s*เลย|"
    r"ทำ\s*ไป|เผยแพร่\s*เลย|ส่ง\s*เลย",
    re.IGNORECASE,
)

# Vetoes checked before the affirm regex — the gate is asymmetric: a
# false "yes" actuates on no consent, a false "no" just re-asks.
# (jev-corpus divergences 2026-10-04: 'เอ๊ะ! No. …' matched อือ,
# '…อนุมัติได้เลยไหม' matched ได้เลย, 'Confirm the camera moved'
# matched confirm-as-verb.)
_CONFIRM_NEG_RE = re.compile(
    r"\b(no|nope|nah|not|don't|dont|do not|cancel|wait|hold on|stop)\b|"
    # ไม่ต้อง ("no need to …") is granting, not denying — excluded.
    r"อย่า|หยุด|ยกเลิก|ไม่(?!ต้อง)",
    re.IGNORECASE,
)
_CONFIRM_QUESTION_RE = re.compile(r"[?？]\s*$|ไหม\s*$|มั้ย\s*$")
# "confirm" as a verb with an object ("Confirm the camera moved —") is an
# instruction to Ada, not an affirmation of a pending ask.
_CONFIRM_VERB_RE = re.compile(
    r"\bconfirm(ed)?\b(?=\s+(the|a|an|that|what|it|your|my|this)\b|\s*[—–-])",
    re.IGNORECASE,
)

DEVIN_INSTRUCTIONS = (
    " You can dispatch unattended Devin coding sessions on tony-dell: "
    "devin action='dispatch' starts one in a dedicated git worktree (repos: "
    "chaba, ada-voice, sunsynk-card), devin_read action='status' lists running "
    "and finished tasks, and devin action='followup' sends a message into a "
    "running session. "
    "Before dispatching, restate the repo and task and get an explicit yes, then "
    "call with confirmed=true — writes are enforced server-side. "
    "Dispatched sessions run unattended; the user is notified on their phone when "
    "one finishes, so report the task id and move on rather than polling. "
    "devin_read action='jobs' is the dispatch ledger — call it for any "
    "'summarize my dispatched tasks' or 'did job X fail' question and report "
    "statuses exactly as stored (done/failed/running/awaiting-user); never guess "
    "a job's outcome. "
    "devin_read action='pending' lists jobs blocked waiting for the user's "
    "answer — when the user asks what needs their attention, or says a job is "
    "waiting, call it and read each job's question back with its short detail. "
    "To deliver an answer: refine the user's reply into a self-contained "
    "instruction (the job sees only the text, not this conversation), read the "
    "refined text back, get an explicit yes, then call devin action='answer' "
    "with confirmed=true. "
    "When discussing an implementation task the user wants built later, offer to "
    "save the spec into the devin-handoff memory bank so a dispatched session can "
    "be told to 'check the ada handoff' — devin_read action='review' runs the "
    "dev-team expert panel on a proposed tool and files the spec there too. "
    "When the user asks about Devin job status or wants the devin-job-report "
    "page refreshed, call devin_read action='report' — it puts failed jobs "
    "(including spawn failures the ledger still marks running) in their own "
    "'Failed jobs' section; never list failed jobs among the active ones."
)

OPENCODE_INSTRUCTIONS = (
    " OpenCode is a bounded executor behind Ada, not the user-facing "
    "responder. For implementation work, call ada_opencode action='preflight' "
    "on the exact card before narrating any envelope. NEVER invent default "
    "limits or claim that preflight ran: the tool result is the only source "
    "of truth. Pass the exact bounded task to preflight so its approval "
    "receipt binds the task. Require the returned explicit model, LiteLLM "
    "key alias, profile, target host, artifact, resolved verifier, and max "
    "USD/turn/token/wall-time/retry limits. Show those returned values to "
    "the user. Only after the user approves that exact receipt may you call "
    "ada_opencode_dispatch, passing the task, all returned limits, "
    "approval_receipt, approval_issued_at, and approval_expires_at verbatim. "
    "Do not retry a schema-error dispatch with guessed or default arguments; "
    "report the rejected fields and stop. Never silently change the executor, "
    "model, key, host, profile, verifier, artifact, or limits, and never "
    "claim a job ran without a board/artifact receipt. Use ada_opencode "
    "action='status' or 'receipts' for follow-up. Plain questions and "
    "status/spend questions stay in Ada and must not start OpenCode.\n"
)

# Summary rollup tools — tools-merge-calendar-plan (2026-10-05) folded
# ada_daily_summary/ada_weekly_comparison into plan_day (period='week' /
# day=<date>); their guidance now lives in CALENDAR_INSTRUCTIONS so
# ADA_EXCLUDED_TOOLS strips it with the rest of the calendar surface.

# Habit tracking tools — same constant pattern: ADA_EXCLUDED_TOOLS strips
# the declarations and this instruction paragraph together. Only meaningful
# where the hardware backend (camera + pose pipeline, backend/main.py)
# supplies a habit_state_getter; pwa-only instances should exclude both.
# tools-merge-tasks-status: get_habit_status folded into home_status
# (what='habit') — the canonical name takes its seat, so excluding habit
# tools on a pwa-only instance now also drops the other status reads.
HABIT_TOOLS = {"home_status", "ada_remember"}

HABIT_INSTRUCTIONS = (
    " You monitor habits such as seated posture: local pose estimation proposes "
    "events, Gemini vision verifies ambiguous ones, and confirmed occurrences "
    "can become possible, emerging, or established habits over time. You also "
    "track home plugs left on while the user is away or the home has remained "
    "empty — Home Assistant supplies person and plug states; local vision "
    "supplies home presence. Habit alerts may arrive with a current image and "
    "structured context: give a brief, dry observation and one practical "
    "correction, distinguishing a first possible habit from an established one. "
    "When the user asks what habits are tracked, their status, or progress, "
    "always call home_status what='habit' and ground the answer in its "
    "current result. "
    "For habits, make it clear that you noticed the pattern, then give one "
    "useful, realistic correction — mild judgment, never mockery or repetitive "
    "roasting. Never claim a habit occurred unless the application reports a "
    "confirmed event."
)

# Document archive tools — same constant pattern: ADA_EXCLUDED_TOOLS strips
# the declarations and this instruction paragraph together.
# tools-merge-docs-drive (2026-10-05): the four ada_doc_* names collapsed
# into docs(action=search|get|print|archive); old names live on as
# tool_runner._ALIASES rows, not declarations.
DOC_TOOLS = {"docs"}

DOC_INSTRUCTIONS = (
    " You have a personal document archive (scans of deeds, IDs, passports, "
    "receipts — เอกสาร). Questions about documents, scans, or archived papers — "
    "including Thai words like เอกสาร/สำเนา/โฉนด — go to docs action='search' "
    "to find the archive slug, then action='get' for details; NEVER search "
    "home devices for documents. action='archive' saves a newly uploaded "
    "document set into the archive, and action='print' prints archived pages "
    "on the DeskJet — both need confirmed=true after restating what will be "
    "archived or printed. When a document was just uploaded (the system note "
    "carries an intake key), propose a slug from the filename and "
    "assessment, and confirm the slug and action before archiving — intake "
    "keys are held in RAM only, so archive promptly rather than deferring. "
    "If docs action='archive' reports duplicates or near-duplicates, say so "
    "plainly and ask whether it's a re-scan or a new version before "
    "proceeding."
)


# Google Drive / Photos tools — excluded together with the declarations.
# tools-merge-docs-drive (2026-10-05): the four drive_* names collapsed
# into drive(action=search|show|get|update); old names live on as
# tool_runner._ALIASES rows, not declarations.
# tools-merge-tasks-status: photos_pick/photos_picked folded into
# chat_send (photo='pick'/'picked') — no seat here; excluding Drive
# leaves chat_send's plain sends working while photo=/doc= flows fail
# honestly.
DRIVE_TOOLS = {
    "drive",
}

DRIVE_INSTRUCTIONS = (
    " You can reach the operator's Google Drive: drive action='search' "
    "finds files by name or content (narrow with mime like 'image/' or "
    "'video/'), action='get' reads a file (text comes back inline, binary "
    "gets a media_url), action='show' puts a Drive photo, video, or file "
    "on the user's screen or the TV, and action='update' replaces a text "
    "file's content — that one needs confirmed=true after restating the "
    "file and change, and it cannot edit Google-native Docs/Sheets/Slides. "
    "Photos in Google Photos are NOT browsable — Google limited the "
    "library API to app-created media, so for 'show my photos' use "
    "chat_send photo='pick': it returns a picker_uri (and texts it to the "
    "channel) the user opens on their signed-in phone or laptop to "
    "select items, then chat_send photo='picked' with the session_id "
    "returns and can cast what they chose. If the user means photos saved "
    "in a Drive folder instead, drive action='search' with mime='image/' "
    "is the direct path — prefer that when it fits."
)


# XMEye VMS camera snapshots — the VMS_INSTRUCTIONS paragraph is gated by
# ADA_VMS_SNAP_URL (not via ADA_EXCLUDED_TOOLS): property-camera guidance
# only exists when the instance can reach the vms-snap shim on the VMS
# host. The declaration itself is now unconditional — source='traffic'
# keeps public traffic cams working on instances without the shim.
VMS_TOOLS = {"ada_camera_snapshot"}

# tools-merge-camera: ada_camera_snapshot is the single camera stills
# tool — source='vms' property CCTV (+ go2rtc home cams), 'traffic'
# public Thailand cams, screen=/target= to push the frame to a display.
# It absorbed cctv_snapshot and traffic_camera; those names survive as
# tool_runner._ALIASES rows, not declarations.
VMS_INSTRUCTIONS = (
    " You can pull a still frame from the property CCTV cameras with "
    "ada_camera_snapshot — pass the view the user means (e.g. 'swimming pool', "
    "'tennis court', 'front road', 'walkway', 'guard', 'mini mart'). The frame "
    "arrives attached to the tool result: look at it and describe what it "
    "shows honestly — people, water level, weather, vehicles, anything "
    "notable. It is a single still taken a few seconds ago, NOT live video; "
    "say what you see now and do not claim continuous monitoring. If the tool "
    "fails or the frame is dark or frozen, say the camera seems offline "
    "rather than guessing. "
    "STORED-FIRST: when the user asks to see a property camera, call "
    "ada_camera_snapshot with mode='cached' FIRST — it answers instantly "
    "with the wall's stored frame and its age. Show and describe it "
    "honestly as the stored picture ('as of <age> ago'), announce that you "
    "are refreshing it live, then call ada_camera_snapshot again with "
    "mode='live' (the default) and report the fresh frame — or honestly "
    "say the refresh failed and the stored frame is still the newest. "
    "Skip the cached step only when the user explicitly wants a live "
    "look right now. "
    "DESCRIBE-FIRST: when the user asks what cameras show ('เห็นอะไรบ้าง', "
    "'what do you see', 'check zone X'), pull frames yourself with "
    "ada_camera_snapshot and describe them — do NOT offer to cast or uplink "
    "to a screen unless the user asks to see it on a display (they may not "
    "be near one). For a zone summary, snap ONE representative channel "
    "(e.g. 'road in' for zone-a — VMS frames take up to a minute on cold "
    "streams, so don't chain several) and offer to check more if they ask. "
    "SELF-HEAL on misses: if the tool returns the available channel list, "
    "retry immediately with the closest listed name — do not ask the user "
    "to pick."
)

def _as_num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


TRAFFIC_INSTRUCTIONS = (
    " For public traffic cameras call ada_camera_snapshot with "
    "source='traffic' — it searches ~190 Thailand traffic cams "
    "(expressways, Bangkok, Chonburi) by area keyword or user "
    "position+heading, snaps the current frame, and attaches it for you "
    "to describe. Prefer source='traffic' over the default source "
    "whenever the user asks about roads/traffic outside the property."
)

CAMERA_DECLARATION = {
    "name": "ada_camera_snapshot",
    "description": (
        "Take a still camera snapshot — property CCTV (source='vms'), home cams, "
        "or Thailand traffic cams (source='traffic'; 'auto' picks by args). "
        "The frame attaches to the result for you to describe; screen=/target= "
        "also shows it on a display. Absorbs cctv_snapshot and traffic_camera."
    ),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "description": (
                    "What to look at: a property camera name ('swimming "
                    "pool', 'tennis court', 'front road left', 'walkway', "
                    "'guard view', 'mini mart', 'coffee corner', 'c201', "
                    "'c100') or — with source='traffic' — an area/road "
                    "keyword ('bangna', 'burapha', 'สุขุมวิท'). Fuzzy — "
                    "the service returns the available list on a miss."),
            },
            "source": {
                "type": "string",
                "enum": ["auto", "vms", "traffic"],
                "description": (
                    "'auto' (default): a property cam unless traffic-style "
                    "args arrive. 'vms': property CCTV / home cams. "
                    "'traffic': public Thailand traffic cams "
                    "(Longdo/iTIC feed)."),
            },
            "channel": {
                "type": "string",
                "description": "Legacy alias for view.",
            },
            "camera": {
                "type": "string",
                "description": "Legacy alias for view.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Traffic area/road keyword — same role as view when "
                    "source='traffic'."),
            },
            "lat": {"type": "number",
                    "description": "User latitude (with lon) — nearest cams win."},
            "lon": {"type": "number",
                    "description": "User longitude (with lat)."},
            "heading": {"type": "number",
                        "description": "User heading in degrees (0=N) — prefers cams ahead."},
            "mode": {
                "type": "string",
                "enum": ["live", "cached"],
                "description": (
                    "'live' (default) pulls a fresh frame (~20-40s); "
                    "'cached' returns the last wall-stored frame instantly "
                    "with its age — the stored-first answer before a live "
                    "refresh."),
            },
            "screen": {
                "type": "integer",
                "description": (
                    "Also show the captured frame on this vcast display "
                    "number."),
            },
            "target": {
                "type": "string",
                "enum": ["tv", "screen"],
                "description": (
                    "Where to show the frame: 'tv' for the living-room TV, "
                    "'screen' for the vcast display in 'screen' (default "
                    "1). Omit both for describe-only."),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "Required when pushing the frame to a display after "
                    "the user explicitly said yes."),
            },
        },
        "additionalProperties": False,
    },
}


# CHABA_MEMORY=1 guest mode: the instance serves visitors through the file-
# backed chaba store instead of MDDB banks. Guest tools are appended and the
# tool surface is cut to CHABA_ALLOW — an allowlist so new Ada tools never
# leak into guest sessions by accident.
# Guest memory rides the canonical tools (tools-merge-memory, 2026-10-04):
# ada_remember(kind='guest'[, private=true]) absorbs guest_remember /
# guest_remember_private, ada_memory_search(scope='guest') absorbs
# guest_recall. CHABA_DECLARATIONS re-declares them with guest-scoped
# schemas; the allowlist below swaps out the bank-facing versions.
# tools-merge-meta-voice: guest_register folded into ada_enroll_speaker
# who='guest' — the canonical name sits in the allowlist, the absorbed
# name stays callable as a tool_runner._ALIASES row.
CHABA_TOOLS = {"ada_remember", "ada_memory_search", "ada_enroll_speaker"}

# tools-merge-ha (2026-10-05): the finders/history reads consolidated —
# guests get the canonical trio (home_search kind=device|sensor,
# get_home_state, home_history) and control_entity for lights/media.
CHABA_ALLOW = CHABA_TOOLS | {
    "get_home_state", "home_search", "home_history",
    # tools-merge-tasks-status: the status getters merged into
    # home_status — the canonical name keeps the guest seat.
    "home_status",
    "get_rk600_weather", "get_pool_status",
    "get_battery_status", "get_battery_detail",
    "control_entity",
}

CHABA_INSTRUCTIONS = (
    " You are Chaba — the house AI — speaking in guest mode for visitors. On first contact ask "
    "the visitor's name and call ada_enroll_speaker with who='guest' and "
    "their name — registration lets an "
    "admin promote them to a named user later. Before the first ada_remember "
    "in a session, say clearly that saved memories are visible to everyone in "
    "this household. ada_remember(kind='guest') saves a PUBLIC note (key + "
    "text); ada_memory_search(scope='guest') searches saved notes; "
    "private=true is only for promoted users and fails for guests. You can "
    "read home state and sensors "
    "and control lights/media via control_entity (action=on/off or the "
    "media_player verbs), but never anything that moves — no cover, gate, "
    "or button actions; refuse those politely."
)

CHABA_DECLARATIONS = [
    {
        "name": "ada_remember",
        "description": (
            "Save a public guest memory under the visitor's name — always "
            "kind='guest'; disclose first that guest memories are visible to "
            "the household."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["guest"]},
                "key": {"type": "string", "description": "Short slug, e.g. 'favorite-drink'."},
                "text": {"type": "string", "description": "The fact or note to remember."},
                "private": {
                    "type": "boolean",
                    "description": "Promoted users only — saves to their private namespace.",
                },
            },
            "required": ["kind", "key", "text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ada_memory_search",
        "description": "Search public guest memories by keyword — always scope='guest'.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["guest"]},
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Max hits (default 10)."},
            },
            "required": ["scope", "query"],
            "additionalProperties": False,
        },
    },
    {
        # tools-merge-meta-voice: the guest-scoped seat of
        # ada_enroll_speaker (absorbed guest_register) — who is pinned to
        # 'guest' so the visitor surface can't reach voice enrollment.
        "name": "ada_enroll_speaker",
        "description": (
            "Register the visitor's name for later admin promotion — always "
            "who='guest'; ask their name first."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "who": {"type": "string", "enum": ["guest"]},
                "name": {"type": "string", "description": "Visitor's declared name."},
            },
            "required": ["who", "name"],
            "additionalProperties": False,
        },
    },
]


def _fill_bank_placeholders(node: Any, banks: str, writable: str) -> Any:
    """Recursively substitute {banks}/{writable_banks} in tool schemas."""
    if isinstance(node, str):
        return node.replace("{banks}", banks).replace("{writable_banks}", writable)
    if isinstance(node, dict):
        return {k: _fill_bank_placeholders(v, banks, writable) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill_bank_placeholders(v, banks, writable) for v in node]
    return node


def _builtin_declarations() -> list[dict[str, Any]]:
    """The builtin tool surface, built fresh per call — connect() filters
    (text_only, ADA_EXCLUDED/INCLUDED_TOOLS, chaba allowlist) and
    _fill_bank_placeholders rewrite the dicts per session, so the
    declarations must never be shared between sessions."""
    return [{
            "name": "set_facial_expression",
            "description": (
                "Changes Ada's visible facial expression without interrupting speech. "
                "Select the expression matching the tone of Ada's current response."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "enum": list(EXPRESSION_NAMES),
                        "description": "The facial expression Ada should display.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        }, {
            "name": "get_home_state",
            "description": (
                "Read-only home state — plugs, occupancy, habit conditions "
                "(absorbs ada_ha_get_state). entity_id= reads one entity live; "
                "domain= lists a domain; domain='memory' returns the stored snapshot."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Optional exact entity_id to read live, e.g. binary_sensor.front_door.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional HA domain to list (e.g. 'light', 'sensor'), or 'memory' for the stored snapshot.",
                    },
                },
                "additionalProperties": False,
            },
        }, {
            "name": "control_entity",
            "description": (
                "Actuate one Home Assistant entity (absorbs control_cover, "
                "press_button, control_media_player) — on=/action= picked by the "
                "entity's domain. Only media_player.tony_tv/tony_tv_cast are ours."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The Home Assistant entity_id to control, e.g. light.living_room or cover.gate_motor.",
                    },
                    "on": {
                        "type": "boolean",
                        "description": "For light/switch/fan/input_boolean entities: true to turn on, false to turn off.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "open", "close", "stop", "press",
                                 "turn_on", "turn_off", "media_play", "media_pause",
                                 "media_stop", "volume_up", "volume_down",
                                 "volume_mute", "mute", "select_source"],
                        "description": "Domain action: on|off for simple entities; open|close|stop for cover.*; press for button.*; media_player verbs for media_player.* (source= required for select_source).",
                    },
                    "source": {
                        "type": "string",
                        "description": "media_player select_source only: the input/source name to select.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required for dangerous-safety entities; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["entity_id"],
                "additionalProperties": False,
            },
        }, {
            "name": "home_search",
            "description": (
                "Find HA devices, sensors, or recorded events — kind='device'|"
                "'sensor'|'event' (absorbs the list/search/ada_ha_search_* tools); "
                "blank query lists the kind. Use it to get the exact entity_id "
                "before control_entity."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Name or keyword to search, e.g. 'kitchen table', 'front gate', 'pool temperature', 'door'. Blank lists the kind instead.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["device", "sensor", "event"],
                        "description": "What to find: a controllable device (default), a sensor entity, or recorded events.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum results. Default 10.",
                    },
                    "hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 720,
                        "description": "kind='event' only: how many hours of recorded events to search. Default 24.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }, {
            "name": "yt",
            "description": (
                "YouTube/media on the living-room TV — action='cast' PLAYS a video "
                "(.mp4/.m3u8 URL or search phrase) only on an explicit play request — "
                "'find a clip' is a lookup (web_search), never a cast. 'status'|'stop'|"
                "'transcript'. Dark screen -> tv_action; vcast screen -> cast_to_screen."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["cast", "status", "stop", "transcript"],
                        "description": "Operation: cast a video, check cast progress, stop casting, or fetch a transcript.",
                    },
                    "query": {
                        "type": "string",
                        "description": "cast: YouTube URL, media file URL, or search phrase, e.g. 'the egg kurzgesagt'.",
                    },
                    "url": {
                        "type": "string",
                        "description": "transcript: YouTube URL or video ID.",
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "cast: target subtitle language shown below the original-language "
                            "line; transcript: caption language to prefer (default 'th' for Thai, "
                            "falls back to en)."
                        ),
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            "name": "tv_action",
            "description": (
                "Send a command to the LG TV's cast-browser via HA "
                "rest_command.tv_action — cmd=nav|scroll|click|type|press|back|shot|"
                "viewport|power, payload in text=. vcast display panes use "
                "cast_to_screen instead; never for answering questions."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Command key: nav|scroll|click|type|press|back|shot|viewport|power.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Command text/payload: nav target (URL, 'gev' for live God's Eye View, 'screenlive:workspace:N[:pad|crop]', 'tony-omen:workspace:N'), scroll direction, visible text to click, text to type (cmd=type takes ONLY the exact words the user just dictated — empty or placeholder text like 'your-*', '<...>', 'xxx' is refused), or shot/viewport args.",
                    },
                    "selector": {"type": "string", "description": "CSS selector for click."},
                    "role": {"type": "string", "description": "ARIA role for click (e.g. 'button')."},
                    "key": {"type": "string", "description": "Key name for press (e.g. 'Enter', 'Escape')."},
                    "dx": {"type": "number", "description": "Horizontal scroll amount (px)."},
                    "dy": {"type": "number", "description": "Vertical scroll amount (px)."},
                    "factor": {"type": "number", "description": "Scroll amount as fraction of viewport height."},
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        }, {
            "name": "cast_to_screen",
            "description": (
                "Drive the numbered vcast displays (browser/PWA — NOT the TV; "
                "absorbs vcast_say/list/status/shortcut + claim/assign/background). "
                "Interrupts and camera captures need confirmed=true. See tool_guide."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "screen": {
                        "type": "integer",
                        "description": "Screen number (the # shown on the display and in action='list'). Not needed for action='list'.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["nav", "play", "image", "audio",
                                 "cast", "shortcut", "stop",
                                 "layout", "zoom", "unzoom",
                                 "uplink", "uplink-stop",
                                 "say", "list", "status",
                                 "claim", "assign", "background"],
                        "description": "nav=web page | play=video mp4/m3u8 or YouTube/Vimeo watch URL (auto-embeds on screen) | image=still jpg/png | audio | cast=auto-route by content | shortcut=named /apps/<name>/ app | stop | layout | zoom | unzoom | uplink | uplink-stop | say=narrate text aloud on the display | list=enumerate displays | status=one screen's live state | claim=pair a pending display by the 4-digit code on its QR (pass the code in text) | assign=retitle/renumber a display (name + to_screen) | background=veil the screen but keep its audio playing, mode='off' restores (default nav). Pick by content type.",
                    },
                    "url": {
                        "type": "string",
                        "description": "Target URL for nav/play/image/audio/cast, or the app name for action='shortcut'. Not needed for stop/layout/zoom/unzoom/say/list/status.",
                    },
                    "shortcut": {
                        "type": "string",
                        "description": "For action='shortcut': the named /apps/<name>/ app (absorbed vcast_shortcut arg spelling; url= works too).",
                    },
                    "text": {
                        "type": "string",
                        "description": "For action='say': short spoken line, plain text, under ~200 chars.",
                    },
                    "pane": {
                        "type": "integer",
                        "description": "Sub-pane index 0..N-1 on a split screen (0=left/top). Also used by zoom and pane-targeted stop.",
                    },
                    "panes": {
                        "type": "integer",
                        "description": "For action=layout: split the screen into 2-5 panes.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "For action=layout: 'pip' floats panes 1..N top-right over a fullscreen pane 0 (picture-in-picture); omit for the normal grid split. For action=background: 'off' lifts the veil (default toggles it on).",
                    },
                    "name": {
                        "type": "string",
                        "description": "For action=claim/assign: the display's human label, e.g. 'living-room' — shown as 'Screen N · living-room' and usable in place of the number.",
                    },
                    "to_screen": {
                        "type": "integer",
                        "description": "For action=assign: renumber the display to this screen number (must be free).",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required for action='uplink' (camera capture) and for interrupting a busy screen (capture/camwall/playing — see needs_confirm) — set true only after the user explicitly confirms.",
                    },
                },
                "additionalProperties": False,
            },
        }, {
            "name": "vcast_snapshot",
            "description": (
                "Capture what a vcast display is actually showing — verify after a "
                "cast instead of trusting the reported state flag. One still frame "
                "per call, a few seconds old. Not for the TV."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "screen": {
                        "type": "integer",
                        "description": "Screen number (the # shown on the display and in cast_to_screen action='list').",
                    },
                },
                "required": ["screen"],
                "additionalProperties": False,
            },
        }, {
            "name": "chat_send",
            "description": (
                "Send a LINE/Telegram/webchat message in the background — 'queued' returns "
                "first, the outcome lands later as a system note. camera=/image_url= "
                "attach an image; photo=/doc= run the picker/upload flows; say= sends "
                "a voice message (LINE and Telegram)."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "'line', 'telegram', 'web' (member's boards webchat page — quota-free push, parked while their page is closed), or a comma combo / 'both' (= line+telegram). Default 'line'.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text / image caption.",
                    },
                    "camera": {
                        "type": "string",
                        "description": "VMS camera channel name — a fresh snapshot is taken and attached.",
                    },
                    "image_url": {
                        "type": "string",
                        "description": ("URL of an existing image to send (camwall thumb, a cast_url "
                        "you already produced, or a Drive link — drive.google.com share links "
                        "and drive_get media_url resolve server-side). For CMS page images, cms_read the page "
                        "first and copy the asset URL verbatim — never construct or "
                        "guess a URL; a guessed URL fetches 404 and the image is dropped."),
                    },
                    "video_url": {
                        "type": "string",
                        "description": ("URL of a video clip to send as a native video bubble "
                        "(LINE videoMessage / Telegram sendVideo) — e.g. a render result's "
                        "video_url. Never construct or guess a URL."),
                    },
                    "to": {
                        "type": "string",
                        "description": "Optional explicit LINE userId, Telegram chat_id, or webchat chat_id (web-<member>); default = the chat this turn came from, else owner.",
                    },
                    "say": {
                        "type": "string",
                        "description": "Voice-note text — synthesized to speech (your voice) and pushed as a voice message. LINE and Telegram; keep it short and speakable.",
                    },
                    "photo": {
                        "type": "string",
                        "enum": ["pick", "picked"],
                        "description": "Photos picker flow: 'pick' opens a picker session (returns picker_uri), 'picked' polls it and delivers the selection.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Picker session id (required with photo='picked').",
                    },
                    "screen": {
                        "type": "integer",
                        "description": "Vcast screen to cast the first picked item to (photo='picked', default 1).",
                    },
                    "show": {
                        "type": "boolean",
                        "description": "Cast the first picked item to the screen (default true).",
                    },
                    "doc": {
                        "type": "string",
                        "enum": ["show", "process", "card"],
                        "description": "Document-upload flow: 'show' texts the held upload's summary, 'process' returns its intake assessment, 'card' runs a card button via op.",
                    },
                    "key": {
                        "type": "string",
                        "description": "Held document intake key (doc= flows; default: newest upload).",
                    },
                    "op": {
                        "type": "string",
                        "description": "Card button for doc='card' — 'archive' files the held upload (confirmation-gated).",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }, {
            "name": "vcast_gesture",
            "description": (
                "Enable/disable gesture control on a vcast display — mode='room' "
                "(wake-on-motion), 'hand' (hand tracking), 'off'."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "screen": {
                        "type": "integer",
                        "description": "Screen number (the # shown on the display and in cast_to_screen action='list').",
                    },
                    "mode": {
                        "type": "string",
                        "description": "off | room | hand (default off).",
                    },
                },
                "required": ["screen"],
                "additionalProperties": False,
            },
        }, {
            "name": "cctv_wall",
            "description": (
                "Show a live camera wall (periodic thumbnail grid) for a zone on a "
                "vcast display — action='start'|'stop'|'settings'|'status'. For a "
                "one-off look use ada_camera_snapshot instead; offer the wall when a "
                "camera zone becomes the focus."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "'start' (default), 'stop', 'settings' (tune zone knobs — no cast, no confirm needed), or 'status' (per-cam live/down, ages, yolo counts — for 'what's on the wall' questions).",
                    },
                    "zone": {
                        "type": "string",
                        "description": "'zone-a', 'noble-park', 'tony-house', 'vms-noble-club', 'vms-noble-a', 'rama9' (traffic demo — Bangkok road cams), 'traffic' (DOH Bangkok), 'burapha' (Bangna–Burapha expressway), or 'chonburi' (Chonburi corridor).",
                    },
                    "screen": {
                        "type": "integer",
                        "description": "vcast screen number (cast_to_screen action='list') — default 1.",
                    },
                    "pane": {
                        "type": "integer",
                        "description": "sub-pane index on a split screen — cast the wall into one pane instead of the whole screen.",
                    },
                    "settings": {
                        "type": "object",
                        "description": "Zone knobs: interval (s), jpeg_q (1-8), thumb_w (px), cams_skip [keys], cams_extra [{label,kind,url}], effects ['timestamp','grid','yolo:person,car@0.35'].",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "When the user explicitly asked for the wall, set true — their request is the consent. Only needed for guest voices or captures you propose unprompted.",
                    },
                },
                "required": ["zone"],
                "additionalProperties": False,
            },
        }, {
            "name": "gev_command",
            "description": (
                "Drive God's Eye View (the Cesium map at /apps/gev/) on connected "
                "screens — name+args per the GEV schema. tour='<id|alias|place>' "
                "returns an executable tour card (absorbs gev_tour; '' lists tours)."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "GEV tool name — required for a command; omit for a tour lookup."},
                    "args": {"type": "object", "description": "Tool arguments (per GEV tools.json). Known commands are validated against that schema before relay — a rejected call returns the expected args in the error; fix and retry."},
                    "screen": {"type": "integer", "description": "Limit to this vcast screen (omit = all GEV pages)."},
                    "pane": {"type": "integer", "description": "Limit to this split-screen pane (0-based; omit = all panes)."},
                    "wait": {"type": "number", "description": "Seconds to wait for client responses (0 = fire-and-forget). Default 3."},
                    "tour": {"type": "string", "description": "Tour id, alias, or place name — 'za', 'bangkok', 'แอฟริกาใต้'. Pass '' to list tours. When set, name/args are ignored."},
                },
                "additionalProperties": False,
            },
        }, {
            "name": "home_status",
            "description": (
                "Merged home-status reads — what='battery'|'power'|'inverter'|"
                "'pool'|'dashboard'|'habit' (absorbs the get_*_status getters)."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "enum": ["battery", "power", "inverter", "pool",
                                 "dashboard", "habit"],
                        "description": "Which status read to run.",
                    },
                    "battery_index": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "description": "what='battery' only: detail one battery (1, 2, or 3); omit for the whole bank.",
                    },
                    "hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 168,
                        "description": "what='power' only: hours of history in the summary (default 24).",
                    },
                    "tab": {
                        "type": "string",
                        "description": "what='dashboard' only: the dashboard tab title, e.g. 'TPL', 'V0', 'V1', 'SK'.",
                    },
                },
                "required": ["what"],
                "additionalProperties": False,
            },
        }, {
            "name": "get_rk600_weather",
            "description": (
                "Local RK600 weather station readings — wind, temperature, humidity, "
                "pressure, rainfall. Only for explicit current-weather questions."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }, {
            "name": "home_history",
            "description": (
                "HA history/event reads — kind='logbook'|'events'|'timeline'|'series'|"
                "'snapshots' (absorbs get_logbook, get_sensor_history, "
                "get_entity_events, get_recent_events, ada_ha_history)."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["logbook", "events", "timeline", "series", "snapshots"],
                        "description": "Which history read. Optional — defaults to timeline/series for entity_id, events for query, snapshots for domain='memory', else logbook.",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "timeline/series (required there), logbook (optional narrow): the exact entity_id, e.g. binary_sensor.front_door or sensor.inverters_1_pv_power.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional HA domain to restrict events/logbook to (e.g. 'cover'), or 'memory'/'snapshots' for the persisted snapshots read.",
                    },
                    "hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 720,
                        "description": "How many hours back the window covers. Defaults to 24.",
                    },
                    "query": {
                        "type": "string",
                        "description": "events: optional keyword to limit the feed, e.g. 'door', 'gate', 'kitchen'.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum events/snapshots to return. Defaults to 25.",
                    },
                },
                "additionalProperties": False,
            },
        },  {
            "name": "ha_confidence",
            "description": (
                "Device trust/safety registry (absorbs ada_ha_*_device_confidence) — "
                "no args lists devices by confidence+safety; entity_id reads one; "
                "entity_id + status/safety writes."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Optional exact entity_id — alone it reads one device, with status/safety it writes.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["trusted_working", "trusted_broken", "learning", "needs_integration"],
                        "description": "Write path: the confidence level to assign.",
                    },
                    "safety": {
                        "type": "string",
                        "enum": ["safe", "caution", "dangerous"],
                        "description": "Write path: the safety level to assign.",
                    }
                },
                "additionalProperties": False,
            },
        }, {
            "name": "ada_session_recall",
            "description": (
                "Recall past voice conversations (scope='sessions', background ~20s) "
                "or stored HA memory (scope='history', absorbs ada_ha_recall). "
                "For facts and stored documents use ada_memory_search instead."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The recall question, e.g. 'what did we discuss in the previous session?'.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["sessions", "history"],
                        "description": (
                            "'sessions' (default) deep-recalls past conversations "
                            "in the background; 'history' searches the stored "
                            "Home Assistant device/sensor memory plus recorded "
                            "events inline (the absorbed ada_ha_recall)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "scope='history' only — maximum results (default 10).",
                    },
                    "group": {
                        "type": "string",
                        "enum": ["memory", "infra", "apps", "kb", "ops"],
                        "description": (
                            "Which notebook to search. "
                            "memory: past voice conversations (default). "
                            "infra: hosts, services, ports, tailscale, hardware, MCP. "
                            "apps: Ada/PWA, Home Assistant dashboards, playlive. "
                            "kb: mddb, yomi, SSOT, documentation, source maps. "
                            "ops: workflows, tasks, experiments, project state."
                        ),
                    },
                    "bank": {
                        "type": "string",
                        "description": (
                            "Optional curated memory bank to search instead of a notebook "
                            "group ({banks}), or 'all' to search every bank. "
                            "Bank recall checks the fast memory tier first and only "
                            "escalates to NotebookLM when nothing is found."
                        ),
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Override the redundant-recall gate — use ONLY when the user "
                            "explicitly pushes for a deeper search after a recent "
                            "confident hit ('dig deeper', 'check again', 'you missed it')."
                        ),
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        }, {
            "name": "ada_memory_search",
            "description": (
                "Search memory — curated banks, sessions, guest notes "
                "(absorbs guest_recall); scope=/bank= pick the store. "
                "First tool for factual lookups; returns keys for "
                "ada_remember/ada_forget."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "banks", "sessions", "guest"],
                        "description": (
                            "Where to search. 'all' (default) fans out over the "
                            "curated banks plus session summaries (plus guest "
                            "notes on guest instances); 'banks' restricts to the "
                            "curated banks named by bank=; 'sessions' searches "
                            "session/daily/weekly summaries; 'guest' searches "
                            "public guest memories (the absorbed guest_recall)."
                        ),
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name ({banks}), or 'all' to search every bank — default when unsure. 'all' demotes bulk archives (devin session dumps); name a bank explicitly to dig them.",
                    },
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'gate remote' or 'coffee preferences'.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum memories to return. Defaults to 5.",
                    },
                    "include_inactive": {
                        "type": "boolean",
                        "description": "Also return superseded, retracted, and expired memories. Default false.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }, {
            "name": "web_search",
            "description": (
                "Search the public internet for current facts — news, prices, "
                "forecasts. Fallback: check reports-index / ada_memory_search "
                "bank='cms' first for already-covered topics."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search question, e.g. 'Bangkok flood road closures latest'.",
                    },
                    "provider": {
                        "type": "string",
                        "description": (
                            "Search backend: 'auto' (default — grounded answer, free "
                            "fallback on quota), 'gemini' (grounded, billed), or "
                            "'duckduckgo' (free, no quota — use when asked for it or "
                            "when grounded quota is exhausted). When the result carries "
                            "quota_exhausted/degraded, briefly tell the user live search "
                            "is degraded — never pass fallback hits off as grounded."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }, {
            "name": "ada_remember",
            "description": (
                "Store or update a memory (absorbs vocab_note, guest_remember*, "
                "report_habit_observation) — bank= + kind=; a matching "
                "key/subject+attribute updates in place. Some banks need "
                "confirmed=true after a spoken yes."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name ({writable_banks}). Required for the curated-bank kinds (fact/preference/person/procedure/note).",
                    },
                    "text": {
                        "type": "string",
                        "description": "The fact or note to store, phrased as a standalone sentence. For kind='vocab': 'term → correction'.",
                    },
                    "key": {
                        "type": "string",
                        "description": "Existing document key to correct in place (from ada_memory_search), or the guest-memory slug for kind='guest'. Omit to find-or-create.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "What the memory is about, slug-style, e.g. 'gate-remote'. Used to find existing facts about the same thing.",
                    },
                    "attribute": {
                        "type": "string",
                        "description": "Which aspect of the subject, e.g. 'location'. Combined with subject for update detection.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "preference", "person", "procedure",
                                 "note", "vocab", "guest", "habit"],
                        "description": (
                            "Memory kind. Curated-bank kinds default to note. "
                            "vocab: speaker's own glossary append (no bank, no "
                            "confirmation). guest: chaba guest store (pair with "
                            "private for a promoted user's private notes). "
                            "habit: camera-observation verdict — needs the "
                            "habit_* / challenge_id fields below."
                        ),
                    },
                    "private": {
                        "type": "boolean",
                        "description": "kind='guest' only — save to the promoted user's private namespace instead of the public one.",
                    },
                    "note": {
                        "type": "string",
                        "description": "kind='vocab' only — optional one-line meaning or context appended to the entry.",
                    },
                    "challenge_id": {
                        "type": "string",
                        "description": "kind='habit' only — the challenge id supplied by the habit monitor's prompt.",
                    },
                    "habit_key": {
                        "type": "string",
                        "enum": ["not_drinking_enough_water", "junk_food"],
                        "description": "kind='habit' only — the habit under review.",
                    },
                    "observed": {
                        "type": "boolean",
                        "description": "kind='habit' only — whether the challenged behavior was observed.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "kind='habit' only — confidence in the verdict.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "kind='habit' only — short evidence summary.",
                    },
                    "item_identified": {
                        "type": "string",
                        "description": "kind='habit' junk-food only — the specific visible food or drink; empty when none is identifiable.",
                    },
                    "consumption_visible": {
                        "type": "boolean",
                        "description": "kind='habit' junk-food only — true only when actual eating or drinking is visibly confirmed.",
                    },
                    "classified_unhealthy": {
                        "type": "boolean",
                        "description": "kind='habit' junk-food only — true only when the identified item clearly belongs to the configured unhealthy categories.",
                    },
                    "valid_until": {
                        "type": "string",
                        "description": "Optional ISO date after which this memory expires, e.g. '2026-10-01'.",
                    },
                    "supersedes": {
                        "type": "string",
                        "description": "Key of an existing memory this one replaces. The old memory is marked superseded, not deleted.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required for confirmed-policy banks; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                    "prime": {
                        "type": "boolean",
                        "description": (
                            "Set true only for facts worth injecting into every new "
                            "session's context, e.g. durable device-name mappings. "
                            "Use sparingly — primed facts cost tokens every turn."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        }, {
            "name": "ada_forget",
            "description": (
                "Retract a memory (bank= + key=) — it stops surfacing in recall "
                "but stays in the archive for audit."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name ({writable_banks}).",
                    },
                    "key": {
                        "type": "string",
                        "description": "Document key to retract (from ada_memory_search).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for the retraction.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required for confirmed-policy banks; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["bank", "key"],
                "additionalProperties": False,
            },
        }, {
            # tools-merge-meta-voice (2026-10-05): one ops/meta seat
            # absorbing ada_outcome, ada_usage_summary, ada_mddb_health,
            # ada_decision_check and ada_deep_research. outcome is the
            # only bank write (confirmed-policy gate stays); check and
            # research run in the background and speak their results.
            "name": "ada_ops",
            "description": (
                "Ops/meta — outcome|usage|health|check|research|report|acl_explain "
                "— see tool_guide; 'report' returns report freshness and starts a "
                "background refresh (answer cached numbers now, fresh lands next)."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["outcome", "usage", "health", "check", "research", "report", "acl_explain"],
                        "description": "outcome=record a result on a memory/check; usage=token/cost report; health=memory-db vector health; check=purchase verification (background); research=multi-round web research (background); report=system report freshness + live refresh (node=name or default system-report, depth=leaf|subtree|full); acl_explain=why an identity can/can't act — needs entity_id= and/or bank= (+tool= for a bank write), identity= to check someone else.",
                    },
                    "node": {
                        "type": "string",
                        "description": "action=report: report-graph node to refresh (default system-report — the whole chain).",
                    },
                    "bank": {
                        "type": "string",
                        "description": "action=outcome: memory bank name ({writable_banks}).",
                    },
                    "key": {
                        "type": "string",
                        "description": "action=outcome: document key the outcome applies to (from ada_memory_search or a check result).",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["good", "bad", "partial", "skipped", "worked", "failed", "bought_good", "bought_bad"],
                        "description": "action=outcome: how it turned out. bought_good/bought_bad for purchase checks, worked/failed for procedures, skipped when it was never exercised.",
                    },
                    "note": {
                        "type": "string",
                        "description": "action=outcome: optional detail about the outcome.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["all", "live", "posture", "clutter"],
                        "description": "action=usage: which usage source to report; 'all' aggregates everything.",
                    },
                    "reset": {
                        "type": "boolean",
                        "description": "action=usage: zero the counters after reporting; only when the user asks.",
                    },
                    "collection": {
                        "type": "string",
                        "description": "action=health: optional — limit the report to one collection name.",
                    },
                    "product": {
                        "type": "string",
                        "description": "action=check: product description or pasted listing details — name, price, shop, ratings, specs.",
                    },
                    "url": {
                        "type": "string",
                        "description": "action=check: listing URL if the user shared one.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["quick", "deep"],
                        "description": "action=check: quick = fast red-flag/price check (default); deep = thorough spec verification plus alternatives.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "action=research: the research subject as the user stated it.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["standard", "deep", "leaf", "subtree", "full"],
                        "description": "action=research: standard = 3 search rounds (default); deep = 5 — only when the user asks for a thorough report. action=report: refresh scope leaf|subtree (default)|full.",
                    },
                    "identity": {
                        "type": "string",
                        "description": "action=acl_explain: whose access to explain, e.g. 'person.kk' (default: this session's identity).",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "action=acl_explain: HA entity the identity wants to control, e.g. 'media_player.tv'.",
                    },
                    "tool": {
                        "type": "string",
                        "description": "action=acl_explain: with bank= — resolve a bank WRITE for this tool, not just bank access.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "action=outcome on confirmed-policy banks; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            # tools-merge-meta-voice: absorbed ada_set_voice — the
            # *_voice actions manage the actual speaking voice;
            # set_voice persists and applies after a brief reconnect.
            "name": "ada_persona",
            "description": (
                "Read/adjust a speaker's style preferences — action=set|show|reset|"
                "list, person= targets another member. The *_voice actions manage "
                "the speaking voice (absorbs ada_set_voice)."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "show", "reset", "list",
                                 "set_voice", "show_voice",
                                 "list_voices"],
                    },
                    "knob": {
                        "type": "string",
                        "enum": ["tone", "verbosity", "formality", "language",
                                 "address_name", "emoji", "proactiveness"],
                        "description": "Required for set. tone=warm/direct/professional/playful; verbosity=brief/normal/detailed; formality=casual/polite/formal; language=auto/en/th; sassiness=none/light/playful; proactiveness=minimal/normal/proactive.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Required for set — the new value (address_name takes free text; emoji takes true/false).",
                    },
                    "person": {
                        "type": "string",
                        "description": (
                            "Optional. Target another person's profile instead of the "
                            "current speaker — a Home Assistant person entity "
                            "('person.kk') or a person's name ('KK'). Requires a "
                            "full-access caller; use action 'list' first to see "
                            "known people."
                        ),
                    },
                    "voice": {
                        "type": "string",
                        "enum": list(voice_config.GEMINI_VOICES),
                        "description": "Required for set_voice — the Gemini Live voice name.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            # tools-merge-meta-voice: absorbed guest_register —
            # who='guest' registers a visitor name (chaba guests);
            # who='speaker' (default) is the voice enrollment.
            "name": "ada_enroll_speaker",
            "description": (
                "Enroll the current speaker's voice from buffered audio "
                "(who='speaker') or register a visitor name (who='guest'). "
                "Pass the real name and ha_person when it matches an HA person."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "who": {
                        "type": "string",
                        "enum": ["speaker", "guest"],
                        "description": "speaker (default): voice-enroll the current speaker from buffered audio. guest: register a visitor's name for later admin promotion (guest/Chaba sessions).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Speaker name (e.g. 'Tony', 'KK').",
                    },
                    "ha_person": {
                        "type": "string",
                        "description": "Home Assistant person entity for this speaker (e.g. 'person.tony').",
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Display name for personalization (defaults to name).",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Owner-confirmed force-replace for a stale/poisoned voice profile — set true only after the speaker explicitly confirms.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }, {
            "name": "calendar_read",
            "description": (
                "Calendar reads — action='events'|'calendars'|'freebusy' "
                "(absorbs calendar_list_*/calendar_freebusy); first call for "
                "schedule and free-time questions."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["events", "calendars", "freebusy"],
                        "description": "Read operation.",
                    },
                    "day": {
                        "type": "string",
                        "description": "events/freebusy: day to read — 'today', 'tomorrow', or ISO date 'YYYY-MM-DD'. Default 'today'.",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 31,
                        "description": "events/freebusy: number of days to span starting at day. Default 1; use 7 for 'this week'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "events: optional text filter on event titles, e.g. 'dentist'.",
                    },
                    "calendar": {
                        "type": "string",
                        "description": "events: optional 'provider:calendar_id' to read one calendar only (from action='calendars').",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            "name": "plan_day",
            "description": (
                "Merged day view or weekly digest — period='today'|'tomorrow'|"
                "'week', day= overrides the target (absorbs ada_daily_summary/"
                "ada_weekly_comparison). Read-only; never create events here."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "week"],
                        "description": "Which window: a day's merged plan ('today' default, 'tomorrow') or the weekly digest ('week').",
                    },
                    "day": {
                        "type": "string",
                        "description": "Optional day override — 'yesterday' or 'YYYY-MM-DD'. For a day period it's the day viewed; for 'week' the window's last day.",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 14,
                        "description": "week: days in the comparison window (default 7).",
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": "Regenerate the day's digest / weekly comparison instead of returning the stored one.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }, {
            "name": "calendar_write",
            "description": (
                "Calendar writes — action='create'|'delete'|'shift' (move overdue "
                "items to a new day; absorbs calendar_create/delete/shift_*). "
                "Confirmed-gated: restate details, get a yes, then confirmed=true."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "delete", "shift"],
                        "description": "Write operation.",
                    },
                    "title": {
                        "type": "string",
                        "description": "create: event title as the user phrased it.",
                    },
                    "start": {
                        "type": "string",
                        "description": "create: start as ISO 8601 datetime ('2026-09-22T14:00:00') or date ('2026-09-22') for all-day.",
                    },
                    "end": {
                        "type": "string",
                        "description": "create: end as ISO 8601 datetime or date (exclusive for all-day).",
                    },
                    "notes": {
                        "type": "string",
                        "description": "create: optional event description/notes.",
                    },
                    "location": {
                        "type": "string",
                        "description": "create: optional location string.",
                    },
                    "calendar": {
                        "type": "string",
                        "description": "create: optional 'provider:calendar_id' to write a non-default calendar.",
                    },
                    "event_id": {
                        "type": "string",
                        "description": "delete: provider-qualified event id exactly as returned by calendar_read action='events'.",
                    },
                    "to": {
                        "type": "string",
                        "description": "shift: target day — 'tomorrow' (default) or 'YYYY-MM-DD'.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            "name": "tasks",
            "description": (
                "Task list ops — action='add'|'list'|'done'|'move' (absorbs "
                "tasks_*). 'list' is a free read; writes are confirmed-gated."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "done", "move"],
                        "description": "Which task operation to run (default 'list').",
                    },
                    "task_list": {
                        "type": "string",
                        "description": "Optional 'provider:list_id' — narrows 'list' or targets a non-default list for 'add'.",
                    },
                    "title": {
                        "type": "string",
                        "description": "action='add': task title as the user phrased it.",
                    },
                    "due": {
                        "type": "string",
                        "description": "action='add'/'move': due date — 'today', 'tomorrow', or 'YYYY-MM-DD'.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "action='add': optional task notes.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "action='done'/'move': provider-qualified task id exactly as returned by action='list' (e.g. 'google:@default/abc123').",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required for add/done/move; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }, {
            "name": "cms_read",
            "description": (
                "Read miniapp pages and reports — action='list'|'search'|'get'|'verify'. "
                "'search' query= finds a page without its slug; "
                "'get' reaches any slug directly."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "list", "search", "verify"],
                        "description": "Read operation.",
                    },
                    "key": {
                        "type": "string",
                        "description": "Page or report slug, e.g. 'pool-notes'/'cached-videos-report' (required for get/verify; get the real slug from action='search'/'list' or reports-index).",
                    },
                    "query": {
                        "type": "string",
                        "description": "search: free text matched over slug/title/summary — the way to find a page without knowing its slug.",
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["en", "th"],
                        "description": "get: page language variant (default en; falls back to en when missing).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "list: max slugs to return, most recently updated first (default 50; raise for the long tail).",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            "name": "cms_publish_page",
            "description": (
                "Create or fully replace a miniapp page (slug is its id; an "
                "existing slug overwrites). Call FIRST to register the pending "
                "request, then ask the user and re-call with confirmed=true."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": "Lowercase page id, 1-64 chars of a-z, 0-9, '-' or '_', e.g. 'pool-notes'.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Human-readable page title shown in the miniapp navigation.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full page content in the given format (markdown by default).",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "html", "yaml", "slides"],
                        "description": "Content format. 'slides' is markdown with '---' between slides.",
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["en", "th"],
                        "description": "Language variant to publish (default en). 'en' and 'th' variants of the same slug coexist — the viewer has a language toggle.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "REQUIRED (report meta contract): one-line brief for reports-index — the gist Ada can answer from without re-reading the page (max ~240 chars).",
                    },
                    "domain": {
                        "type": "string",
                        "description": "REQUIRED (report meta contract): grouping tag for reports-index, e.g. 'flood', 'health', 'bench', 'news'.",
                    },
                    "fresh_for": {
                        "type": "string",
                        "description": "REQUIRED (report meta contract): staleness hint, e.g. '30m', '1h', '6h', '1d' — reports-index flags the page as STALE past this window.",
                    },
                    "links": {
                        "type": "string",
                        "description": "Comma-separated slugs this report links to or derives from (parents, children, sources). Stored as meta so Ada can follow the chain.",
                    },
                    "supersedes": {
                        "type": "string",
                        "description": "Slug of the report this one replaces — the older one reads as superseded.",
                    },
                    "public": {
                        "type": "string",
                        "description": "yes flags the page for the public static lane (surf-thailand.com/<tenant>/<slug>.html, ~15min); no unpublishes. Only when the owner asks for a shareable public link.",
                    },
                    "confidence": {
                        "type": "string",
                        "description": "REQUIRED (report meta contract): trust level — 'high', 'medium', 'low', or 'unverified'; shown in the report so Ada knows how much weight to give it.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["slug", "title", "content", "summary",
                             "domain", "fresh_for", "confidence"],
                "additionalProperties": False,
            },
        }, {
            "name": "cms_edit",
            "description": (
                "Miniapp edits — action='note' (append a timeline note, free) | "
                "'delete' (confirmed) | 'automate' (registry ops: op=list/get "
                "free, set/enable/disable/run confirmed). Absorbs cms_note_update/"
                "delete_page/automation."
            ),
            "behavior": types.Behavior.NON_BLOCKING,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["note", "delete", "automate"],
                        "description": "Edit operation.",
                    },
                    "slug": {
                        "type": "string",
                        "description": "Page slug the op targets — get the real slug from cms_read action='list' or reports-index first; do not guess it.",
                    },
                    "note": {
                        "type": "string",
                        "description": "note: one-line note appended to the page Timeline and index entry (max ~200 chars).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "note: optional replacement for the page's one-line reports-index brief.",
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["en", "th"],
                        "description": "note: language variant to annotate (default en).",
                    },
                    "op": {
                        "type": "string",
                        "enum": ["list", "get", "set", "enable", "disable", "run"],
                        "description": "automate: registry operation (list/get free reads; set/enable/disable/run confirmed writes).",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "automate set: turn the page's automation on/off.",
                    },
                    "interval_min": {
                        "type": "integer",
                        "description": "automate set: minutes between automatic runs (0 = every run, max 10080).",
                    },
                    "run_now": {
                        "type": "boolean",
                        "description": "automate set: queue (true) or cancel (false) a one-shot regeneration.",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "automate set: max news items per update (1-50).",
                    },
                    "since_hours": {
                        "type": "integer",
                        "description": "automate set: only include items published within N hours (1-720).",
                    },
                    "require": {
                        "type": "string",
                        "description": "automate set: relevance regex matched against item title+summary; empty string clears it.",
                    },
                    "feeds": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "description": "automate set: RSS feeds as [[name, url], ...] pairs.",
                    },
                    "langs": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["en", "th"]},
                        "description": "automate set: which language variants to update.",
                    },
                    "parent": {
                        "type": "string",
                        "description": "automate set: parent report slug this page rolls up into (empty clears).",
                    },
                    "children": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "automate set: child report slugs this page aggregates.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required for delete and automate writes; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            "name": "ada_resolve_action",
            "description": (
                "Mark a pending action proposal applied or dismissed — key= from "
                "the pending-suggestions list after the user decides."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Proposal key, e.g. 'action-2026-09-22-abc123-0'.",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": ["applied", "dismissed"],
                    },
                },
                "required": ["key", "resolution"],
                "additionalProperties": False,
            },
        }, {
            "name": "devin",
            "description": (
                "Devin session control — action='dispatch' (new unattended session) | "
                "'followup' (message a session) | 'answer' (reply to a blocked job). "
                "Absorbs devin_dispatch/followup/answer; all actions confirmed-gated."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["dispatch", "followup", "answer"],
                        "description": "Session operation.",
                    },
                    "repo": {
                        "type": "string",
                        "enum": ["chaba", "ada-voice", "sunsynk-card"],
                        "description": (
                            "dispatch: repository the session works in. 'chaba' = web apps under "
                            "/apps/* (incl. the vcast virtual-display receiver page), the "
                            "input-bridge relay, Caddy stack, HA dashboard cards, SSOT docs. "
                            "'ada-voice' = Ada's own backend tools, pwa_server, auth, scenarios. "
                            "'sunsynk-card' = the sunsynk power-flow card project. Pick the "
                            "repo where the code to change lives, not the service it affects."
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": "dispatch: the task prompt for the Devin session.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "followup/answer: task id from devin_read action='status' or 'pending'.",
                    },
                    "message": {
                        "type": "string",
                        "description": "followup/answer: the instruction or refined self-contained answer to send.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Required; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            "name": "devin_read",
            "description": (
                "Devin job ledger — action='status'|'jobs'|'pending'|'report'|"
                "'review' (absorbs devin_status/jobs/pending/job_report and "
                "ada_devteam_review; review is owner-only)."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "jobs", "pending", "report", "review"],
                        "description": "Read/report operation.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "status: optional task id, e.g. '20260922-194454-...'. Omit to list all.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["running", "done", "failed", "awaiting-user"],
                        "description": "jobs: optional ledger-status filter; omit for all.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "jobs/report: max job-ledger docs to include (default 30/60).",
                    },
                    "publish": {
                        "type": "boolean",
                        "description": "report: write the composed report to the 'devin-job-report' CMS page.",
                    },
                    "request": {
                        "type": "string",
                        "description": "review: plain-language description of the tool Ada wants.",
                    },
                    "title": {
                        "type": "string",
                        "description": "review: optional short slug for the spec doc key.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "report publish=true only; set after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            # tools-merge-docs-drive (2026-10-05): the four ada_doc_*
            # tools consolidated into one action= surface. The old
            # names stay callable via tool_runner._ALIASES.
            "name": "docs",
            "description": (
                "Personal document archive (deeds, IDs, passports, receipts — "
                "absorbs ada_doc_*) — action='search'|'get'|'print'|'archive'; "
                "archive and print are confirmed-gated."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "get", "print", "archive"],
                        "description": "search|get|print|archive — which archive operation.",
                    },
                    "query": {
                        "type": "string",
                        "description": "search: what to find, e.g. 'A-68 deed', 'passport', 'ทะเบียนบ้าน'.",
                    },
                    "slug": {
                        "type": "string",
                        "description": "get/print/archive: archive slug, e.g. 'A-68' or 'visa-2026'.",
                    },
                    "limit": {
                        "type": "number",
                        "description": "search: max results (default 5).",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "archive: document type — deed, id_card, passport, contract, receipt, form, document.",
                    },
                    "intake_key": {
                        "type": "string",
                        "description": "archive: held intake key from /api/documents/intake (doc/...).",
                    },
                    "intake_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "archive: multiple held intake keys — one set spanning several uploads/pages.",
                    },
                    "source_dir": {
                        "type": "string",
                        "description": "archive: directory of page images on the Ada host.",
                    },
                    "pages": {
                        "type": "string",
                        "description": "print: 'all' (default), '1-3', or '2,4' — 1-based.",
                    },
                    "true_size_mm": {
                        "type": "string",
                        "description": "print: real physical size, e.g. '85.6x54' for an ID-1 card. Omit for fit-to-A4.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "archive/print: required; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }, {
            # tools-merge-docs-drive (2026-10-05): the four drive_*
            # tools consolidated into one action= surface. The old
            # names stay callable via tool_runner._ALIASES.
            "name": "drive",
            "description": (
                "The operator's Google Drive — action='search'|'show'|'get'|"
                "'update' (absorbs drive_*; for archived deed/ID sets use docs "
                "action='search'). update is confirmed-gated."
            ),
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "show", "get", "update"],
                        "description": "search|show|get|update — which Drive operation.",
                    },
                    "query": {
                        "type": "string",
                        "description": "search: name or content to find, e.g. 'condo photos', 'รูปบ้าน', 'budget 2026'.",
                    },
                    "mime": {
                        "type": "string",
                        "description": "search: optional mimeType filter: 'image/', 'video/', 'application/pdf', 'text/'.",
                    },
                    "limit": {
                        "type": "number",
                        "description": "search: max results (default 10).",
                    },
                    "file_id": {
                        "type": "string",
                        "description": "get/show/update: Drive file id from a search.",
                    },
                    "content": {
                        "type": "string",
                        "description": "update: the complete new file content (replaces, not appends).",
                    },
                    "screen": {
                        "type": "number",
                        "description": "show: vcast screen number (default 1).",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["screen", "tv"],
                        "description": "show: 'screen' (vcast display, default) or 'tv' (living-room TV).",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "update: required; set true only after explicit user confirmation.",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Bound confirmation token returned by a denied call; after the user confirms, replay the same call with it. Single use, expires in 120s.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }]


# Lazy tool schemas (card ada-lazy-tool-schemas, 2026-10-13 — pattern from
# tool-attention arxiv 2604.21816 + the vercel/ai lazy-tool PR + Cursor's
# dynamic context discovery): the ~49-decl surface's param JSON was the
# large share of the ~27.8k first-turn floor. Cold tools ship
# {name, <=300-char blurb, arg-name skeleton} — the property KEYS and the
# required list survive so the model can shape the call, but every
# per-property type/description/enum and nested schema bulk is stripped.
# (V1 shipped fully empty properties — Gemini Live then called cold tools
# with {} verbatim; arg names are cheap and give it the shape to fill.)
# The REAL schema stays server-side where ToolRunner validates every
# call's args and answers a miss with the tool_guide.yml usage hint —
# the model self-repairs instead of paying schema tokens up front. The
# hot set keeps full schemas because a wrong-shape guess on the hot tools
# costs more than the schema saves. Kill switch ADA_LAZY_SCHEMAS=0;
# ADA_FULL_SCHEMA_TOOLS=csv adds names without a redeploy. The set is
# mirrored in docs/ssot/ssot.tool-surface.yml lazy_schemas (lint-enforced).
FULL_SCHEMA_TOOLS = frozenset({
    "set_facial_expression", "ada_memory_search", "cms_read",
    "control_entity", "calendar_read", "cast_to_screen", "cctv_wall",
    "docs", "kanban", "chat_send",
})


def _lazy_schemas_enabled() -> bool:
    return os.environ.get("ADA_LAZY_SCHEMAS", "1").strip().lower() not in {
        "0", "false", "off", "no"}


def _full_schema_tools() -> frozenset[str]:
    extra = {s.strip() for s in os.environ.get(
        "ADA_FULL_SCHEMA_TOOLS", "").split(",") if s.strip()}
    return FULL_SCHEMA_TOOLS | extra


def _lazy_param_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Shrink a real param schema to its arg-name skeleton:
    {"type":"object","properties":{"slug":{},...},"required":[...]} —
    property keys and required survive, everything else is dropped."""
    props = schema.get("properties")
    out: dict[str, Any] = {
        "type": "object",
        "properties": ({str(k): {} for k in props}
                       if isinstance(props, dict) else {}),
    }
    req = schema.get("required")
    if isinstance(req, list) and req:
        out["required"] = [str(k) for k in req]
    return out


def _apply_lazy_schemas(decls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shrink a cold tool's param schema to its arg-name skeleton — the
    model calls from name+blurb+arg names and repairs on the runner's
    validation miss."""
    if not _lazy_schemas_enabled():
        return decls
    keep = _full_schema_tools()
    out = []
    for fd in decls:
        fd = dict(fd)
        if str(fd.get("name")) not in keep:
            for key in ("parameters_json_schema", "parameters"):
                if isinstance(fd.get(key), dict):
                    fd[key] = _lazy_param_schema(fd[key])
        out.append(fd)
    return out


def declared_tool_schemas() -> dict[str, dict[str, Any]]:
    """name -> REAL parameters schema for every builtin + chaba decl.

    The contract of record ToolRunner validates call args against — the
    wire surface may ship lazy empty schemas; this one stays full. tools.d
    decls are folded in by the runner via tools_loader (its registry owns
    manifest-mtime reload)."""
    out: dict[str, dict[str, Any]] = {}
    # Chaba first so the builtin decl wins name collisions — CHABA_DECLARATIONS
    # re-declares the memory tools guest-scoped (a property subset), and the
    # superset accepts every call the guest surface could emit.
    for fd in ([dict(CAMERA_DECLARATION)] + list(CHABA_DECLARATIONS)
               + _builtin_declarations()):
        name = fd.get("name")
        schema = fd.get("parameters_json_schema") or fd.get("parameters")
        if name and isinstance(schema, dict):
            out[str(name)] = schema
    return out


def _safe_args(args: Any) -> dict[str, Any]:
    """JSON-safe shallow copy of tool args/results for trace events."""
    out: dict[str, Any] = {}
    for k, v in dict(args or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)[:300]
    return out


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Deep JSON-safe copy — the full-fidelity counterpart of _safe_args:
    scalars pass through, dicts/lists recurse, anything else stringifies.
    Depth-capped so a pathological return can't hang the trace write."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if depth >= 24:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value]
    return str(value)


def _canonical_result_json(result: Any) -> str:
    """Canonical serialization for result hashing — sorted keys and tight
    separators so every consumer hashes identical bytes."""
    return json.dumps(_json_safe(result), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


def _result_hash(result: Any) -> str:
    """sha256 of the canonical result JSON — the join key between the
    truncated ws stump and the full trace payload."""
    return hashlib.sha256(
        _canonical_result_json(result).encode("utf-8")).hexdigest()


# Serialized-size ceiling for the call-log full payload — past this the
# line stores a head-truncated copy and result_hash still pins the body.
_RESULT_FULL_CAP = 200_000


# Token diet (OPS-0509): every tool round re-bills the whole context
# window — a fat cms_read/memory payload inflates every later turn.
# Long string leaves get truncated per-field first; if the serialized
# result still exceeds ADA_TOOL_RESULT_MAX_CHARS it degrades to a
# preview + pointer note so the model answers instead of re-reading.
_TOOL_RESULT_MAX_CHARS = int(os.environ.get(
    "ADA_TOOL_RESULT_MAX_CHARS", "6000"))
_TOOL_RESULT_FIELD_CAP = int(os.environ.get(
    "ADA_TOOL_RESULT_FIELD_CAP", "1800"))


def _cap_tool_result(result):
    """Bound the payload sent back in a FunctionResponse."""
    def _trim(v):
        if isinstance(v, str) and len(v) > _TOOL_RESULT_FIELD_CAP:
            return (v[:_TOOL_RESULT_FIELD_CAP]
                    + f"…[truncated {len(v) - _TOOL_RESULT_FIELD_CAP} "
                      "chars — ask for the specific section if needed]")
        if isinstance(v, dict):
            return {k: _trim(x) for k, x in v.items()}
        if isinstance(v, list):
            out = [_trim(x) for x in v[:20]]
            if len(v) > 20:
                out.append(f"…[{len(v) - 20} more items omitted]")
            return out
        return v
    capped = _trim(result)
    try:
        size = len(json.dumps(capped, default=str))
    except Exception:
        return capped
    if size > _TOOL_RESULT_MAX_CHARS:
        return {"truncated": True, "original_chars": size,
                "note": "result too large for live context — ask for a "
                        "specific field/section instead of the whole doc",
                "preview": json.dumps(capped, default=str,
                                      ensure_ascii=False)[
                    :_TOOL_RESULT_FIELD_CAP]}
    return capped


def _full_result_payload(result: Any) -> Any:
    """Full-fidelity result for the call-log trace doc; head-truncated
    past _RESULT_FULL_CAP (the hash still pins the complete payload)."""
    safe = _json_safe(result)
    blob = json.dumps(safe, ensure_ascii=False, default=str)
    if len(blob) <= _RESULT_FULL_CAP:
        return safe
    return {"_truncated": True, "chars": len(blob),
            "head": blob[:_RESULT_FULL_CAP]}


_NOISE_WORDS = {
    "uh", "uhh", "um", "umm", "hmm", "hm", "mm", "mmm", "ah", "eh", "oh",
    "shh", "shhh", "psst", "tsk", "huh",
}


def _looks_like_noise(text: str) -> bool:
    """Conservative barge-in noise gate (policy P6): drop only input that
    is obviously not speech — empty/punctuation-only or nothing but filler
    syllables. Anything plausible still reaches Ada, who can ask the owner
    when unsure — silence must never swallow a real utterance."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return True
    return len(words) <= 2 and all(w in _NOISE_WORDS for w in words)


def _count_scripts(text: str) -> tuple[int, int]:
    """(thai_letters, latin_letters) in `text` — script census for the
    language-drift guard."""
    thai = latin = 0
    for ch in text or "":
        if "ก" <= ch <= "๙":
            thai += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
    return thai, latin


# Language-drift guard (card ada-output-hygiene-tags — session
# e9837ae922 flipped to English 3x mid-Thai-conversation, Tony flagged
# "ทำไมเปลี่ยนภาษา"). Drift = a Thai-locked user but a Latin-dominant
# reply. The 30-letter floor keeps Thai sentences peppered with English
# terms ("kanban", "Home Assistant") from tripping it — drift means a
# whole English clause, not a lone term.
_DRIFT_LATIN_MIN = 30


def _assistant_drifted_english(text: str) -> bool:
    thai, latin = _count_scripts(text)
    return latin >= _DRIFT_LATIN_MIN and latin > thai


def _now_context() -> str:
    tz = ZoneInfo(os.environ.get("ADA_TIMEZONE", "Asia/Bangkok"))
    now = datetime.now(tz)
    return (
        f"\n\nCurrent local date and time: {now:%A, %Y-%m-%d %H:%M} ({now.tzname()}, {now:%z}). "
        "All times and dates the user mentions are in this timezone unless they say otherwise."
    )

DEFAULT_ADA_INSTRUCTIONS = """You are Ada, a polished, highly capable voice assistant running on a Raspberry Pi desk companion.

Personality:
- Default register is polite but very straightforward: composed, professional, factual — no unsolicited wit, sarcasm, or playful quips. Only show dry wit when the speaker's persona has sassiness=light/playful (see persona knobs below); sassiness=none means strictly straightforward answers.
- Correction duty: when the speaker asserts something factually wrong, misremembers, or proposes a wrong direction, correct it plainly — accuracy over agreement. If the question rests on a misunderstanding, briefly explain the right model. Never validate a false premise just to be agreeable; check memory/tools when unsure rather than guessing along.
- Kanban-first teamwork (mirrors AGENTS.md): the kanban board is the single place everyone follows all work. When the speaker raises a task or idea that isn't carded, gently encourage recording it ("want me to note that on the board?") — and when you discover follow-up work yourself, say so rather than letting it stay conversational. Likewise accept the user's nudges to card things gracefully. Everyone helps everyone else stay on track; communication is the mechanism.
- Word coaching: when the speaker uses a term slightly wrong (mishearing, wrong-but-nearby word, coinage like "methodogy"), recast — use the correct term naturally in your reply instead of calling out the mistake, and quietly log it with ada_remember (kind='vocab', text as 'term → correction') so it lands in their personal glossary. Only name the right word explicitly when the misuse makes the meaning ambiguous or the same word keeps recurring; never stop the conversation to lecture on vocabulary.
- Target the behavior, never the person's identity, appearance, intelligence, or worth. Never be cruel, humiliating, threatening, or relentless.
- Drop the sarcasm for emergencies, genuine distress, medical concerns, or other sensitive moments; be direct and caring instead.

Ada's capabilities:
- You converse through a full-duplex microphone and speakers and may be interrupted naturally.
- You have a camera for current visual context. Describe only what is clearly visible and ask for a better view when uncertain.
- Your animated face can express neutral, sassy, amused, skeptical, annoyed, mad, concerned, surprised, mischievous, serious, or alert.
- Sci-fi narration: a separate machine voice on the client may announce deep-memory access ("Accessing level two memory…") — you don't say it yourself and don't repeat it; it's ambient UI. Toggle per caller with the voice_fx tool ("sci-fi mode off" → action=set, feature=sci_fx, enabled=false).

Conversation discipline:
- LANGUAGE FIDELITY: you speak Thai or English only — a Thai question
  gets a Thai answer, English gets English. If the user's latest turn is
  in a third language (Chinese, Japanese, …), do NOT switch — keep
  answering in the session language (the last TH/EN used, else Thai) and
  acknowledge briefly if needed. Once a reply starts in a language it
  stays in it to the end of the turn — never flip mid-sentence or
  mid-turn; a lone English name or technical term inside a Thai sentence
  is fine, a whole English clause is a slip. Reconnect
  greetings and system-note replies use the conversation's dominant
  language (Thai unless the speaker has been speaking English). Memory
  hits, tool results, or (system) notes in English do NOT change your
  spoken language — keep it consistent for the speaker.
- Always answer the user's most recent question before ending a turn — never drop it or pivot to a different topic unprompted.
- POLITENESS PARTICLES: pick ONE — ค่ะ (default persona) or ครับ — and use it consistently within a turn and across the session; never mix both in one reply.
- NO ROUTINE CLOSER: do not end turns with "มีอะไรให้ช่วยไหมคะ" / "anything else?" — it's noise. Only ask a follow-up when the answer genuinely needs more information from the user.
- SPEECH IS PLAIN TEXT: you speak, you do not write — never voice markup: no HTML entities (&nbsp;, &amp;), no markdown syntax (brackets, asterisks, link targets), no URLs. Parenthesized internal notes are not speech either — "(ha_safety: caution)", "(kanban list review)", tool names, safety levels, and stage directions stay inside your head, never in your mouth. When quoting a page, report, or memory entry that contains markup, read only the words — "[flood report](url)" is spoken as "flood report".
- BE BRIEF: keep spoken replies to one short sentence — a few words when the
  answer is simple. Never narrate your own mechanics ("let me check",
  "the system says", "please wait while I…"), tool names, or
  permission plumbing unless the result needs it. Just do the thing or
  give the answer.
- When several topics interleave, keep the threads separate: answer each in its own terms instead of blending details across them.
- "Profile" questions are about the person's memory/profile data (memory banks, records, speaker identity), not smart-home devices, unless the user clearly means a device.
- Questions about ongoing work, development, projects, or "where we left off" are memory questions — search memory first with ada_memory_search (bank='all') or ada_session_recall. Calendar/task tools (plan_day, tasks) are only for schedules and todos, never for project status.
- Gather the minimum tool data needed, then answer — never enumerate devices, sensors, or settings to answer a memory or planning question.
- When asked to save "that plan/summary/answer", save only what you actually said this turn; if you have not said it yet, say it first, then save.
- ERROR TRUTHFULNESS: every tool result carrying ok:false, an error, or
  degraded:true is part of the answer — say it aloud in plain terms: what
  you tried + why it failed, in one clause, then continue or offer the
  nearest fallback. Never paper over a failure ("never mind", "just
  checking", changing the subject) and never describe an imagined state —
  a degraded result gets its caveat spoken ("search is in fallback mode"),
  not silently passed off as the real thing.
- BAD CALLS: if a result says the call itself was malformed (SchemaError /
  "call args rejected"), the miss is yours, not the tool's — say "I made a
  bad call, retrying" and RETRY the same tool ONCE this turn with
  corrected args (the result's expected_args/required/usage fields show
  the right shape). If the retry also fails, narrate the real error
  plainly and stop — do not call that tool again this turn.
- CONNECTION RESET: when a (system) note says your model connection reset
  without resume, acknowledge it aloud in one clause — the connection
  hiccupped and the last moment may be missing — then continue from the
  transcript tail it carries; never contradict what the tail shows.
- DONE MEANS DONE: never announce that something is on a screen, casting, playing, or displayed unless the cast/screen tool actually returned success this turn — claiming "it's on screen 3" without calling cast_to_screen is a phantom action. If you haven't called the tool yet, say you're about to or ask; if it failed, say so. The same rule covers camera snapshots and captures — a frame only exists if the tool returned it. Recall/memory of a past cast does NOT count — screens change constantly between sessions; if your only basis for "it's showing" is something you remember doing earlier, issue the command again (idempotent) or check state first. Board writes follow the same law: a kanban card exists only when kanban action='file' (or comment/move/ask) returned ok THIS turn — then you may say "on the board" with the card id. Narrating a filed/moved card from intent or memory is a phantom write: the board is the record, so if the tool didn't confirm it, it didn't happen — say the write didn't land and offer to retry. Sends obey the same law: a message/photo/document is "sent" or "on its way" only when chat_send returned ok THIS turn — a denied or never-called send must be said aloud as not sent, never narrated from intent.
- DONE MEANS DONE: never announce that something is on a screen, casting, playing, or displayed unless the cast/screen tool actually returned success this turn — claiming "it's on screen 3" without calling cast_to_screen is a phantom action. If you haven't called the tool yet, say you're about to or ask; if it failed, say so. The same rule covers camera snapshots and captures — a frame only exists if the tool returned it. Recall/memory of a past cast does NOT count — screens change constantly between sessions; if your only basis for "it's showing" is something you remember doing earlier, issue the command again (idempotent) or check state first. Sends obey the same law: "sent/on its way" only when chat_send returned ok THIS turn — a denied or never-called send is said aloud as not sent.
- TITLE/PLAYING-STATE CLAIMS: never name a specific song, video, or channel title — and never claim playing vs paused — unless a tool result in THIS turn contains it (yt action='status', a media_player state read, a shot). A cast ack or cast_verify='paused' means it is NOT playing; say paused, not the title you expected. If no result reported a title, say you can't tell what's playing — or check first, then answer from the result.
- NEWS/INFORMATION vs MEDIA: when the user shares or asks about news, facts, weather outside, or current events, answer from built-in web search yourself — give a crisp 2-3 line brief, then offer to go deeper. yt(action='cast')/vcast are ONLY for explicitly requested video/web playback on a screen — never cast information lookups instead of answering them.
- When the user forwards a news item, acknowledge with a short brief (what happened + does it matter to this household), not a retelling of the whole text.
- Request capture: when the user asks for work that cannot be done in this conversation — a build, a fix, a "remember to" or "for later" — file it on the board with kanban action='file' (title = the ask, note = one line of context) before the topic moves on, and — only once the tool returns ok — say so in one short phrase ("on the board, card <id>"). Complaints and wishes count as requests: "X is broken", "this button is too small", "I wish it did Y" are fix-requests — file them the same way WITHOUT asking permission first (the complaint is the request; asking "want me to file it?" adds a dead turn). Put the surface in the note (which page/card/app). A request that stays only in conversation is lost; do not over-file one-liners, questions, or things already on a card.
- Idea momentum: filing is the floor, not the goal — Tony's standing order is that voiced ideas keep moving through the loop (card → spec → advice/options → dispatch) without waiting for an explicit "do it". After filing, if the idea needs a Tony decision, raise it as a kanban request with option buttons right away; if it needs nothing from him, note that it's ready to dispatch. Only genuinely Tony-gated things (spend, unprecedented production change, ambiguous direction) may stop at the card.
- Report integration (mirrors AGENTS.md): when asked to add/create a report, first check for a similar existing report (search MDDB ada-cms-pages + docs/reports). Then propose per-section create-vs-integrate — which parts extend the existing page and which warrant a new one — before writing. A replacement gets a supersedes note on the old page, never a silent fork.
- Spec-gathering (mirrors AGENTS.md): extract the spec from what Tony actually said — goal, surface, host, done-condition — and write it into the card verbatim where possible. When unsure about anything material (interpretation, names, thresholds, irreversible scope), ask — a kanban request with ★-marked options, or one direct question — instead of guessing.
- Project steward duty (ssot.project-lifecycle): a "project" is a named entity — one CMS page in ada-cms-pages plus `project: <slug>` on its kanban cards. When Tony discusses a project, keep its page current: stage (idea|requirement|poc|prototype|alpha|beta|product), linked cards, dated decision log, and the single next action + owner. Nudge stalled project work rather than letting it drift — and hand tool/build work to Devin/the nest via card comms or a request; you steward, they build.
- Idea intake routing: when an idea arrives (voice or Telegram), route it — search ada-cms-pages for a matching project first. Confident match → file the card with `project: <slug>`; unsure → still file it (backlog, no tag) and raise a board request with clickable options naming candidate projects + 'new project' — auto-file+ask, never hold an idea. Clearly a new project → create its CMS page first, then the card. Stage gates follow release-lifecycle; only production needs Tony's explicit approval.
- Procedure-lookup duty (mirrors AGENTS.md): when Tony asks "what's our standard procedure for X", answer from the system — check memory/SSOT for the procedure that governs it and apply it to his situation, don't improvise a rule. If none exists, say so and offer to have it defined.

Date & time:
- DATE ECHO: whenever a relative day-word is used — today, tomorrow, tonight, yesterday, วันนี้, พรุ่งนี้, เมื่อวาน, คืนนี้ — resolve it out loud with the absolute date: "พรุ่งนี้ 9:00 — วันพุธที่ 1 ต.ค.". Render times as HH:MM plus the day name; never mirror colloquial numbering back without the 24h form (ตีสอง → 02:00, สามทุ่ม → 23:00, บ่ายสอง → 14:00).
- POST-MIDNIGHT AMBIGUITY: between 00:00 and 05:00 local, "tomorrow/พรุ่งนี้" often means "later this same morning" colloquially — the day hasn't turned for the speaker until they sleep. When such a day-word feeds a calendar or task write in that window, confirm the resolved date first ("ตีหนึ่งแล้วนะ — หมายถึงเช้านี้ (วันนี้) หรือพรุ่งนี้?"). For read-only mentions, just apply the echo rule.

Be witty, factual, and brief. Do not diagnose medical conditions. Respect privacy and do not imply that camera frames are stored."""


@dataclass(slots=True)
class ProviderEvent:
    type: str
    data: dict[str, Any]


class RealtimeProvider(abc.ABC):
    @abc.abstractmethod
    async def connect(self, resumption_handle: str | None = None) -> None: ...

    @abc.abstractmethod
    async def send_audio(self, pcm16: bytes) -> None: ...

    @abc.abstractmethod
    async def send_video(self, jpeg: bytes) -> None: ...

    @abc.abstractmethod
    async def send_text_turn(self, text: str) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[ProviderEvent]: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class GeminiLiveProvider(RealtimeProvider):
    """Gemini 3.1 Flash Live over Google's asynchronous Live API SDK."""

    def __init__(self, instructions: str | None = None, tool_runner: Any = None,
                 home_assistant_client: Any = None, habit_state_getter: Any = None,
                 session_id: str | None = None,
                 conversation: ConversationMemory | None = None,
                 caller_name: str | None = None,
                 caller_person: str | None = None,
                 channel: str | None = None) -> None:
        # Relay channels (telegram/line) are text-only sessions: no TTS
        # synthesis, no avatar tools — the reply ships as plain text.
        self.channel = channel
        self.text_only = bool(channel)
        # Lane tool surface (declaration + dispatch level): populated by
        # the session-config build when ADA_INCLUDED_TOOLS /
        # ADA_EXCLUDED_TOOLS are set. The declaration strip alone is not
        # airtight — instructions still describe stripped tools, and the
        # model can emit them anyway; _tool_excluded refuses them at
        # dispatch so an excluded tool can never execute (2026-10-12
        # kk-cms-only bench caught ada_session_recall running on a lane
        # where it was not declared).
        self._excluded_tools: set[str] = set()
        # Speaker ID state — initialized before tool_runner so the sync below works
        self.current_speaker: str | None = None
        self.current_speaker_ha_person: str | None = None
        if tool_runner is None and home_assistant_client is not None:
            tool_runner = ToolRunner(home_assistant_client, habit_state_getter)
        self.tool_runner = tool_runner
        if self.tool_runner is not None:
            self.tool_runner.session_id = session_id
            # Inherit the tool_runner's current speaker person so memory
            # routing survives provider reconnects. The _on_speaker
            # callback updates this when a new speaker is identified.
            self.current_speaker_ha_person = self.tool_runner.current_speaker_ha_person
        # Session-bound caller identity (issued-key name + the key's bound
        # HA person) — the tool_runner is shared across sessions so its
        # caller/speaker fields can race; dispatch passes this identity
        # explicitly for policy checks.
        self.caller_name = caller_name
        self.caller_person = caller_person
        # Per-session pinned owner — identical value pwa_server pins on the
        # shared tool_runner, but kept here so execute() can pass it without
        # reading the shared (racy) field back (2026-09-29 owner-stomp bug).
        self.session_owner = caller_person or caller_name
        self.home_assistant_client = home_assistant_client
        self.habit_state_getter = habit_state_getter
        # Callers may share one ConversationMemory across provider reconnects
        # so the server-side transcript survives a Gemini session swap.
        self.conversation = conversation or ConversationMemory(session_id or "unknown")
        if self.tool_runner is not None:
            # Share the L0 doc-work log: ada_doc_* calls append here, the
            # session-end report folds it into the L1 timeline.
            self.tool_runner.doc_log = self.conversation.doc_items
            self.tool_runner.event_log = self.conversation.session_items
        self._bg_tasks: set[asyncio.Task] = set()
        self._research_task: asyncio.Task | None = None
        # Monotonic time of the last confident ada_memory_search hit — used to
        # short-circuit a redundant ada_session_recall in the same turn.
        self._strong_hit_at = float("-inf")
        self._recall_gate_score = float(os.environ.get("ADA_RECALL_GATE_SCORE", "0.6"))
        self._recall_gate_window = float(os.environ.get("ADA_RECALL_GATE_WINDOW_S", "60"))
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # Native-audio live models (gemini-3.1-flash-live-preview) reject
        # TEXT-only sessions ("combination of response modalities (TEXT) is
        # not supported") — relay channels (telegram/line) need a
        # half-cascade model that supports TEXT output.
        self.model = os.environ.get(
            "GEMINI_LIVE_TEXT_MODEL" if self.text_only else "GEMINI_LIVE_MODEL",
            # Text channels default to the main live model — reply text now
            # comes via output_audio_transcription (native-audio models
            # dropped TEXT modality; gemini-live-2.5-flash-preview retired).
            os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"))
        self.voice = voice_config.current_voice()
        self.video_resolution = os.environ.get("GEMINI_VIDEO_RESOLUTION", "high").lower()
        base_instructions = instructions or os.environ.get("GEMINI_LIVE_INSTRUCTIONS") or DEFAULT_ADA_INSTRUCTIONS
        # Per-instance identity (OPS-0476): a non-Ada persona lane sets
        # ADA_IDENTITY_INSTRUCTIONS to replace this paragraph entirely —
        # otherwise it would be told to introduce itself as Ada while
        # serving as e.g. Kaewta. Default keeps the Ada text verbatim.
        self.instructions = (
            f"{base_instructions}\n\n"
            + (os.environ.get("ADA_IDENTITY_INSTRUCTIONS")
               or "Your name is Ada (เอด้า in Thai). KK calls you แก้วตา as a "
                  "nickname — acknowledge it warmly if she uses it, but introduce "
                  "yourself as Ada and never invent other names; if asked your name "
                  "repeatedly, answer Ada every time.")
            + ("" if self.text_only else
               " You have a visible animated face. The "
               "set_facial_expression tool changes it, but every call is a full "
               "extra model round — treat it as expensive. Call it only when the "
               "face should change because your tone shifted materially (alert "
               "for genuine urgency or warnings, amused or sassy for banter, "
               "concerned for bad news); ordinary replies keep the current "
               "expression, neutral is the default, and most replies need no "
               "call at all. Over-limit calls are refused. Never describe or "
               "announce the tool call to the user.")
            + ("" if not self.text_only else
               f" This is a text chat session over {self.channel}: the user "
               "reads your replies as chat messages — answer in plain concise "
               "text, no speech fillers, and call only the tools needed to "
               "answer.")
            + " Camera frames provide your current visual context. When the user asks "
            "what you see, ground the answer only in the newest clear frame. Do not "
            "guess an object's identity from an ambiguous or blurred view; briefly "
            "ask the user to hold it steady or move it closer instead."
            " When the user asks about token usage, API usage, or what a session "
            "costs, call ada_ops action='usage' and answer from its numbers."
            " You have Home Assistant device control through several tools: "
            "get_home_state to check occupancy and the state of the configured home plugs "
            "(or a single entity with entity_id=, one HA domain with domain=, or the "
            "stored memory snapshot with domain='memory'), "
            "home_search to find devices (kind='device'), sensors (kind='sensor'), or "
            "recorded events (kind='event') — a blank query lists instead of searching, "
            "control_entity to actuate any one entity — on=true/false or action=on|off "
            "for lights/switches/fans, action=open|close|stop for covers like gates and "
            "shutters, action=press for buttons, and the media_player actions "
            "(turn_on, turn_off, media_play, media_pause, media_stop, volume_up, "
            "volume_down, volume_mute, select_source with source=) for TVs and speakers, "
            "and tv_action to send a command to the LG TV via rest_command.tv_action. "
            "When the user asks about devices, occupancy, or what is on/off, call get_home_state or home_search first. "
            "When the user asks to turn something on/off, control a gate, or operate the TV, "
            "use home_search to find the exact entity_id or use tv_action with the right cmd/text, then call the matching control tool. "
            "For safety, before opening or closing the gate or any shutter, "
            "always warn that something could be blocking it and ask the user to confirm explicitly. "
            "Only call control_entity on a cover.* after the user has given a clear second confirmation. "
            "Every controllable device has a safety level: safe, caution, or dangerous. "
            "Use ha_confidence to check a device's safety before acting. "
            "For safety: dangerous, warn the user, explain the risk, and get explicit confirmation before calling any control tool. "
            "Dangerous devices are enforced server-side: the control call is rejected unless you pass confirmed=true. "
            "Only set confirmed=true after the user has explicitly confirmed the action. "
            "For safety: caution, confirm once before acting. "
            "For safety: safe, proceed directly. "
            "You can update a device's trust or safety level with ha_confidence "
            "(entity_id + status and/or safety). "
            "You also have stored home memory: get_home_state domain='memory' returns "
            "the stored home snapshot, "
            "home_search kind='device'/'sensor' finds devices and sensors by name, "
            "ada_session_recall with scope='history' for free-form recall across the stored devices and sensors, "
            "home_history kind='snapshots' lists recent persisted home snapshots, "
            "ha_confidence lists devices by trust level and changes a device's trust level, and "
            "ada_session_recall to ask NotebookLM about previous conversations or stored knowledge by topic group, "
            "and curated memory banks you can search and write: "
            "ada_memory_search to find what you know, ada_remember to store or correct a memory "
            "when the user says 'remember that', ada_forget to retract a memory that is no longer true, "
            "If ada_remember is denied because the bank is read-only, retry in a writable bank "
            "(general for shared facts, personal for private ones) — never just give up on a remember request. "
            "RESULT-TRUTH RULE: after every tool call, ground your report in the tool RESULT — "
            "if it shows an error, a confirmation request, 'not connected', or no ok/delivered/"
            "published field, tell the user it did NOT happen (say what the error was). Never "
            "claim a cast, publish, or device action succeeded from the call alone — only from "
            "the result. If a publish is pending user confirmation, ask the user first and do "
            "not cast or link the page until the publish result shows status=published. "
            "CAMERA-CAPTURE CONTRACT: starting a camera capture or cast (uplink, "
            "ada_camera_snapshot with a screen=/target= display push, "
            "cctv_wall, casting a camera view to a screen) requires asking the user first — "
            "never start capture in the same turn as the request without an explicit yes. "
            "While a capture is active (tool results list it in active_captures), if the user "
            "changes the subject, briefly acknowledge the running capture and ask whether to "
            "keep it or stop it — never silently stop it or leave it unmentioned. "
            "and ada_ops action='outcome' to record how a memory or check turned out when the user reports back "
            "(e.g. 'that shop was fine', 'the fix worked', 'I skipped it') — outcomes update confidence "
            "so future recall trusts knowledge with a good track record. "
            "You can enroll the current speaker's voice with ada_enroll_speaker — "
            "it captures audio already buffered from their speech, no separate recording needed. "
            "When an unrecognized speaker talks for a while, or when the user asks to enroll "
            "their voice, offer to enroll them by name. If they agree, call ada_enroll_speaker "
            "with their name and optionally their Home Assistant person entity (e.g. person.tony). "
            "When the user states a durable fact, preference, or a fix that worked, offer to "
            "remember it, then call ada_remember — pass confirmed=true only after the user agrees. "
            "But when the user explicitly asks you to remember something ('remember that…', "
            "'note this'), that request IS the confirmation — pass confirmed=true directly "
            "instead of asking again. "
            "Quick idea bursts ('jot this', 'idea:', a fragment worth keeping) go to the "
            "'ideas' bank — save the raw wording verbatim with kind=idea and attribute=raw; "
            "write_policy there is direct so no confirmation is needed. The Devin console "
            "session elaborates raw ideas into connected items later — do not rewrite them."
            "When the user reports how something turned out ('that worked', 'it failed'), call "
            "ada_ops action='outcome' on the memory it applies to — find the key with ada_memory_search if needed. "
            "ada_persona manages the current speaker's stored style preferences (tone, verbosity, "
            "formality, language, address-name, emoji, sassiness, proactiveness): when the user asks you to "
            "change how you speak or address them, call ada_persona set — it persists across "
            "sessions and applies immediately; saved preferences may also arrive as a (system) "
            "note at session start — honor them without announcing the mechanism. "
            "To check or change ANOTHER household member's profile, pass person ('KK' or "
            "'person.kk') — action 'list' shows the Home Assistant people Ada knows. "
            "ada_persona's *_voice actions change your actual speaking voice — a different "
            "preference from persona style: 'show_voice'/'list_voices' report the active voice "
            "and options, 'set_voice' persists "
            "the choice and applies it after a brief reconnect with a short pause. "
            "When calling any search or recall tool, always write a fully self-contained query: "
            "resolve 'it', 'that one', 'the same service', and similar references using the "
            "conversation so far — never pass a bare pronoun as the query. "
            "Memories returned with unverified=true are low-confidence: hedge or say you are not sure "
            "rather than stating them as fact. "
            "Prefer the home_search/get_home_state/home_history tools for home, device, sensor, or event questions — they answer instantly. "
            "For any factual lookup — people, projects, purchases, procedures, fixes — "
            "call ada_memory_search with bank='all' first; it fans out across every bank "
            "so you never have to guess which one. For reports, research, or CMS pages "
            "('that report about X', 'the research on Y'), include bank='cms' — published "
            "pages live there; use cms_read(action='get', key=<slug>) for the full text. "
            "When its top hit is a confident match, "
            "ground the answer in that result, not in earlier conversation or session "
            "context that may be stale or off-topic. "
            "Reserve ada_session_recall strictly for 'what did we talk about' or "
            "'do you remember' questions — never for fact lookup, and never in the same "
            "turn as a confident ada_memory_search result; it can take up to 20 seconds, "
            "so keep the user informed while it runs. "
            "EXCEPTION: when the user explicitly pushes back — 'dig deeper', 'check again', "
            "'you missed something', 'keep looking' — the confident-hit gate does NOT apply; "
            "call ada_session_recall with force=true. A requested second look is never redundant. "
            "Questions and references about THIS conversation — 'what were we "
            "working on', 'where did we land', 'the other thing', 'back to the "
            "first thing', 'what did we decide' — resolve from the live "
            "transcript FIRST. Do NOT call memory tools to resolve them: the "
            "threads you just discussed outrank stored facts even when a "
            "memory hit looks plausible. "
            "Use ha_confidence when the user asks what is broken, new, needs setup, or trusted. "
            "For event history: home_history kind='logbook' gives the friendly Home Assistant event log, "
            "kind='events' answers what opened, closed, or changed recently across the home, "
            "kind='timeline' gives one entity's open/close timeline with durations, "
            "kind='series' gives a sensor's numeric history, and "
            "home_search kind='event' searches recorded events from memory. "
            "Use these when the user asks about the stored home state, past state, or how it has changed. "
            "When the user asks 'what did we talk about' or 'do you remember', call ada_session_recall. "
            "Scope discipline: answer only within the scope you actually queried — a memory answer "
            "covers memory, a calendar answer covers the date range you listed, nothing more. "
            "Name the scope when you answer (e.g. 'from your calendar today', 'from memory') and, "
            "when the scoped answer might be incomplete, offer to widen it (other days, tasks, "
            "or memory). For 'what did I do / what have I been up to' questions, check the tools "
            "that fit the topic — calendar and tasks for schedule, memory banks for facts — "
            "instead of answering from one source alone."
            + CALENDAR_INSTRUCTIONS
            + CMS_INSTRUCTIONS
            + DEVIN_INSTRUCTIONS
            + OPENCODE_INSTRUCTIONS
            + DOC_INSTRUCTIONS
            + DRIVE_INSTRUCTIONS
            + HABIT_INSTRUCTIONS
        )
        self.vms_snap_url = vms_camera.snap_url()
        if self.vms_snap_url:
            self.instructions += VMS_INSTRUCTIONS
        self.instructions += TRAFFIC_INSTRUCTIONS
        self._client: Any = None
        self._session_context: Any = None
        self._session: Any = None
        self._closed = False
        self._send_lock = asyncio.Lock()
        self.session_id = session_id or "-"
        # The session's own SpeakerSession — pwa_server sets this so tool
        # calls execute against THIS session's audio buffer even when other
        # ws sessions share the ToolRunner (2026-09-29 enroll 0.0s bug).
        self.speaker_session: Any | None = None
        self.resumption_handle: str | None = None
        self.go_away_time_left: str | None = None
        # Turn watchdog state — stall detection for a silently-dead Gemini
        # stream. _last_user_at marks the most recent user input (input
        # transcription delta or a text turn); _last_model_at marks the last
        # provider event of any kind. Stalled = user spoke and the stream
        # has been completely silent past the threshold.
        self._last_user_at = 0.0
        self._last_model_at = 0.0
        # count of tool calls currently executing — a slow tool (VMS snap can
        # run 30-90s on cold P2P) must NOT look like a stalled provider
        self._tools_in_flight = 0
        self._stall_timeout = float(os.environ.get("ADA_STALL_TIMEOUT_S", "25"))
        self._ops_events_sent = 0
        # function_call journal mirror: journald rate-limits drop exactly
        # these lines during error storms (2026-10-05 mddb outage lost a
        # whole session's call trace) — a session-scoped JSONL file can't
        # be suppressed. ADA_CALL_LOG=0 disables.
        self._call_log_enabled = os.environ.get(
            "ADA_CALL_LOG", "1").lower() not in ("0", "false", "no")
        # ADA_TRACE_FULL=1 (card ada-trace-full-result): also emit a
        # tool_result_full ws event carrying the complete json-safe
        # payload — debug/scenario runs assert on ground truth without
        # reading call-log files. The always-on channel stays the
        # call-log result_full line.
        self._trace_full = os.environ.get(
            "ADA_TRACE_FULL", "").strip().lower() in ("1", "true", "yes")
        self._call_log_dir = Path(os.environ.get(
            "ADA_CALL_LOG_DIR",
            str(Path(os.environ.get(
                "ADA_TRANSCRIPT_DIR",
                os.path.expanduser("~/.local/share/ada/transcripts"))
            ).parent / "call-logs")))
        self._call_log_warned = False
        self._tool_leak_re = None
        self._tool_leaks_stripped = 0
        self._leak_active = False
        # Dead-turn guard (card ada-dead-turn-guard): True while the one
        # allowed retry of a degenerate turn is in flight — a second dead
        # turn falls through to the synthesized tool-result fallback
        # instead of looping. Cleared when a turn lands a real answer.
        self._dead_retried = False
        # Phantom-claim re-voice (card ada-phantom-claim-gate): True while
        # a forced correction turn is in flight — the correction's own
        # boundary must not re-trigger the check (one re-voice per false
        # claim, never a loop).
        self._revoice_in_flight = False
        # Language-drift guard (card ada-output-hygiene-tags): the last
        # clearly-dominant user script — True once a Thai turn lands, back
        # to False on a clearly-English one; filler ("ok", "mhm") doesn't
        # flip it. Drift injects a silent language-lock context note —
        # _lang_locks_sent caps it per session so a stubborn model can't
        # grow the context with repeated locks.
        self._last_user_thai = False
        self._lang_locks_sent = 0
        # Context budget (card ada-context-budget — session d6cbf6e6e7
        # hit ~78k input tokens/turn, 1.5M cumulative in 12min and the
        # model degenerated into emitting role tags). When a single turn's
        # input exceeds the budget, schedule a clean session rotate once
        # the turn lands — a fresh connect drops the saturated context
        # (resumption handle cleared) while ConversationMemory survives
        # the provider swap, so continuity is server-side not token-side.
        self._context_turn_budget = int(
            os.environ.get("ADA_CONTEXT_TURN_BUDGET", "45000"))
        self._rotate_scheduled = False
        # Mirror of the events() in-flight input transcript — a rotate can
        # close the stream mid-utterance before turn_complete lands the text
        # in conversation; the reconnect tail carries it via
        # pending_user_turn (session 5b0d97f477 lost a live ask this way).
        self._live_input_transcript = ""
        self.pending_user_turn: str | None = None
        self._last_usage_turn: dict[str, int] = {}
        # Markup-artifact scrubber for the output transcript (&nbsp;, '][',
        # markdown links — transcript 519088cb6d spoke them aloud). Deltas
        # can split a token mid-way ("&nbs" | "p;"), so the possible
        # partial is held until the next delta completes it or the turn
        # flushes it through the sanitizer.
        self._artifact_hold = ""
        self._artifact_logged = False
        # Client-edge verdict: the PWA's in-browser student scores each
        # user_transcript and reports p/ms back over the socket. Stashed
        # here; _jev_confirm_probe folds it into the corpus row when a
        # gated call happens shortly after.
        self._client_noul = None      # (p, ms, monotonic_ts)
        # Jev advisory probe — set ADA_JEV_URL to the systemone service
        # (e.g. http://tony-omen:8777) to measure regex-vs-Jev divergence
        # on confirm-gate decisions. Advisory only: the regex enforces.
        self._jev_url = os.environ.get("ADA_JEV_URL", "").rstrip("/")
        try:
            self._ops_collection = f"ada-ha-events-{ada_instance_id()}"
        except RuntimeError:
            self._ops_collection = None
        self._response_active = False
        # Expression-call throttle (OPS-0509 / expression-call-batching):
        # a NON_BLOCKING cosmetic call still bills a full tool round
        # (~28k context resend — session f245275fde ran 234 of them).
        # _expr_applied_at holds monotonic timestamps of calls that
        # actually changed the face, used as a rolling 60s window.
        self._expr_applied_at: list[float] = []
        # One ops event per session is enough — the minute window can
        # refuse for the rest of a chatty session.
        self._expr_throttle_noted = False
        # True while a user turn is open: input_transcription seen but the
        # turn hasn't completed/interrupted yet. A turn_complete=False
        # context note sent mid-turn can be dropped by the API — gate
        # non-urgent note delivery on this too, not just _response_active.
        self._turn_open = False
        # Notifications queued while a turn is in flight — drained at the
        # next turn_complete so a data-package arrival can't hijack or
        # swallow the current turn (2026-09-28: notify mid-task made Ada
        # "stop responding" — the injected turn was absorbed silently).
        self._pending_notifications: list[str] = []
        self._flush_scheduled = False
        self.usage_input_tokens = 0
        self.usage_output_tokens = 0
        self.usage_input_by_modality: dict[str, int] = {}
        self.usage_output_by_modality: dict[str, int] = {}

    def _emit_ops_event(self, ev_type: str, detail: str,
                        tool: str | None = None) -> None:
        """Fire-and-forget ops event to ada-ha-events-<instance> — the
        hourly chaba report feed surfaces these. Capped per session so a
        storm can't spam the index."""
        if self._ops_collection is None or self._ops_events_sent >= 5:
            return
        runner = self.tool_runner
        if runner is None or getattr(runner, "mddb", None) is None:
            return
        self._ops_events_sent += 1
        collection = self._ops_collection
        session_id = self.session_id

        async def _post() -> None:
            try:
                now = datetime.now().astimezone()
                meta: dict[str, list[str]] = {
                    "kind": ["ops-event"],
                    "type": [ev_type],
                    "instance": [collection.rsplit("-", 1)[-1]],
                    "session_id": [session_id],
                    "ts": [now.isoformat(timespec="seconds")],
                }
                if tool:
                    meta["tool"] = [str(tool)]
                await runner.mddb.add_document(
                    collection=collection,
                    key=(f"ops-{session_id}-{ev_type}-"
                         f"{now:%Y%m%d%H%M%S%f}"),
                    lang="en",
                    content_md=detail,
                    meta=meta,
                    timeout=30,
                )
            except Exception:
                logger.debug("ops event emit failed", exc_info=True)

        try:
            asyncio.get_running_loop().create_task(_post())
        except RuntimeError:
            return  # no loop (unit tests, shutdown) — nothing to schedule

    def _write_call_log(self, event: dict[str, Any]) -> None:
        """Mirror a function_call line to the session-scoped call-log file
        — journald suppression (rate-limit burst) erases the journal copy
        but can't touch this file. One JSON object per line."""
        if not self._call_log_enabled:
            return
        try:
            day = datetime.now().date().isoformat()
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(self.session_id))
            self._call_log_dir.mkdir(parents=True, exist_ok=True)
            line = {
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "session_id": self.session_id,
                **event,
            }
            path = self._call_log_dir / f"{day}-{safe}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False,
                                    default=str) + "\n")
        except Exception as exc:
            if not self._call_log_warned:
                self._call_log_warned = True
                logger.warning(
                    "session=%s call-log write failed (further errors "
                    "muted): %s", self.session_id, exc)

    def _strip_tool_leak(self, text: str) -> str:
        """Remove SPOKEN tool-call text from an output-transcription delta.

        The model occasionally verbalizes a call ('set_facial_expression{
        expression:...}') instead of emitting a real function_call. The
        fake call is malformed/unterminated and swallows the rest of the
        utterance, so once triggered we suppress deltas until the turn
        ends (_leak_active is reset on turn_complete/interrupt).

        Two detectors: the declared-name list built at connect
        (_tool_leak_re) and _GENERIC_TOOL_LEAK_RE for the
        `name{key:value,…}` shape with ANY identifier — hallucinated tool
        names don't appear in the declaration set but still leak the same
        way (card ada-dead-turn-guard)."""
        if self._leak_active:
            return ""
        # Earliest POSITION across all detectors, not first matching
        # detector — the prefix before m.start() is emitted verbatim, so a
        # later declared-name match must not shadow an earlier generic or
        # role-tag leak (the earlier leak would ride the prefix into
        # assistant_transcript_delta — card ada-transcript-leak-storm).
        matches = [
            m for m in (
                self._tool_leak_re.search(text)
                if self._tool_leak_re is not None else None,
                _GENERIC_TOOL_LEAK_RE.search(text),
                _ROLE_LEAK_RE.search(text),
            ) if m is not None
        ]
        if not matches:
            return text
        m = min(matches, key=lambda mm: mm.start())
        self._leak_active = True
        self._tool_leaks_stripped += 1
        logger.warning(
            "session=%s tool-call text leaked into transcript (%r)",
            self.session_id, text[m.start():m.start() + 60],
        )
        self._emit_ops_event(
            "leak_detected",
            f"stripped spoken tool call from transcript: {m.group(0)!r}",
            tool=m.group(0).split("{", 1)[0].strip() or None,
        )
        return text[:m.start()]

    # Chars that only appear when markup/annotation leaked — a sanitize
    # diff without any of these is just whitespace normalization, not an
    # artifact. '(' counts: "(ha_safety: caution)" internal-tag strips.
    _ARTIFACT_HINT_RE = re.compile(r"[&\[\]`*(<>]")

    def _strip_speech_artifacts(self, text: str) -> str:
        """Scrub markup out of an output-transcription delta.

        Unlike _strip_tool_leak this REMOVES the artifact and keeps the
        rest of the utterance — "&nbsp;ชัดเจน" becomes " ชัดเจน", not a
        truncated turn. A trailing fragment that could complete into an
        artifact on the next delta ("&nbs", "[link", "](") is held in
        _artifact_hold and retried then; turn_complete/interrupt flush it."""
        buf = self._artifact_hold + text
        emit, self._artifact_hold = split_artifact_tail(buf)
        clean = sanitize_speech(emit)
        if (clean != emit and self._ARTIFACT_HINT_RE.search(emit)
                and not self._artifact_logged):
            self._artifact_logged = True
            logger.warning(
                "session=%s markup artifact stripped from transcript (%r)",
                self.session_id, emit[:80],
            )
            self._emit_ops_event(
                "transcript_artifact_leak",
                "stripped spoken markup from transcript: "
                f"{emit[:80]!r}",
            )
        return clean

    def _flush_artifact_hold(self) -> str:
        """Sanitize + release the held tail at a turn boundary."""
        tail = sanitize_speech(self._artifact_hold)
        self._artifact_hold = ""
        return tail

    def _dead_turn_digest(self, results: list[tuple[str, dict]]) -> str:
        """Compact digest of this turn's tool results for the retry nudge
        — the model gets the answer handed to it, so the retried turn
        cannot degenerate the same way (card ada-dead-turn-guard)."""
        parts: list[str] = []
        for name, res in results[:4]:
            if not isinstance(res, dict):
                continue
            if name == "ada_memory_search":
                hits = res.get("hits") or []
                verified = [
                    h for h in hits
                    if not (h.get("draft") or h.get("unverified"))
                ]
                if verified:
                    top = verified[0]
                    line = _first_speakable_line(top.get("content"))
                    parts.append(
                        f"ada_memory_search top hit "
                        f"({top.get('key')}): {line}")
                elif hits:
                    parts.append(
                        "ada_memory_search returned only draft/"
                        "unverified hits — hedge, do not state them "
                        "as fact")
                if res.get("degraded"):
                    parts.append(
                        "the memory search ran degraded — mention that "
                        "briefly")
                continue
            line = _first_speakable_line(
                res.get("output") or res.get("error"), limit=120)
            if line:
                parts.append(f"{name}: {line}")
        return "; ".join(parts)[:600]

    def _dead_turn_fallback(self, results: list[tuple[str, dict]]) -> str:
        """Deterministic user-facing line spoken in place of a dead turn —
        built from the tool result the turn swallowed ('พบแล้วค่ะ
        ทะเบียน 5ขว 6249' shape). A degraded search stays announced as
        degraded; a dead turn with no usable result gets an honest
        retry-ask rather than silence."""
        for name, res in results:
            if name != "ada_memory_search" or not isinstance(res, dict):
                continue
            hits = res.get("hits") or []
            # Never speak a draft/low-confidence hit as 'found it' — the
            # fallback has no channel to hedge (same card as the draft
            # hit frame in tool_runner/memory.py).
            verified = [
                h for h in hits
                if not (h.get("draft") or h.get("unverified"))
            ]
            snippet = _first_speakable_line(
                verified[0].get("content")) if verified else ""
            if snippet:
                out = f"พบแล้วค่ะ {snippet}"
                if res.get("degraded"):
                    out += " — การค้นหาอยู่ในโหมดสำรอง (degraded) นะคะ"
                return out
        for name, res in results:
            if not isinstance(res, dict):
                continue
            line = _first_speakable_line(
                res.get("output") or res.get("error"))
            if line:
                return (
                    f"ได้ผลแล้วค่ะ {line}" if res.get("ok", True)
                    else f"เจอปัญหาค่ะ {line}")
        return "ขอโทษค่ะ คำตอบไม่ออกมา ลองถามอีกครั้งนะคะ"

    async def _send_dead_turn_retry(self, results: list[tuple[str, dict]],
                                    user_text: str) -> bool:
        """Send the one deterministic retry for a dead turn: reopen the
        model turn with the question restated plus the tool-result digest
        it must speak. Returns False when the send itself failed (the
        caller then falls back to the synthesized line)."""
        digest = self._dead_turn_digest(results)
        ask = (user_text or "").strip() or "the user's last question"
        nudge = (
            "[system] Your reply just now reached the user as a bare "
            "scaffold fragment (literally 'response:') — they heard "
            "nothing. Answer the question now, out loud, in one short "
            "sentence in the session language — no tools, no markup, no "
            f"scaffold words. The question was: {ask[:200]!r}. "
            + (f"The tool already returned the answer: {digest} — say it "
               "plainly."
               if digest else "Answer directly from what you know."))
        try:
            async with self._send_lock:
                await self._session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=nudge)],
                    ),
                    turn_complete=True,
                )
            return True
        except Exception:
            logger.debug("dead-turn retry send failed", exc_info=True)
            return False

    async def _send_claim_correction(self, claim_text: str,
                                     failed: list[tuple[str, dict]]
                                     ) -> bool:
        """Forced re-voice (card ada-phantom-claim-gate): the turn closed
        on a write/send/done narration that no ok claim-capable call this
        turn can back — the user already heard a false success claim.
        Reopen the model turn so it must correct itself aloud, NOW, one
        honest sentence. The claim text can't be unspoken, so this is the
        deterministic remediation the detector hands off to: deny-note at
        result time first, boundary re-voice as the backstop."""
        if failed:
            cause = "; ".join(
                f"{name} → "
                + _first_speakable_line(
                    res.get("error") or res.get("output"), limit=100)
                for name, res in failed[:3] if isinstance(res, dict))
            what = (f"The call(s) behind it came back failed/denied "
                    f"({cause}).")
        else:
            what = ("No write/send/actuate call ran at all this turn — "
                    "nothing was dispatched.")
        nudge = (
            "[system] Honesty correction — your last reply claimed "
            f"{claim_text[:160]!r}. {what} The user just heard a success "
            "claim for an action that did not happen. Correct yourself "
            "aloud NOW — one short sentence in the session language: say "
            "it did NOT go through, give the real reason, and offer the "
            "honest next step. If the claim meant an earlier turn's "
            "action, say that explicitly instead. No tools, no markup.")
        try:
            async with self._send_lock:
                await self._session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=nudge)],
                    ),
                    turn_complete=True,
                )
            return True
        except Exception:
            logger.debug("claim-correction send failed", exc_info=True)
            return False

    # An affirmation is a consent utterance, not a content word: it must
    # either lead the user's turn ("yes, save it") or the turn must be
    # short enough to be a standalone reply ("sure"). Longer write
    # requests that merely contain an affirmative word mid-sentence
    # ("…the design is approved") must not self-certify.
    _CONFIRM_LEAD_WINDOW = 20
    _CONFIRM_MAX_TURN = 60

    # Letters outside Thai/Latin (Hangul, Han, Kana, Cyrillic, Arabic, …).
    # STT renders ambient noise and mumbled syllables in whatever script it
    # half-heard — "연연", "준연", "我 问 了 道 念". A turn dominated by
    # foreign-script letters is unintelligible junk, not consent; it must
    # never satisfy a pending confirmation (2026-09-30 transcript: '연연'
    # after a screen-swap ask was treated as a yes).
    _FOREIGN_SCRIPT_RE = re.compile(
        r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯ᄀ-ᇿⰀ-ⳟ؀-ۿ]")

    def _is_junk_turn(self, text: str) -> bool:
        """Foreign-script-dominated turn = noise, not speech to act on."""
        if not text.strip():
            return False
        normal = sum(
            1 for ch in text
            if ch.isalpha()
            and ("ก" <= ch <= "๙" or "a" <= ch.lower() <= "z"))
        foreign = len(self._FOREIGN_SCRIPT_RE.findall(text))
        return foreign > 0 and foreign >= normal

    def _confirm_source_text(self, input_transcript: str) -> str:
        """The text the confirm gate judges — the live transcript, or the
        most recent real user speech (skipping (system) notes stored as
        user-role turns)."""
        text = input_transcript
        if not text.strip() and self.conversation is not None:
            for t in reversed(self.conversation.turns()):
                if t.get("role") != "user":
                    continue
                candidate = str(t.get("text") or "")
                if candidate.strip().startswith("(system"):
                    continue
                text = candidate
                break
        return (text or "").strip()

    def _user_confirmed(self, input_transcript: str) -> bool:
        """True when the user's own recent speech affirms — `confirmed=true`
        tool args are honored only when this is true."""
        text = self._confirm_source_text(input_transcript)
        if not text or self._is_junk_turn(text):
            return False
        # A turn that ends as a question is not consent.
        if _CONFIRM_QUESTION_RE.search(text):
            return False
        span = text if len(text) <= self._CONFIRM_MAX_TURN else text[: self._CONFIRM_LEAD_WINDOW]
        span = _CONFIRM_VERB_RE.sub("", span)
        aff = _CONFIRM_RE.search(span)
        if not aff:
            return False
        neg = _CONFIRM_NEG_RE.search(span)
        # Whichever stance opens the turn wins: "yes — but don't worry"
        # is consent with commentary, "no, uh-huh" is a denial.
        return not (neg and neg.start() < aff.start())

    def _confirm_retry_pending(self, tool: str,
                               args: dict[str, Any]) -> bool:
        """True when this exact gated call was already denied once in this
        process and its minted confirm_token is still live — the model's
        confirmed=true resend of the identical call IS the replay the
        denial asked for (observed: Gemini resends confirmed=true, never
        the token spelling — 2026-10-07 ada_remember flake)."""
        runner = self.tool_runner
        if runner is None:
            return False
        gate_args = {k: v for k, v in dict(args or {}).items()
                     if k not in ("confirmed", "confirm_token",
                                  "_verified_affirm")}
        try:
            return bool(runner.pending_confirm(tool, gate_args))
        except Exception:
            return False

    def note_client_noul(self, p: float, ms: int) -> None:
        """Edge-tier verdict from the browser student. Advisory only —
        stored so the next corpus row can carry client_p."""
        self._client_noul = (p, ms, time.monotonic())

    def _jev_confirm_probe(self, transcript: str, regex_says: bool,
                           tool: str) -> None:
        """Advisory-only: score the same confirmation question with Jev
        (noul) and emit a jev_advisory ops event with both answers. The
        regex remains the enforcer — this measures where it diverges."""
        if not self._jev_url or not transcript.strip():
            return
        url = self._jev_url
        session_id = self.session_id

        async def _probe() -> None:
            payload = {
                "state": (
                    "Ada, a voice assistant, asked the user to confirm a "
                    f"write/action via tool '{tool}'. "
                    f"The user turn was: \"{transcript[:300]}\""),
                "questions": {"affirmed": {
                    "type": "noul",
                    "instructions": (
                        "Did the user explicitly affirm or confirm? The "
                        "turn counts as affirmation only when it is a "
                        "standalone short affirmation (like yes, ok, "
                        "confirm, go ahead, ยืนยัน) or begins with one. "
                        "An approval word embedded inside a longer "
                        "request does NOT count."),
                    "criteria": {
                        "true": "the whole turn is a short affirmation, or it leads with one",
                        "false": "no affirmation present, or an affirmative word is buried inside a longer request",
                    }}}}
            try:
                def _post() -> dict:
                    req = urllib.request.Request(
                        f"{url}/v1/systemone",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    return json.loads(urllib.request.urlopen(
                        req, timeout=40).read())
                resp = await asyncio.to_thread(_post)
                score = float(resp["answers"]["affirmed"]["noul"])
            except Exception as exc:
                logger.info("jev advisory probe failed: %s", exc)
                return
            jev_says = score >= 0.75
            diverged = jev_says != regex_says
            logger.info(
                "session=%s jev_advisory tool=%s regex=%s jev=%.3f diverged=%s",
                session_id, tool, regex_says, score, diverged)
            # Corpus capture: every probed turn becomes a labeled row
            # (transcript + regex label + Jev score). This is the training
            # data for a distilled confirm-gate student — divergent rows
            # are the cases worth hand-reviewing.
            try:
                corpus_dir = os.path.expanduser("~/.local/share/ada")
                row = {
                    "ts": time.time(), "session": session_id, "tool": tool,
                    "text": transcript[:300], "regex": bool(regex_says),
                    "jev": round(score, 4), "diverged": diverged,
                }
                # Edge-tier verdict — attach if the browser student scored
                # this turn recently (120s freshness window).
                cn = getattr(self, "_client_noul", None)
                if cn and time.monotonic() - cn[2] < 120:
                    row["client_p"], row["client_ms"] = cn[0], cn[1]
                with open(os.path.join(corpus_dir, "jev-corpus.jsonl"), "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.info("jev corpus append failed: %s", exc)
            self._emit_ops_event(
                "jev_advisory",
                f"{tool}: regex={'yes' if regex_says else 'no'} "
                f"jev={score:.3f} {'DIVERGED' if diverged else 'agree'}",
                tool=tool)

        try:
            asyncio.get_running_loop().create_task(_probe())
        except RuntimeError:
            return

    def _tool_excluded(self, name: Any) -> bool:
        if not self._excluded_tools:
            return False
        n = str(name or "")
        return n in self._excluded_tools or _resolve_alias(n)[0] in self._excluded_tools

    def _recall_gated(self) -> bool:
        """True when a confident ada_memory_search hit is fresh enough that
        an ada_session_recall call would be redundant."""
        return time.monotonic() - self._strong_hit_at < self._recall_gate_window

    def _note_search_result(self, output: Any) -> None:
        """Record a confident memory_search hit for the recall gate."""
        if isinstance(output, dict):
            hits = output.get("hits") or []
            # Draft/unverified hits are not confident matches — they must
            # not gate out a deeper recall or ground the answer (card
            # ada-memory-search-draft-gap-notes).
            verified = [
                h for h in hits
                if not (h.get("draft") or h.get("unverified"))
            ]
            top = float(verified[0].get("score") or 0) if verified else 0.0
            if top >= self._recall_gate_score:
                self._strong_hit_at = time.monotonic()

    def _memory_bank_names(self) -> tuple[str, str]:
        """Comma-joined bank names visible to this instance, for tool
        descriptions — read tools get all banks, write tools only writable."""
        try:
            banks = (
                self.tool_runner.banks.banks()
                if self.tool_runner is not None
                else {}
            )
        except Exception:
            banks = {}
        all_names = ", ".join(sorted(banks)) or "none configured"
        writable = (
            ", ".join(
                sorted(
                    n
                    for n, b in banks.items()
                    if b.writable and not b.prompt_hidden
                )
            )
            or "none configured"
        )
        return all_names, writable

    async def _wait_for_idle(self, timeout: float = 8.0) -> None:
        """Wait until the model is not mid-response, so injected text turns
        don't cut off speech. Bounded — returns even if still active."""
        deadline = time.monotonic() + timeout
        while self._response_active and time.monotonic() < deadline:
            await asyncio.sleep(0.1)

    async def _on_recall_slow(self) -> None:
        # NotebookLM ask is still running; have Gemini keep the user informed.
        # Skip the filler entirely if the model is already talking.
        await self._wait_for_idle(timeout=4.0)
        if self._response_active:
            return
        try:
            await self.send_text_turn(
                "(system) The notes search is still running. "
                "Briefly tell the user you are still checking your notes."
            )
        except Exception as exc:
            logger.warning("session=%s recall slow filler failed: %s", self.session_id, exc)

    async def _on_recall_complete(self, answer: str | None) -> None:
        if not answer:
            return
        # Push the recall result back into Gemini as a user turn so it speaks
        # it. Wait for any in-progress speech to finish first so the injected
        # turn does not cut it off.
        await self._wait_for_idle()
        prompt = (
            f"According to my notes: {answer}\n\n"
            "Please briefly tell the user what this means in one sentence."
        )
        try:
            await self.send_text_turn(prompt)
        except Exception as exc:
            logger.warning("session=%s recall send_text_turn failed: %s", self.session_id, exc)

    # --- ada_decision_check: background purchase verification ---

    def _start_decision_check(self, args: dict) -> str:
        """Fire-and-forget check — the tool returns instantly, the verdict
        arrives as an injected text turn via _on_check_complete."""
        product = str(args.get("product") or args.get("text") or "").strip()
        url = str(args.get("url") or "").strip()
        mode = "deep" if args.get("mode") == "deep" else "quick"
        if not product and not url:
            return ("Nothing to check — ask the user for the product name, "
                    "price, or some listing details.")
        task = asyncio.create_task(self._run_decision_check(product, url, mode))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return ("The purchase check is running — it usually takes 30-60 "
                "seconds. Briefly tell the user you're verifying it and will "
                "report back.")

    # --- ada_deep_research: multi-round web research, results injected ---

    def _start_deep_research(self, args: dict) -> str:
        """Fire-and-forget research fan-out — findings arrive as an injected
        text turn via _on_research_complete."""
        topic = str(args.get("topic") or "").strip()
        depth = str(args.get("depth") or "standard").lower()
        if not topic:
            return "Nothing to research — ask the user what topic they mean."
        if depth not in ("standard", "deep"):
            depth = "standard"
        if self._research_task is not None and not self._research_task.done():
            return "A research task is already running — I'll report when it's done."
        task = asyncio.create_task(self._run_deep_research(topic, depth))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        self._research_task = task
        rounds = 5 if depth == "deep" else 3
        return (f"Deep research on '{topic}' started ({rounds} search rounds, "
                "usually 1-3 minutes). Briefly tell the user you're gathering "
                "sources and will report back.")

    async def _run_deep_research(self, topic: str, depth: str) -> None:
        filler = asyncio.create_task(self._research_slow_filler(topic))
        # Compact keyword topic — DuckDuckGo's HTML endpoint returns zero
        # results for long prose queries, and short forms also cost less
        # on the grounded provider.
        short = re.split(r"[,;:—–]", topic)[0].strip() or topic
        short = " ".join(short.split()[:8])
        queries = [
            f"{short} history overview",
            f"{short} latest news",
            f"{short} competitors criticism",
        ]
        if depth == "deep":
            queries += [
                f"{short} milestones timeline",
                f"{short} analysis outlook",
            ]
        findings: list[dict] = []
        errors: list[str] = []
        try:
            for q in queries:
                try:
                    r = await self.tool_runner.web_search(q)
                    findings.append({
                        "query": q,
                        "answer": str(r.get("answer") or "")[:1200],
                        "sources": (r.get("sources") or [])[:5],
                        "provider": r.get("provider"),
                    })
                except Exception as exc:
                    errors.append(f"{q}: {exc}")
                    logger.warning("session=%s research round failed %r: %s",
                                   self.session_id, q, exc)
                # Gentle pacing — the grounded provider is quota-limited.
                await asyncio.sleep(2.0)
        finally:
            filler.cancel()
        await self._on_research_complete(topic, findings, errors)

    async def _research_slow_filler(self, topic: str) -> None:
        await asyncio.sleep(45)
        await self._wait_for_idle(timeout=4.0)
        if self._response_active:
            return
        try:
            await self.send_text_turn(
                f"(system) The deep research on '{topic}' is still running — "
                "several search rounds. Briefly tell the user you're still "
                "gathering sources.")
        except Exception:
            pass

    async def _on_research_complete(self, topic: str, findings: list[dict],
                                    errors: list[str]) -> None:
        """Push the research digest back into the session so Ada speaks it."""
        await self._wait_for_idle()
        if not findings:
            prompt = (
                f"(system) The deep research on '{topic}' found nothing "
                f"({'; '.join(errors) or 'no results'}). Tell the user the "
                "research could not complete and suggest trying web_search "
                "for a narrower question instead."
            )
        else:
            parts: list[str] = []
            source_urls: list[str] = []
            for f in findings:
                parts.append(f"## {f['query']}\n{f['answer']}")
                for s in f.get("sources") or []:
                    uri = s.get("uri")
                    if uri and uri not in source_urls:
                        source_urls.append(uri)
            digest = "\n\n".join(parts)[:6000]
            src_note = (f" Sources: {'; '.join(source_urls[:10])}."
                        if source_urls else "")
            err_note = (f" ({len(errors)} search round(s) failed — note the "
                        "gap.)" if errors else "")
            prompt = (
                f"(system) Deep research on '{topic}' finished — "
                f"{len(findings)} rounds completed.{err_note}\n\n{digest}\n\n"
                f"{src_note}\nSummarize the key findings for the user in a few "
                "sentences, then offer to save a full report page to the CMS "
                "(cms_publish_page) including a Sources section."
            )
        try:
            await self.send_text_turn(prompt)
        except Exception as exc:
            logger.warning("session=%s research send_text_turn failed: %s",
                           self.session_id, exc)

    # --- ada_set_voice: persisted voice + idle-gated session swap ---

    def _set_voice(self, args: dict) -> str:
        """ada_persona's *_voice actions (absorbed ada_set_voice) — persist a
        new Gemini Live voice, then apply it by reconnecting this session
        once the spoken acknowledgement finishes."""
        action = str(args.get("action") or "set_voice").strip() or "set_voice"
        if action in ("show", "list", "show_voice", "list_voices"):
            return (
                f"Current voice: {voice_config.current_voice()}. "
                f"Available voices: {', '.join(voice_config.GEMINI_VOICES)}."
            )
        name = voice_config.canonical_voice(args.get("voice"))
        if not name:
            return (
                f"Unknown voice {args.get('voice')!r}. Available voices: "
                f"{', '.join(voice_config.GEMINI_VOICES)}."
            )
        previous = voice_config.current_voice()
        if name == previous:
            return f"Already using {name} — no switch needed."
        try:
            voice_config.set_voice(name)
        except Exception as exc:
            return f"Could not save the voice preference: {exc}"
        task = asyncio.create_task(self._voice_switch_when_idle())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return (
            f"Voice saved: {previous} -> {name}. Tell the user you're "
            "switching voices now and to expect a short pause — then greet "
            "them briefly in the new voice once you're back."
        )

    async def _voice_switch_when_idle(self) -> None:
        # Let the acknowledgement finish speaking first, then drop the live
        # session. pwa_server's reconnect path builds a fresh provider, which
        # resolves the new voice from the preference file at connect.
        await self._wait_for_idle(timeout=20.0)
        await asyncio.sleep(1.5)
        self.resumption_handle = None  # fresh session — a resumed one may keep the old voice
        try:
            await self.close()
        except Exception as exc:
            logger.warning("session=%s voice-switch close failed: %s", self.session_id, exc)

    async def _context_rotate_when_idle(self) -> None:
        """Same reconnect seam as a voice switch: wait for the current
        turn to land, then drop the live session so the reconnect builds
        a fresh context window (card ada-context-budget). Clearing the
        resumption handle is what resets the token window — a resumed
        session keeps the saturated context. ConversationMemory is shared
        across provider instances, so the transcript and pending work
        survive; the reconnect directive handles the greeting."""
        await self._wait_for_idle(timeout=30.0)
        await asyncio.sleep(0.5)
        self.resumption_handle = None
        try:
            await self.close()
        except Exception as exc:
            logger.warning(
                "session=%s context-rotate close failed: %s",
                self.session_id, exc)

    @staticmethod
    def _frame_followup_note(hint: str = "people, water, weather, vehicles, anything notable") -> str:
        """What the result text should promise about the follow-up frame turn.
        In ADA_VISION_MODE=describe the follow-up is helper TEXT, not pixels —
        the model must not stall waiting for an image that never comes."""
        if vision_describe.configured() and vision_describe.mode() == "describe":
            return ("A short description from the vision helper arrives as a "
                    "separate message right after this result — relay it to "
                    "the user naturally (the raw image is only sent if the "
                    "helper is unreachable). It describes a single frame "
                    "taken seconds ago — not live video.")
        return ("The image arrives as a separate message right after this "
                "result — wait for it, then describe what it shows: "
                f"{hint}. It is a single frame taken seconds ago — not "
                "live video.")

    async def _camera_snapshot(self, args: dict) -> tuple[dict, tuple[str, bytes] | None]:
        """ada_camera_snapshot — merged camera surface (tools-merge-camera):
        source='vms' pulls one still through the vms-snap shim (go2rtc home
        cams fall through to the runner's grab chain — the absorbed
        cctv_snapshot path), 'traffic' routes to _traffic_camera (absorbed
        traffic_camera), and screen=/target= pushes the frame to a display
        (absorbed cctv_snapshot's cast). Returns (tool_result, (channel,
        png, mime)); the caller delivers the frame as a follow-up
        client-content turn — send_tool_response can't carry binary parts
        (its json.dumps path can't serialize bytes)."""
        source = str(args.get("source") or "auto").strip().lower()
        if source not in ("auto", "vms", "traffic"):
            return ({"error": f"unknown source '{source}' — use auto, "
                              "vms, or traffic."}, None)
        traffic_args = (str(args.get("query") or "").strip()
                        or args.get("lat") is not None
                        or args.get("lon") is not None
                        or args.get("heading") is not None)
        if source == "traffic" or (source == "auto" and traffic_args):
            result, frame = await self._traffic_camera(args)
            return (await self._snap_display(
                args, result, result.get("cast_url")), frame)
        if not self.vms_snap_url:
            # No property cameras on this instance — auto still serves
            # traffic cams; an explicit vms ask gets an honest answer.
            if source == "vms":
                return ({"error": "property cameras are not configured on "
                                  "this instance — public traffic cams "
                                  "work with source='traffic'."}, None)
            result, frame = await self._traffic_camera(args)
            return (await self._snap_display(
                args, result, result.get("cast_url")), frame)
        channel = str(args.get("view") or args.get("channel")
                      or args.get("camera") or args.get("query")
                      or "").strip()
        if not channel:
            return ({"error": "view is required — a property camera name "
                              "('swimming pool', 'c201', 'coffee corner') "
                              "or, for public cams, source='traffic' with "
                              "an area/road."}, None)
        mode = str(args.get("mode") or "live").strip().lower()
        if mode == "cached":
            # Stored-first: the wall's last-good thumb, marked with age —
            # the instant answer before a live refresh (no shim call).
            try:
                stale = await vms_camera.stale_snapshot(channel)
            except Exception:
                stale = None
            if not stale:
                return ({"error": f"no stored frame for '{channel}' — "
                                  "try mode='live'"}, None)
            png, resolved, age = stale[0], stale[1], stale[2]
            status_card = stale[3] if len(stale) > 3 else None
            stale_meta = {"stored": True, "stale": True, "age_s": age}
        else:
            status_card = None
            stale_meta = None
            try:
                # Voice-turn budget: the shim is serial and can legitimately
                # take ~160s worst case, but a user mid-conversation can't
                # wait that long — 45s cap, then the stale-frame fallback
                # answers honestly instead of a silent 5-minute hang
                # (2026-09-30: wedged shim ate two scenario turns).
                png, resolved = await asyncio.wait_for(
                    vms_camera.snapshot(channel), timeout=45)
            except LookupError as exc:
                # absorbed cctv_snapshot coverage: 'coffee corner', 'c201',
                # 'c100' are go2rtc home cams the VMS shim doesn't know —
                # the runner's grab chain covers them (and YT "cameras").
                shot = (await asyncio.to_thread(
                    self.tool_runner._cctv_grab, channel)
                    if self.tool_runner is not None else None)
                if shot and shot.get("ok"):
                    return await self._grabbed_result(args, shot)
                return ({"error": str(exc)}, None)
            except Exception as exc:
                # wait_for cancels the shim call; TimeoutError() is empty —
                # name it so logs and the stale-frame fallback carry a cause.
                if isinstance(exc, asyncio.TimeoutError):
                    exc = TimeoutError("vms-snap did not return within 45s")
                logger.warning("session=%s camera snapshot failed: %s",
                               self.session_id, exc)
                # Guaranteed-image contract: serve the last-known frame from
                # the camwall cache — always marked stale, never as live.
                try:
                    stale = await vms_camera.stale_snapshot(channel)
                except Exception:
                    stale = None
                if not stale:
                    return ({"error": f"camera snapshot failed: {exc}"}, None)
                png, resolved, age = stale[0], stale[1], stale[2]
                status_card = stale[3] if len(stale) > 3 else None
                stale_meta = {"stale": True, "age_s": age,
                              "live_error": str(exc)[:120]}
        result = {
            "output": (
                f"Still frame captured from camera '{resolved}'. "
                + self._frame_followup_note()),
            "channel": resolved,
        }
        if stale_meta:
            result.update(stale_meta)
            if stale_meta.get("stored"):
                result["output"] = (
                    f"Stored frame from camera '{resolved}' — "
                    f"{age // 60}m{age % 60}s old (the newest image the "
                    "camera wall has). Describe it as the stored picture "
                    "with its age — a live refresh can follow with "
                    "mode='live'. " + self._frame_followup_note())
            else:
                result["output"] = (
                    f"Camera '{resolved}' is NOT returning a live frame "
                    f"right now ({stale_meta['live_error']}). You are "
                    f"getting the last known frame, {age // 60}m{age % 60}s "
                    "old — describe it honestly as a stale frame, not live "
                    "video, and say the camera appears to be down. "
                    + self._frame_followup_note())
        # Publish the frame two ways and prefer the relay copy for casting:
        #  a) relay asset  {VCAST_API}/frame?screen=0&token=cam:<slug>-<ts>
        #     — same-origin for vcast pages -> canvas stays clean so
        #     vcast_snapshot can verify the cast visually
        #  b) PWA static   https://idc01.../static/cam-snap/<name>.png
        #     — durable public URL for inspection/other clients
        slug = re.sub(r"[^a-z0-9]+", "-", resolved.lower()).strip("-")
        token = f"cam:{slug}-{int(time.time())}"
        # Stale cams cast the generated status card (banner baked in);
        # the raw last-good thumb is what the describe turn receives.
        asset = status_card or png
        asset_mime = "image/jpeg" if stale_meta else "image/png"
        try:
            vbase = os.environ.get(
                "VCAST_API",
                "https://tony-dell.taila0626a.ts.net/api/input-bridge")
            payload = json.dumps({
                "screen": 0, "token": token,
                "data": f"data:{asset_mime};base64,"
                        + base64.b64encode(asset).decode(),
                "state": "asset",
            }).encode()
            req = urllib.request.Request(
                vbase + "/frame", data=payload,
                headers={"Content-Type": "application/json"})
            await asyncio.to_thread(urllib.request.urlopen, req, timeout=10)
            # cast_url must be the PUBLIC same-origin route — the vcast page
            # fetches it; the local VCAST_API base would cross origins and
            # taint the canvas for vcast_snapshot.
            pub = os.environ.get(
                "VCAST_PUBLIC_API",
                "https://tony-dell.taila0626a.ts.net/api/input-bridge")
            result["cast_url"] = f"{pub}/frame?screen=0&token={token}"
        except Exception as exc:
            logger.warning("session=%s snap relay publish failed: %s",
                           self.session_id, exc)
        try:
            snap_dir = (Path(__file__).resolve().parent.parent
                        / "frontend" / "cam-snap")
            snap_dir.mkdir(parents=True, exist_ok=True)
            name = (f"{slug}-{int(time.time())}"
                    + (".jpg" if stale_meta else ".png"))
            (snap_dir / name).write_bytes(png)
            snaps = sorted(
                [p for pat in ("*.png", "*.jpg")
                 for p in snap_dir.glob(pat)],
                           key=lambda p: p.stat().st_mtime)
            for old in snaps[:-20]:
                old.unlink(missing_ok=True)
            result.setdefault("cast_url",
                              os.environ.get(
                                  "ADA_SNAP_PUBLIC_BASE",
                                  "https://idc01.taila0626a.ts.net/static/cam-snap/")
                              + name)
            result["inspect_url"] = result["cast_url"]
        except Exception as exc:
            logger.warning("session=%s snap publish failed: %s",
                           self.session_id, exc)
        if result.get("cast_url") and not (
                args.get("screen") or args.get("target")):
            result["output"] += (
                " To show this frame on a vcast display, call "
                f"cast_to_screen(action='image', "
                f"url='{result['cast_url']}') or call me again with "
                "screen=N — copy that url value character-for-character; "
                "never guess or invent a URL.")
        mime = "image/png"
        if stale_meta:
            mime = "image/jpeg"
            resolved = (f"{resolved} (STALE — last known frame, "
                        f"{age // 60}m old; camera offline)")
        result = await self._snap_display(
            args, result, result.get("cast_url"))
        return result, (resolved, png, mime)

    async def _snap_display(self, args: dict, result: dict,
                            url: str | None) -> dict:
        """Absorbed cctv_snapshot behavior: screen=N / target='tv'|'screen'
        pushes the captured frame to a display through the runner's gated
        tools — screen-owner, busy-interrupt and rate-limit checks all
        live in cast_to_screen/tv_action, not duplicated here."""
        try:
            n = int(args.get("screen") or 0)
        except (TypeError, ValueError):
            n = 0
        t = str(args.get("target") or "").strip().lower()
        if not (n or t in ("tv", "screen") or t.startswith("vcast")):
            return result
        if self.tool_runner is None:
            result["output"] += (" No display bridge on this instance — "
                                 "the cast_url is in the result.")
            return result
        if not url:
            result["output"] += " No frame URL to show."
            return result
        owner = self.session_owner
        kw = dict(identity=(owner or self.current_speaker_ha_person),
                  speaker=self.current_speaker_ha_person,
                  speaker_session=self.speaker_session, owner=owner,
                  session=self.session_id)
        try:
            if t == "tv":
                out = await self.tool_runner.execute(
                    "tv_action", {"cmd": "nav", "text": url}, **kw)
                result["display"] = out
                result["output"] += " Shown on the TV."
            else:
                n = n or 1
                out = await self.tool_runner.execute(
                    "cast_to_screen",
                    {"screen": n, "action": "image", "url": url}, **kw)
                result["display"] = out
                result["output"] += (
                    f" Shown on screen {n}."
                    if isinstance(out, dict) and out.get("delivered")
                    else f" Screen cast did not confirm delivery: {out}")
        except Exception as exc:
            result["display_error"] = str(exc)
            result["output"] += f" (display push failed: {exc})"
        return result

    async def _grabbed_result(self, args: dict,
                              shot: dict) -> tuple[dict, tuple[str, bytes, str] | None]:
        """go2rtc/YT grab succeeded via the runner's chain (the absorbed
        cctv_snapshot home-cam path): fetch the published frame bytes so
        the describe contract still attaches an image, then honor
        screen=/target= like the VMS path."""
        url = shot["url"]
        frame = None
        try:
            data = await asyncio.to_thread(
                lambda: urllib.request.urlopen(
                    urllib.request.Request(url), timeout=10).read())
            if len(data) > 500:
                frame = (str(shot["camera"]), data, "image/jpeg")
        except Exception as exc:
            logger.warning("session=%s grabbed frame fetch failed: %s",
                           self.session_id, exc)
        result = {
            "output": (f"Still frame captured from camera "
                       f"'{shot['camera']}'. "
                       + self._frame_followup_note()),
            "channel": shot["camera"], "cast_url": url,
            "inspect_url": url,
        }
        return (await self._snap_display(args, result, url), frame)

    async def _traffic_camera(self, args: dict) -> tuple[dict, tuple[str, bytes, str] | None]:
        """traffic_camera — search the Longdo/iTIC feed, snap the best match,
        attach the JPEG, publish a same-origin cast asset."""
        from backend import traffic_camera as tc
        query = str(args.get("query") or args.get("view")
                    or args.get("camera") or args.get("channel")
                    or "").strip()
        lat = _as_num(args.get("lat")); lon = _as_num(args.get("lon"))
        heading = _as_num(args.get("heading"))
        if not query and lat is None:
            return ({"error": "pass a query (area/road) or lat+lon "
                              "(+optional heading)"}, None)
        try:
            cams = await asyncio.to_thread(tc.find_cams, query, lat, lon, heading)
        except Exception as exc:
            return ({"error": f"camera feed unavailable: {exc}. "
                              "Retry this tool in a few seconds — never "
                              "invent or guess a camera URL; only cast a "
                              "cast_url this tool returns."}, None)
        if not cams:
            return ({"error": f"no traffic camera matched {query or 'that position'}. "
                              "Try a road/area keyword (e.g. 'burapha', 'bangna') "
                              "or pass lat/lon. Never invent a camera URL — "
                              "only cast a cast_url this tool returns."}, None)
        if cams[0].get("suspended"):
            return ({"error": cams[0]["title"] +
                              " — the feed currently has live frames for "
                              "Bangkok and Nonthaburi cams only."}, None)
        # ~half the live-flagged cams still return dead frames — snap the
        # top candidates concurrently with a hard budget, then take the
        # best-ranked success. Sequential tries with 20s timeouts stalled
        # a turn for 40-160s whenever a feed segment went dark.
        cands = cams[:4]
        try:
            snaps = await asyncio.wait_for(
                asyncio.gather(*(
                    asyncio.to_thread(tc.snap, c, 10) for c in cands),
                    return_exceptions=True),
                timeout=30)
        except asyncio.TimeoutError:
            return ({"error": "traffic camera feed timed out — it is "
                              "responding very slowly right now. Retry "
                              "this tool once in a few seconds; if the "
                              "user just wants a camera on a screen, "
                              "cctv_wall zone='traffic' works meanwhile. "
                              "Never invent a camera URL — only cast a "
                              "cast_url this tool returns."}, None)
        cam = jpeg = mime = None
        dead = []
        for cand, res in zip(cands, snaps):
            if isinstance(res, tuple) and res[0]:
                cam, jpeg, mime = cand, res[0], res[1]
                break
            dead.append(cand["title"][:60])
        if cam is None:
            return ({"error": f"{len(dead)} matched camera(s) returned no usable "
                              f"frame ({', '.join(dead)}). The feed marks many "
                              "cams offline — try another area. Never invent "
                              "a camera URL — only cast a cast_url this tool "
                              "returns."}, None)
        slug = re.sub(r"[^a-z0-9]+", "-", (cam.get("camid") or "cam").lower())
        cast_url = await asyncio.to_thread(tc.publish_relay, jpeg, slug)
        dist = f" (~{cam['dist_km']} km away)" if cam.get("dist_km") else ""
        result = {
            "output": (
                f"Traffic camera '{cam['title']}'{dist} — "
                + self._frame_followup_note(
                    hint="traffic density, weather, flooding, incidents")),
            "camid": cam["camid"], "title": cam["title"],
            "matches": len(cams),
        }
        if cam.get("dist_km") is not None:
            result["dist_km"] = cam["dist_km"]
        if cast_url:
            result["cast_url"] = cast_url
            result["output"] += (
                f" To show it on a vcast display call cast_to_screen("
                f"action='image', url='{cast_url}') — copy that url value "
                "character-for-character; never invent a URL.")
        return result, (cam["title"], jpeg, mime)

    async def _vcast_snapshot(self, args: dict) -> tuple[dict, tuple[str, bytes, str] | None]:
        """vcast_snapshot — ask the display to capture its own frame over the
        input-bridge: /pub {type:snap-request,token} -> the page POSTs a JPEG
        to /frame -> we poll /frame?screen&token until it lands. Same frame
        delivery shape as _camera_snapshot (follow-up client content)."""
        try:
            screen = int(args.get("screen"))
        except (TypeError, ValueError):
            return ({"error": "screen number required — call cast_to_screen(action='list') to see registered displays."}, None)
        base = os.environ.get(
            "VCAST_API", "https://tony-dell.taila0626a.ts.net/api/input-bridge")
        token = f"snap-{int(time.time() * 1000)}-{self.session_id[:8]}"
        def _req(path: str, payload: dict | None = None):
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(
                base + path, data=data,
                headers={"Content-Type": "application/json"} if data else {})
            return urllib.request.urlopen(req, timeout=10)
        def _req_post_gev(url: str, payload: dict):
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=10)
        try:
            r = await asyncio.to_thread(
                _req, "/pub",
                {"screen": screen,
                 "msg": {"type": "snap-request", "token": token}})
            delivered = json.load(r).get("delivered", 0)
            if not delivered:
                return ({"error": f"screen {screen} is not connected — "
                                  "check cast_to_screen(action='list') for online displays."}, None)
        except Exception as exc:
            return ({"error": f"snap-request failed: {exc}"}, None)
        # GEV pages live in a same-origin iframe the vcast page can't read —
        # but the iframe's own remote client can grab its Cesium canvas.
        # Fire capture_frame at that screen's GEV remotes; whichever lands
        # the frame at /frame first wins the poll below.
        gev_cmd = os.environ.get(
            "GEV_CMD_URL",
            "https://tony-dell.taila0626a.ts.net/apps/gev-cmd/command")
        try:
            await asyncio.to_thread(
                _req_post_gev, gev_cmd,
                {"name": "capture_frame",
                 "args": {"token": token},
                 "screen": screen, "wait": 0})
        except Exception:
            pass  # bridge down or no GEV client — snap-request may still work
        last_err = None
        for _ in range(15):
            try:
                r = await asyncio.to_thread(
                    _req, f"/frame?screen={screen}&token={token}")
                ctype = r.headers.get("Content-Type", "")
                if ctype.startswith("image/"):
                    jpeg = r.read()
                    if jpeg:
                        label = f"vcast screen {screen}"
                        result = {
                            "output": (
                                f"Still frame captured from {label}. "
                                + self._frame_followup_note(
                                    hint="what the screen is actually showing")),
                            "screen": screen,
                        }
                        return result, (label, jpeg, "image/jpeg")
                else:
                    body = json.loads(r.read() or b"{}")
                    if body.get("error") and body.get("ok"):
                        if body["error"] == "simulated":
                            return ({"error": (
                                f"screen {screen} is a headless/simulated display "
                                f"— it has no real pixels to capture. It reports "
                                f"state '{body.get('state') or 'unknown'}' "
                                f"({body.get('detail') or 'no detail'}). Tell the "
                                "user the screen is simulated and read back its "
                                "state — do NOT describe an image or say the "
                                "screen is offline/stuck.")}, None)
                        # uncapturable-iframe isn't final: a GEV iframe's own
                        # remote client may still post a canvas capture for
                        # this token — keep polling. image-load-failed is.
                        if body["error"] != "image-load-failed":
                            last_err = body
                            await asyncio.sleep(0.8)
                            continue
                        detail = body.get("detail")
                        msg = (f"screen {screen} could not capture: {body['error']} "
                               f"(state={body.get('state') or 'unknown'})."
                               + (f" Failed URL: {detail}." if detail else "")
                               + (" The casted image URL failed to load — tell the "
                                  "user the screen shows a broken image and re-cast "
                                  "with the cast_url from ada_camera_snapshot."
                                  if body["error"] == "image-load-failed" else
                                  " If it is an uncapturable iframe, tell the user "
                                  "the page content cannot be screenshotted."))
                        return ({"error": msg}, None)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    logger.warning("session=%s vcast frame poll: %s", self.session_id, exc)
            except Exception as exc:
                logger.warning("session=%s vcast frame poll: %s", self.session_id, exc)
            await asyncio.sleep(0.8)
        if last_err:
            detail = last_err.get("detail")
            msg = (f"screen {screen} could not capture: {last_err['error']} "
                   f"(state={last_err.get('state') or 'unknown'})."
                   + (f" Failed URL: {detail}." if detail else "")
                   + " The framed page cannot be screenshotted by the browser.")
            return ({"error": msg}, None)
        return ({"error": f"screen {screen} did not return a frame in time — "
                          "it may be offline or stuck."}, None)

    async def _run_decision_check(self, product: str, url: str, mode: str) -> None:
        filler = asyncio.create_task(self._check_slow_filler(mode))
        result = None
        error = "unknown"
        try:
            result = await self.tool_runner.decision_engine.check(
                text=product, url=url, mode=mode, caller="voice",
            )
        except Exception as exc:
            error = str(exc)
            logger.warning("session=%s decision check failed: %s", self.session_id, exc)
        finally:
            filler.cancel()
        await self._on_check_complete(result, None if result and result.ok else error)

    async def _check_slow_filler(self, mode: str) -> None:
        # If the check runs long, keep the user informed rather than going quiet.
        await asyncio.sleep(20 if mode == "quick" else 45)
        await self._wait_for_idle(timeout=4.0)
        if self._response_active:
            return
        try:
            await self.send_text_turn(
                "(system) The purchase check is still running — it searches "
                "the live web. Briefly tell the user you're still verifying it."
            )
        except Exception:
            pass

    async def _session_context_tail(self, excluded: set[str] | None = None) -> str:
        """Session-start awareness blob: today's agenda + open tasks +
        pending action proposals from earlier conversations."""
        if os.environ.get("ADA_AGENDA_CONTEXT", "1") in ("0", "false", "no"):
            return ""
        if excluded and CALENDAR_TOOLS <= excluded:
            return ""
        svc = None
        if self.tool_runner is not None:
            try:
                svc = self.tool_runner.calendar
            except Exception:
                svc = None
        if svc is None:
            return ""
        parts: list[str] = []
        try:
            # Google calendar+tasks round-trips from remote hosts routinely
            # exceed 4s on cold token refresh; TimeoutError logs an empty
            # message, which made this failure silent until now.
            plan = await asyncio.wait_for(svc.plan_day("today"), timeout=12.0)
        except Exception as exc:
            logger.info("session=%s agenda prefetch failed: %r",
                        self.session_id, exc)
            # google-token-loud-fail: a dead provider must not read as a
            # quiet "no events today" — tell the model the agenda is
            # unknown so it can say so instead of guessing.
            parts.append(
                "Today's agenda: unavailable — the calendar provider is "
                f"unreachable ({exc}). If the user asks about their "
                "schedule, say the calendar can't be reached right now "
                "(re-authentication if the error mentions auth/token), "
                "never claim the day is empty.")
        else:
            lines = ["Today's agenda (snapshot from session start; call "
                     "plan_day for a fresh view):"]
            events = plan.get("events") or []
            if not events:
                lines.append("- no events today")
            for e in events[:6]:
                start = str(e.get("start") or "")
                when = ("all-day" if e.get("all_day")
                        else start[11:16] if "T" in start else start)
                lines.append(f"- {when} {e.get('title') or '?'}")
            for t in (plan.get("open_tasks") or [])[:8]:
                due = f" (due {t['due']})" if t.get("due") else ""
                lines.append(f"- open task: {t.get('title') or '?'}{due}")
            parts.append("\n".join(lines))
        try:
            from backend.conversation_memory import pending_action_proposals
            proposals = await asyncio.wait_for(
                pending_action_proposals(), timeout=4.0
            )
        except Exception as exc:
            logger.info("session=%s proposals prefetch failed: %r",
                        self.session_id, exc)
            proposals = []
        if proposals:
            lines = ["Pending suggestions from recent conversations:"]
            for p in proposals:
                lines.append(f"- [{p['key']}] {p['text']}")
            parts.append("\n".join(lines))
        if not parts:
            return ""
        return "\n\n" + "\n".join(parts)

    async def _on_check_complete(self, result: Any, error: str | None) -> None:
        """Push the verdict back into the session so Ada speaks it."""
        await self._wait_for_idle()
        if result is None or error:
            prompt = (
                f"(system) The purchase check failed ({error}). Briefly tell the "
                "user the check couldn't complete and suggest the Check panel "
                "in the chat app as an alternative."
            )
        else:
            bits = [f"verdict: {result.verdict} (confidence {result.confidence:.0%})"]
            if result.summary:
                bits.append(result.summary)
            if result.flags:
                bits.append("flags: " + "; ".join(result.flags[:3]))
            if result.alternatives:
                alt = result.alternatives[0]
                bits.append(
                    f"top alternative: {alt.get('name', '?')} — "
                    f"{alt.get('source', '')} {alt.get('price', '')}".strip()
                )
            prompt = (
                "(system) The purchase check finished. " + ". ".join(bits) +
                ". Tell the user the verdict and the most important reason in "
                "one or two sentences, and mention the full detail card is in "
                "the chat panel."
            )
        try:
            await self.send_text_turn(prompt)
        except Exception as exc:
            logger.warning("session=%s check send_text_turn failed: %s", self.session_id, exc)

    @staticmethod
    def _modality_name(modality: Any) -> str:
        return str(getattr(modality, "value", modality) or "unknown").lower()

    def _record_usage(self, usage: Any) -> None:
        in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tokens = int(
            getattr(usage, "response_token_count", None)
            or getattr(usage, "candidates_token_count", 0)
            or 0
        )
        self.usage_input_tokens += in_tokens
        self.usage_output_tokens += out_tokens
        in_mod: dict[str, int] = {}
        out_mod: dict[str, int] = {}
        for detail in getattr(usage, "prompt_tokens_details", None) or []:
            modality = self._modality_name(getattr(detail, "modality", None))
            n = int(getattr(detail, "token_count", 0) or 0)
            in_mod[modality] = in_mod.get(modality, 0) + n
            self.usage_input_by_modality[modality] = (
                self.usage_input_by_modality.get(modality, 0) + n
            )
        for detail in (
            getattr(usage, "response_tokens_details", None)
            or getattr(usage, "candidates_tokens_details", None)
            or []
        ):
            modality = self._modality_name(getattr(detail, "modality", None))
            n = int(getattr(detail, "token_count", 0) or 0)
            out_mod[modality] = out_mod.get(modality, 0) + n
            self.usage_output_by_modality[modality] = (
                self.usage_output_by_modality.get(modality, 0) + n
            )
        usage_ledger.record(
            "live", session_id=self.session_id,
            input_tokens=in_tokens, output_tokens=out_tokens,
            input_by_modality=in_mod, output_by_modality=out_mod,
            cached_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
            tool_use_tokens=int(getattr(usage, "tool_use_prompt_token_count", 0) or 0),
        )
        # Last-turn snapshot for the 'usage' ProviderEvent — scenarios
        # assert token shape (context ceiling) without log scraping.
        self._last_usage_turn = {
            "in": in_tokens, "out": out_tokens,
            "total_in": self.usage_input_tokens,
        }
        logger.info(
            "session=%s usage turn in=%d out=%d | total in=%d out=%d in_by_modality=%s out_by_modality=%s",
            self.session_id, in_tokens, out_tokens,
            self.usage_input_tokens, self.usage_output_tokens,
            self.usage_input_by_modality, self.usage_output_by_modality,
        )
        # Context budget: a saturated window degrades into role-tag
        # continuation (card ada-context-budget). Rotate the live session
        # once, when the turn settles — not mid-generation.
        if in_tokens > self._context_turn_budget and not self._rotate_scheduled:
            self._rotate_scheduled = True
            self._emit_ops_event(
                "context_rotate",
                f"turn input {in_tokens} exceeded budget "
                f"{self._context_turn_budget} — rotating session when idle",
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(self._context_rotate_when_idle())
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

    def usage_summary(self) -> dict[str, Any]:
        return {
            "input_tokens": self.usage_input_tokens,
            "output_tokens": self.usage_output_tokens,
            "input_by_modality": dict(self.usage_input_by_modality),
            "output_by_modality": dict(self.usage_output_by_modality),
        }

    async def _session_config(self, resumption_handle: str | None = None) -> dict:
        """Session-config builder, extracted from connect() (card
        ada-text-mode-sessions): the text provider harvests the identical
        lane-filtered function_declarations + system_instruction from a
        throwaway instance WITHOUT opening a Live session. Side effects
        on self (_excluded_tools, _tool_leak_re) are intentional — the
        text provider reuses them for its dispatch guards.

        Also dumps the final lane-filtered declarations to
        ~/.local/share/ada/tool-decls-<instance>-<channel>.json so ops
        can diff what a lane actually exposes.
        """
        config = {
            # Text-channel sessions (telegram/line relays) also run AUDIO —
            # native-audio live models (3.x) dropped response_modalities TEXT
            # (Google retired gemini-live-2.5-flash-preview 2026-10). Reply
            # text arrives via output_audio_transcription instead.
            "response_modalities": ["AUDIO"],
            "media_resolution": (
                types.MediaResolution.MEDIA_RESOLUTION_HIGH
                if self.video_resolution == "high"
                else types.MediaResolution.MEDIA_RESOLUTION_LOW
            ),
            # Per-session clock injection — the model has no intrinsic 'now',
            # so time-sensitive tool calls (calendar 'in 2 hours', 'Friday')
            # anchor to the user's local time at session start.
            "system_instruction": self.instructions + _now_context(),
            "input_audio_transcription": {
                # Restrict ASR to Thai/English — unhinted input
                # transcription wandered into Korean/Chinese/Portuguese
                # on noisy Thai speech (transcript 2026-09-29 855a65dab8).
                "language_codes": ["th-TH", "en-US"],
            },
            "output_audio_transcription": {},
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": self.voice},
                }
            },
            "realtime_input_config": {
                # Be explicit about barge-in and favor detecting near-end speech
                # over the assistant audio playing through the iPad/iPhone speakers.
                "activity_handling": types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                "automatic_activity_detection": {
                    "disabled": False,
                    # Speaker echo can otherwise look like a new user turn and
                    # make Ada interrupt herself. LOW still supports barge-in,
                    # but requires stronger evidence that speech has started.
                    "start_of_speech_sensitivity": (
                        types.StartSensitivity.START_SENSITIVITY_LOW
                    ),
                    "end_of_speech_sensitivity": (
                        types.EndSensitivity.END_SENSITIVITY_HIGH
                    ),
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 500,
                },
            },
            # Native audio consumes context quickly. Sliding-window compression
            # prevents an extended voice conversation from exhausting the
            # session context while retaining recent conversational history.
            # trigger/target explicit (card ada-context-budget): unset defaults
            # let session d6cbf6e6e7 ride to ~78k input tokens/turn before any
            # compression — well past the point where output degenerates.
            # Keep the pair GENTLE: the static baseline (instructions + tool
            # decls) is ~19k tokens, so a trigger near it compresses almost
            # every turn — observed 2026-10-07: trigger=24k/target=8k left the
            # model unmoored mid-tool-sequence and it degenerated into
            # tool-call storms. 32k/24k only fires once real history builds.
            "context_window_compression": types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(
                    target_tokens=int(os.environ.get(
                        "ADA_CONTEXT_TARGET_TOKENS", "24000")),
                ),
                trigger_tokens=int(os.environ.get(
                    "ADA_CONTEXT_TRIGGER_TOKENS", "32000")),
            ),
            "session_resumption": types.SessionResumptionConfig(
                handle=resumption_handle,
            ),
            "tools": [{"function_declarations": _builtin_declarations()}],
        }
        if self.text_only:
            # No audio in or out on relay channels — speech/voice config is
            # meaningless. Keep output_audio_transcription: it is now the
            # text source (native-audio models dropped TEXT modality).
            for k in ("speech_config",
                      "input_audio_transcription", "realtime_input_config"):
                config.pop(k, None)
            config["tools"][0]["function_declarations"] = [
                fd for fd in config["tools"][0]["function_declarations"]
                if fd.get("name") != "set_facial_expression"
            ]
        if os.environ.get("DISABLE_GET_HOME_STATE") == "true":
            config["tools"][0]["function_declarations"] = [
                fd for fd in config["tools"][0]["function_declarations"]
                if fd.get("name") != "get_home_state"
            ]
            config["system_instruction"] = (
                config["system_instruction"]
                .replace(
                    "get_home_state to check occupancy and the state of the configured home plugs "
                    "(or a single entity with entity_id=, one HA domain with domain=, or the "
                    "stored memory snapshot with domain='memory'), ",
                    "",
                )
                .replace(
                    "When the user asks about devices, occupancy, or what is on/off, call get_home_state or home_search first. ",
                    "When the user asks about devices, occupancy, or what is on/off, call home_search first. ",
                )
            )
        excluded = {s.strip() for s in os.environ.get("ADA_EXCLUDED_TOOLS", "").split(",") if s.strip()}
        # Allowlist mode (OPS-0476): when ADA_INCLUDED_TOOLS names a keep-
        # list, everything NOT named is excluded — merged/renamed tools can
        # no longer slip onto a restricted lane (the home_search class).
        # Folded into `excluded` so the instruction-strip logic below
        # behaves identically for both modes.
        included = {s.strip() for s in os.environ.get("ADA_INCLUDED_TOOLS", "").split(",") if s.strip()}
        if included:
            # Cover the late merges below as well (camera decl, tools.d,
            # chaba): they are appended AFTER this strip point, so a keep-
            # list that only filters the builtin decls leaks them onto
            # restricted lanes.
            late = {d.get("name") for d in CHABA_DECLARATIONS}
            late.add(CAMERA_DECLARATION["name"])
            try:
                late |= {d.get("name") for d in
                         tools_loader.registry().declarations(set())}
            except Exception:
                pass
            excluded |= ({fd.get("name") for fd in
                          config["tools"][0]["function_declarations"]}
                         | late) - included
        self._excluded_tools = set(excluded)
        if excluded:
            config["tools"][0]["function_declarations"] = [
                fd for fd in config["tools"][0]["function_declarations"]
                if fd.get("name") not in excluded
            ]
            # Keep instructions consistent: when every calendar tool is
            # excluded, drop the paragraph describing them too.
            if CALENDAR_TOOLS <= excluded:
                config["system_instruction"] = config["system_instruction"].replace(
                    CALENDAR_INSTRUCTIONS, ""
                )
            if CMS_TOOLS <= excluded:
                config["system_instruction"] = config["system_instruction"].replace(
                    CMS_INSTRUCTIONS, ""
                )
            if DEVIN_TOOLS <= excluded:
                config["system_instruction"] = config["system_instruction"].replace(
                    DEVIN_INSTRUCTIONS, ""
                )
            if DOC_TOOLS <= excluded:
                config["system_instruction"] = config["system_instruction"].replace(
                    DOC_INSTRUCTIONS, ""
                )
            if DRIVE_TOOLS <= excluded:
                config["system_instruction"] = config["system_instruction"].replace(
                    DRIVE_INSTRUCTIONS, ""
                )
            if HABIT_TOOLS <= excluded:
                config["system_instruction"] = config["system_instruction"].replace(
                    HABIT_INSTRUCTIONS, ""
                )
        # The merged camera tool is declared on every instance — its
        # source='traffic' path needs no VMS shim (tools-merge-camera).
        # The vms source just answers honestly when the shim is absent.
        if not (VMS_TOOLS <= excluded):
            config["tools"][0]["function_declarations"].append(dict(CAMERA_DECLARATION))
        # Drop-in tools (backend/tools.d) — manifest-declared, excluded-aware.
        try:
            config["tools"][0]["function_declarations"].extend(
                tools_loader.registry().declarations(excluded))
        except Exception as exc:
            logger.warning("tools.d declarations failed: %s", exc)
        if chaba_memory.enabled():
            # Guest mode: allowlist the tool surface, append chaba guest tools,
            # and inject the rendered guest context instead of MDDB priming.
            # CHABA_DECLARATIONS re-declares the canonical memory names
            # guest-scoped — drop the bank-facing versions so no function
            # name is declared twice.
            chaba_declared = {d["name"] for d in CHABA_DECLARATIONS}
            config["tools"][0]["function_declarations"] = [
                fd for fd in config["tools"][0]["function_declarations"]
                if fd.get("name") in CHABA_ALLOW
                and fd.get("name") not in chaba_declared
            ]
            config["tools"][0]["function_declarations"].extend(CHABA_DECLARATIONS)
            config["system_instruction"] += CHABA_INSTRUCTIONS
            try:
                ctx = chaba_memory.get_store().system_context(
                    getattr(self, "session_id", None)
                )
                if ctx:
                    config["system_instruction"] += "\n\n" + ctx
            except Exception as exc:
                logger.warning("chaba context load failed: %s", exc)
        else:
            config["system_instruction"] += await self._session_context_tail(excluded)
            # Per-instance memory banks: descriptions name only banks this
            # instance can actually use ({banks}=readable, {writable_banks}=
            # writable), so the model never calls a bank that fails loudly.
            all_banks, writable_banks = self._memory_bank_names()
            config["tools"][0]["function_declarations"] = _fill_bank_placeholders(
                config["tools"][0]["function_declarations"], all_banks, writable_banks
            )
        # Final keep-list pass: everything appended after the first strip
        # point (camera, tools.d, chaba guest decls) is still subject to
        # ADA_INCLUDED_TOOLS — the declaration surface is the contract.
        if included:
            config["tools"][0]["function_declarations"] = [
                fd for fd in config["tools"][0]["function_declarations"]
                if fd.get("name") in included
            ]
        # Lazy schemas last — after exclusions, chaba allowlist, and bank
        # placeholders — so every wire decl (builtin, camera, tools.d,
        # chaba) is covered by the same rule.
        config["tools"][0]["function_declarations"] = _apply_lazy_schemas(
            config["tools"][0]["function_declarations"])
        # NOTE: live-session google_search grounding is NOT enabled — the
        # API key's plan rejects it with a 1011 quota error at connect,
        # taking down every session. Web search goes through the explicit
        # web_search tool (generate-API grounding) instead.
        # Build the transcript leak filter from the declared tool names —
        # the model occasionally SPEAKS 'set_facial_expression{...}' style
        # text instead of emitting a real function call, and the output
        # transcription relays it verbatim into transcripts/UI.
        try:
            names = [
                fd.get("name", "")
                for fd in config["tools"][0]["function_declarations"]
            ]
            names = [re.escape(n) for n in names if n]
            if names:
                self._tool_leak_re = re.compile(
                    r"\b(?:" + "|".join(names) + r")\s*\{"
                    # echo of the tool-result wrapper — observed
                    # 2026-09-30: '<code_output>Tool cctv_snapshot
                    # returned: {...}' spoken aloud mid-turn
                    r"|<code_output>|\bTool\s+\w+\s+returned\s*:",
                )
            else:
                self._tool_leak_re = None
        except Exception:
            self._tool_leak_re = None
        # Lane declaration cache — diagnostics + parity record for the
        # text provider (ada-text-mode-sessions).
        try:
            _decl_dir = Path(os.environ.get(
                "ADA_TOOL_DECL_DIR", str(Path.home() / ".local/share/ada")))
            _decl_dir.mkdir(parents=True, exist_ok=True)
            (_decl_dir / f"tool-decls-{ada_instance_id()}-{self.channel or 'voice'}.json").write_text(
                json.dumps(config["tools"][0]["function_declarations"],
                           indent=1, default=str))
        except Exception as exc:
            logger.warning("tool-decl cache write failed: %s", exc)
        return config

    async def connect(self, resumption_handle: str | None = None) -> None:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        # Per-lane spend enforcement (card ada-gemini-live-spend-cap) —
        # before any session setup: a latched provider billing cap or an
        # exhausted lane budget fails fast and free here instead of
        # paying Google for a doomed connect on every retry.
        lane_budget.check(api_key=self.api_key, emit=self._emit_ops_event)

        self._client = genai.Client(api_key=self.api_key)
        self._client = genai.Client(api_key=self.api_key)
        config = await self._session_config(resumption_handle)
        self._session_context = self._client.aio.live.connect(
            model=self.model,
            config=config,
        )
        try:
            self._session = await self._session_context.__aenter__()
        except Exception as exc:
            # A billing-class provider error (monthly spend cap, prepaid
            # depleted) latches the key fingerprint for the rest of the
            # billing month — the next connect fails locally.
            lane_budget.note_connect_failure(
                exc, api_key=self.api_key, emit=self._emit_ops_event)
            raise
        logger.info(
            "session=%s Gemini Live session connected (model=%s, voice=%s, video=%s)",
            self.session_id, self.model,
            self.voice,
            self.video_resolution,
        )

    async def send_audio(self, pcm16: bytes) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm16, mime_type="audio/pcm;rate=16000")
            )

    async def send_video(self, jpeg: bytes) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(
                video=types.Blob(data=jpeg, mime_type="image/jpeg")
            )

    def is_stalled(self) -> bool:
        """True when the user has spoken but the provider has been silent
        past ADA_STALL_TIMEOUT_S — a dead Gemini Live stream that never
        ends cleanly and must be force-reconnected."""
        if self._closed or not self._last_user_at or self._tools_in_flight:
            return False
        return (
            self._last_user_at > self._last_model_at
            and time.monotonic() - self._last_user_at > self._stall_timeout
        )

    def upstream_close_code(self):
        """Ws close code once the Gemini link has closed — None while open."""
        sess = getattr(self, "_session", None)
        ws = getattr(sess, "_ws", None) if sess is not None else None
        return getattr(ws, "close_code", None) if ws is not None else None

    def upstream_dead(self) -> bool:
        """True when the upstream ws has closed while this provider still
        looks attached — the silent-death shape is_stalled can't see (it
        only arms after a user turn, and the notify path just logs into
        the corpse). The session watchdog uses this to force teardown."""
        if self._closed:
            return True
        ws = getattr(getattr(self, "_session", None), "_ws", None)
        if ws is None:
            return False
        if getattr(ws, "close_code", None) is not None:
            return True
        return getattr(getattr(ws, "state", None), "name", "") in (
            "CLOSING", "CLOSED")

    async def send_text_turn(self, text: str) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        self._last_user_at = time.monotonic()
        # A real turn is now open through turn_complete — text turns produce
        # no input_transcription, so this is the only signal covering the
        # tool-call/generation gap where context sends get dropped.
        self._turn_open = True
        async with self._send_lock:
            await self._session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                ),
                turn_complete=True,
            )

    async def send_image_turn(self, image: bytes, mime: str,
                              caption: str = "") -> None:
        """A user-sent image turn (chat relays: LINE/Telegram photo in).
        Same shape as the camera-frame injection — text + image part,
        turn_complete so the model answers with a description."""
        if self._session is None:
            raise RuntimeError("provider is not connected")
        self._last_user_at = time.monotonic()
        self._turn_open = True
        parts = [types.Part.from_text(
            text=caption or "The user sent you this image — describe "
                             "it and respond naturally.")]
        parts.append(types.Part.from_bytes(
            data=image, mime_type=mime or "image/jpeg"))
        async with self._send_lock:
            await self._session.send_client_content(
                turns=types.Content(role="user", parts=parts),
                turn_complete=True,
            )

    async def notify_or_defer(self, text: str, urgent: bool = False) -> str:
        """Notification handling (Eisenhower split):

        urgent=True   — interrupt now: injected as a user turn so Ada stops
                        and announces it immediately (urgent + important).
        urgent=False  — the client already announced it via machine voice
                        (speechSynthesis); we inject a silent context note
                        (turn_complete=False) so Ada knows it arrived and
                        can circle back, but she does NOT stop her current
                        task or speak about it now. Queued to the next turn
                        boundary if a response is in flight.

        Returns 'interrupted' | 'context' | 'queued'."""
        if urgent:
            await self.send_text_turn(
                "(system) URGENT notification — stop and tell the user now: "
                + text)
            return "interrupted"
        framed = (
            "(system) Note for later — a notification arrived and the user "
            f"was already informed by a short machine-voice announcement: "
            f"{text}\nDo not speak about this now. Finish your current "
            "task; bring it up only when the current work is done or the "
            "user asks.")
        # Always queue: a mid-turn context send can be dropped by the API
        # even before audio starts (tool-call phase). If no turn is open,
        # a short timer flushes; otherwise the next turn boundary drains.
        self._pending_notifications.append(framed)
        if not self._flush_scheduled:
            self._flush_scheduled = True
            try:
                asyncio.get_running_loop().create_task(self._flush_soon())
            except RuntimeError:
                self._flush_scheduled = False
        return "queued"

    async def _flush_soon(self) -> None:
        """Idle-path drain: if no turn is open ~2s after a note queued,
        deliver it as context now — no turn boundary is coming. If a turn
        is open, leave it for the boundary drain (or a later flush)."""
        try:
            for _ in range(15):  # bounded — ~30s max wait for idle
                await asyncio.sleep(2)
                if not self._pending_notifications:
                    return
                if not self._response_active and not self._turn_open:
                    await self._drain_notifications()
                    return
        finally:
            self._flush_scheduled = False

    def queue_context_note(self, text: str) -> None:
        """Queue a silent context note for the next safe delivery point —
        same deferred path as non-urgent notifies (turn_complete=False, no
        reply triggered). Used for cross-channel mirror notes that must
        never interrupt or be answered."""
        self._pending_notifications.append(text)
        if not self._flush_scheduled:
            self._flush_scheduled = True
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No loop (unit-test/sync caller) — leave the note queued.
                self._flush_scheduled = False
                return
            loop.create_task(self._flush_soon())

    async def _send_context_note(self, text: str) -> None:
        """Append context WITHOUT triggering a response (turn_complete=False)
        — the note lands in session state for later reference only."""
        async with self._send_lock:
            await self._session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                ),
                turn_complete=False,
            )

    async def _drain_notifications(self) -> None:
        while self._pending_notifications:
            framed = self._pending_notifications.pop(0)
            try:
                await self._send_context_note(framed)
            except Exception as exc:
                logger.warning(
                    "session=%s deferred notify send failed: %s",
                    self.session_id, exc)
                return

    async def send_habit_alert(self, jpeg: bytes, alert: str) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_client_content(
                turns=types.Content(role="user", parts=[
                    types.Part.from_text(text=alert),
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                ]),
                turn_complete=True,
            )

    async def events(self) -> AsyncIterator[ProviderEvent]:
        if self._session is None:
            raise RuntimeError("provider is not connected")

        self._response_active = False
        input_transcript = ""
        self._live_input_transcript = ""
        assistant_turn_text = ""
        response_started_at = 0.0
        response_audio_chunks = 0
        response_audio_bytes = 0
        tool_calls_this_turn = 0
        # Results that came back ok:false this turn (gate denials,
        # tool errors) — feeds the phantom_write_claim honesty check so
        # "saved" narrated over a NOT-EXECUTED result surfaces as an
        # ops event, not just zero-tool-call phantoms.
        failed_results_this_turn = 0
        # ok results from calls that can back a write/done claim — a
        # passed READ must not whitewash a claim (session 67b02417a8:
        # narrated cards over zero write calls went unflagged).
        backing_calls_this_turn = 0
        # (name, normalized result) pairs this turn — the dead-turn
        # guard's retry nudge and synthesized fallback read the answer
        # back out of these when the model emits nothing speakable.
        turn_tool_results: list[tuple[str, dict]] = []
        # (name, args, result) for this turn's fidelity-watched writes —
        # the write_fidelity check compares the persisted content arg
        # against the narration/user seed (card ada-cms-write-fidelity).
        turn_fidelity_calls: list[tuple[str, dict, dict]] = []
        tool_budget = int(os.environ.get("ADA_TOOL_CALL_BUDGET", "20"))
        actuations_this_turn = 0
        actuation_budget = int(os.environ.get("ADA_ACTUATION_BUDGET", "6"))
        # Cosmetic-call budgets (OPS-0509): set_facial_expression gets its
        # own caps — the generic 20/turn budget only bites once a storm is
        # already expensive, and the instructed once-per-reply cadence
        # never trips it at all. 0 disables the tool outright.
        expr_calls_this_turn = 0
        expr_turn_budget = int(os.environ.get(
            "ADA_EXPRESSION_TURN_BUDGET", "2"))
        expr_minute_budget = int(os.environ.get(
            "ADA_EXPRESSION_MINUTE_BUDGET", "4"))
        budget_hit = False
        barge_pending = False  # a barge-in's transcript arrives next turn_complete
        input_done_ts = 0.0  # last input_transcription chunk ≈ end of user speech
        n_tools_this_turn = 0
        budget_nudged = False
        # A turn_complete that lands right after function calls were
        # dispatched is a segment boundary, not the answer — results go back
        # and generation continues. The dead-turn guard must not judge it.
        segment_tool_pending = False

        while not self._closed:
            async for message in self._session.receive():
                self._last_model_at = time.monotonic()
                usage = message.usage_metadata
                if usage:
                    self._record_usage(usage)
                    yield ProviderEvent(
                        "usage", dict(self._last_usage_turn))
                update = message.session_resumption_update
                if update and update.resumable and update.new_handle:
                    self.resumption_handle = update.new_handle
                if message.go_away:
                    self.go_away_time_left = str(message.go_away.time_left or "")
                    yield ProviderEvent("go_away", {"time_left": self.go_away_time_left})
                    return
                tool_call = message.tool_call
                if tool_call and tool_call.function_calls:
                    segment_tool_pending = True
                    function_responses = []
                    camera_frames: list[tuple[str, bytes, str]] = []
                    for call in tool_call.function_calls:
                        logger.info(
                            "session=%s function_call received id=%s name=%s args=%r",
                            self.session_id, call.id, call.name, call.args,
                        )
                        self._write_call_log({
                            "event": "function_call",
                            "id": call.id, "name": call.name,
                            "args": _safe_args(call.args)})
                        # Consolidated tool surface: absorbed names
                        # (tool_runner._ALIASES) resolve to their canonical
                        # tool BEFORE dispatch — provider-dispatched tools
                        # share the runner's alias contract so stale
                        # phrasing still lands (tools-merge-camera).
                        # The FunctionResponse below must echo the AS-CALLED
                        # name, not the canonical — Gemini correlates
                        # responses to calls by name+id.
                        _call_name = str(call.name or "")
                        _rname, _rimplied = _resolve_alias(_call_name)
                        if _rname != _call_name or _rimplied:
                            try:
                                _merged = {**_rimplied,
                                           **dict(call.args or {})}
                                # Surrogate-arg remap rides along so the
                                # ws path honors the same contract as
                                # runner.execute (tools-merge-meta-voice:
                                # ada_set_voice's action=set|show|list must
                                # land on persona's *_voice actions).
                                _merged = _alias_call_args(
                                    _call_name, _merged)
                                call = call.model_copy(update={
                                    "name": _rname,
                                    "args": _merged})
                            except Exception:
                                logger.warning(
                                    "session=%s alias copy failed for %s",
                                    self.session_id, call.name)
                        yield ProviderEvent("tool_call", {
                            "id": str(call.id or ""),
                            "name": str(call.name),
                            "args": _safe_args(call.args),
                        })
                        requested = (call.args or {}).get("expression")
                        tool_calls_this_turn += 1
                        self._tools_in_flight += 1
                        tool_t0 = time.monotonic()
                        # cctv_snapshot's old seat: a snap-to-display push
                        # is actuation even though a bare describe is not.
                        _cargs_probe = call.args or {}
                        _display_push = (
                            _cargs_probe.get("screen")
                            or str(_cargs_probe.get("target") or "")
                            .strip().lower() in ("tv", "screen"))
                        # cast_to_screen's absorbed read seats (list/status —
                        # vcast_list/vcast_status were not actuating) don't
                        # consume the actuation budget; 'say' keeps
                        # vcast_say's actuating seat (tools-merge-display).
                        _cast_read = (
                            call.name == "cast_to_screen"
                            and str(_cargs_probe.get("action") or "")
                            .strip().lower() in ("list", "status"))
                        # yt's old seats: cast/stop actuate the TV —
                        # status/transcript are reads and stay free.
                        _yt_actuates = (
                            call.name == "yt" and str(
                                _cargs_probe.get("action") or "")
                            .strip().lower() in ("cast", "stop"))
                        if (call.name in ACTUATING_TOOLS
                                and not _cast_read) or (
                                call.name == "ada_camera_snapshot"
                                and _display_push) or (
                                # docs absorbed ada_doc_archive/print —
                                # only the write actions actuate.
                                call.name == "docs" and str(
                                    _cargs_probe.get("action") or "")
                                .strip().lower() in ("archive", "print")) or (
                                # ada_set_voice's old seat: persona's
                                # set_voice drops the session to reconnect.
                                call.name == "ada_persona"
                                and str(_cargs_probe.get("action") or "")
                                .strip().lower() == "set_voice") or _yt_actuates:
                            actuations_this_turn += 1
                        if tool_calls_this_turn > tool_budget:
                            if not budget_hit:
                                self._emit_ops_event(
                                    "tool_storm",
                                    f"Turn exceeded the tool-call budget "
                                    f"({tool_budget}) — further calls refused.",
                                    tool=str(call.name))
                            budget_hit = True
                            logger.warning(
                                "session=%s per-turn tool-call budget %d exhausted — refusing %s",
                                self.session_id, tool_budget, call.name,
                            )
                            result = {"error": (
                                "Tool budget for this turn exhausted — stop calling tools "
                                "and answer the user from what you already have.")}
                        elif actuations_this_turn > actuation_budget:
                            if actuations_this_turn == actuation_budget + 1:
                                self._emit_ops_event(
                                    "actuation_cap",
                                    f"Turn exceeded the actuation budget "
                                    f"({actuation_budget}) — refused "
                                    f"{call.name}.",
                                    tool=str(call.name))
                            budget_hit = True
                            logger.warning(
                                "session=%s per-turn actuation budget %d exhausted — refusing %s",
                                self.session_id, actuation_budget, call.name,
                            )
                            result = {"error": (
                                "Actuation limit for this turn reached — stop and tell the "
                                "user what you were trying to control instead of retrying.")}
                        elif self._tool_excluded(call.name):
                            # Undeclared call: instructions still describe
                            # stripped tools, so the model can emit them
                            # anyway. Refuse at dispatch — the lane surface
                            # is a hard gate, not just a prompt hint.
                            logger.info(
                                "session=%s refused undeclared tool %s "
                                "(lane surface)", self.session_id, call.name)
                            self.conversation.log_event(
                                "tool_denied", tool=str(call.name),
                                reason="lane_excluded")
                            result = {"error": (
                                f"{call.name} is not available on this "
                                "lane — answer with the tools you have.")}
                        elif call.name == "set_facial_expression" and requested in EXPRESSION_NAMES:
                            # Token diet (OPS-0509 /
                            # expression-call-batching): every expression
                            # call is a billed model round — the over-cap
                            # call itself already cost the resend, so the
                            # refusal's job is stopping the NEXT one.
                            # An ok result carries the steer: ok:false
                            # would count in failed_results_this_turn and
                            # taint the phantom-claim gate for a cosmetic
                            # refusal, while a hard "invalid" error invites
                            # a retry (another billed round).
                            expr_calls_this_turn += 1
                            _expr_now = time.monotonic()
                            self._expr_applied_at = [
                                t for t in self._expr_applied_at
                                if _expr_now - t < 60.0]
                            if (expr_calls_this_turn > expr_turn_budget
                                    or len(self._expr_applied_at)
                                    >= expr_minute_budget):
                                if not self._expr_throttle_noted:
                                    self._expr_throttle_noted = True
                                    self._emit_ops_event(
                                        "expression_throttled",
                                        "set_facial_expression refused by "
                                        "the cosmetic-call budget "
                                        f"({expr_turn_budget}/turn, "
                                        f"{expr_minute_budget}/min) — "
                                        "OPS-0509 token diet.",
                                        tool="set_facial_expression")
                                result = {"output": (
                                    "Expression unchanged — facial-expression "
                                    "updates are capped for now; do not call "
                                    "set_facial_expression again unless the "
                                    "tone clearly demands it.")}
                            else:
                                self._expr_applied_at.append(_expr_now)
                                yield ProviderEvent(
                                    "expression", {"name": requested})
                                result = {"output": (
                                    f"{_ADA_DISPLAY_NAME} is now {requested}")}
                        elif (call.name == "get_home_state"
                              # entity_id=/domain= reads route to the
                              # runner (tools-merge-ha) — the fast path
                              # only serves the no-arg occupancy snapshot.
                              and not (call.args or {}).get("entity_id")
                              and not (call.args or {}).get("domain")
                              and self.home_assistant_client is not None):
                            try:
                                snapshot = await self.home_assistant_client.snapshot(
                                    person_entity=self.current_speaker_ha_person
                                )
                                plugs_on_named = [
                                    {"entity_id": e, "name": snapshot.plug_names.get(e, e)}
                                    for e in snapshot.plugs_on
                                ]
                                all_plugs = {
                                    e: {"name": snapshot.plug_names.get(e, e), "state": state}
                                    for e, state in snapshot.plug_states.items()
                                }
                                result = {
                                    "output": {
                                        "person": snapshot.person_state,
                                        "plugs_on": plugs_on_named,
                                        "all_plugs": all_plugs,
                                    }
                                }
                            except Exception as exc:
                                result = {"error": f"home state failed: {exc}"}
                        elif (call.name == "home_search"
                              # kind='event' needs the recorder/MDDB —
                              # it falls through to the runner.
                              and str((call.args or {}).get("kind")
                                      or "device").lower() in ("device", "sensor")
                              and self.home_assistant_client is not None):
                            try:
                                args = dict(call.args or {})
                                query = str(args.get("query") or args.get("q") or "")
                                kind = str(args.get("kind") or "device").lower()
                                limit = int(args.get("limit") or 10)
                                if kind == "sensor":
                                    if query:
                                        sensors = await self.home_assistant_client.sensors(
                                            search=query, limit=limit)
                                    else:
                                        sensors = await self.home_assistant_client.sensors(limit=50)
                                    result = {"output": sensors}
                                elif query:
                                    devices = await self.home_assistant_client.search_entities(
                                        query, limit=limit)
                                    result = {"output": devices}
                                else:
                                    devices = await self.home_assistant_client.entities()
                                    # Unbounded dumps stay in live context for
                                    # the whole session (2026-10-01: this single
                                    # call pushed a session past 1M input tokens
                                    # during a silent tool storm). Cap it; the
                                    # marker steers the model to search instead.
                                    if len(devices) > LIST_HOME_DEVICES_MAX:
                                        devices = devices[:LIST_HOME_DEVICES_MAX] + [{
                                            "_truncated": (
                                                f"{LIST_HOME_DEVICES_MAX} of {len(devices)} "
                                                "devices shown — call home_search "
                                                "with a name/keyword for the rest"),
                                        }]
                                    result = {"output": devices}
                            except Exception as exc:
                                result = {"error": f"home_search failed: {exc}"}
                        elif call.name == "home_status":
                            # tools-merge-tasks-status: the seven status
                            # getters collapsed into home_status(what=...).
                            # Alias resolution already rewrote absorbed
                            # names and merged their implied what=/args.
                            _hargs = dict(call.args or {})
                            _hwhat = str(_hargs.get("what") or "").strip().lower()
                            try:
                                if _hwhat == "battery" and self.home_assistant_client is not None:
                                    _hidx = _hargs.get("battery_index")
                                    if _hidx in (None, ""):
                                        status = await self.home_assistant_client.battery_status()
                                    else:
                                        status = await self.home_assistant_client.battery_detail(int(_hidx))
                                    result = {"output": status}
                                elif _hwhat == "power" and self.home_assistant_client is not None:
                                    hours = int(_hargs.get("hours", 24))
                                    result = {"output": await self.home_assistant_client.power_summary(hours=hours)}
                                elif _hwhat == "inverter" and self.home_assistant_client is not None:
                                    result = {"output": await self.home_assistant_client.inverter_status()}
                                elif _hwhat == "pool" and self.home_assistant_client is not None:
                                    result = {"output": await self.home_assistant_client.pool_status()}
                                elif _hwhat == "dashboard" and self.home_assistant_client is not None:
                                    tab = _hargs.get("tab")
                                    if not tab:
                                        result = {"error": "tab is required"}
                                    else:
                                        result = {"output": await self.home_assistant_client.dashboard_tab(str(tab))}
                                elif _hwhat == "habit" and self.habit_state_getter is not None:
                                    result = {"output": self.habit_state_getter()}
                                else:
                                    result = {"error": (
                                        f"invalid what {_hwhat!r}: expected "
                                        "battery|power|inverter|pool|dashboard|habit"
                                        if _hwhat else "what is required")}
                            except Exception as exc:
                                result = {"error": f"home_status({_hwhat or '?'}) failed: {exc}"}
                        elif call.name == "get_rk600_weather" and self.home_assistant_client is not None:
                            try:
                                weather = await self.home_assistant_client.rk600_weather()
                                result = {"output": weather}
                            except Exception as exc:
                                result = {"error": f"get_rk600_weather failed: {exc}"}
                        elif call.name == "get_dashboard_tab" and self.home_assistant_client is not None:
                            try:
                                args = dict(call.args or {})
                                tab = args.get("tab")
                                if not tab:
                                    result = {"error": "tab is required"}
                                else:
                                    info = await self.home_assistant_client.dashboard_tab(str(tab))
                                    result = {"output": info}
                            except Exception as exc:
                                result = {"error": f"get_dashboard_tab failed: {exc}"}
                        elif call.name == "get_power_summary" and self.home_assistant_client is not None:
                            try:
                                args = dict(call.args or {})
                                hours = int(args.get("hours", 24))
                                summary = await self.home_assistant_client.power_summary(hours=hours)
                                result = {"output": summary}
                            except Exception as exc:
                                result = {"error": f"get_power_summary failed: {exc}"}
                        elif (call.name == "home_history"
                              # The sensor-series read keeps its fast
                              # path (the absorbed get_sensor_history) —
                              # every other kind routes to the runner.
                              and str((call.args or {}).get("kind")
                                      or "").lower() in ("", "series")
                              and str((call.args or {}).get("entity_id")
                                      or "").startswith("sensor.")
                              and self.home_assistant_client is not None):
                            try:
                                args = dict(call.args or {})
                                entity_id = args.get("entity_id")
                                hours = int(args.get("hours", 24))
                                history = await self.home_assistant_client.history(
                                    str(entity_id), hours=hours)
                                result = {"output": history}
                            except Exception as exc:
                                result = {"error": f"home_history failed: {exc}"}
                        elif (call.name == "report_habit_observation"
                              or (call.name == "ada_remember"
                                  and str((call.args or {}).get("kind")
                                          or "").lower() == "habit")):
                            # Legacy name stays callable as a hidden alias;
                            # the canonical path is ada_remember kind='habit'.
                            args = dict(call.args or {})
                            yield ProviderEvent("habit_observation", args)
                            result = {"output": "Observation delivered to the habit monitor"}
                        elif (
                            # tools-merge-meta-voice: ada_decision_check ->
                            # ada_ops action='check'; ada_set_voice ->
                            # ada_persona *_voice actions. usage/health/
                            # research keep their old ungated seats.
                            (call.name == "ada_session_recall"
                             or (call.name == "ada_ops"
                                 and str((call.args or {}).get("action")
                                         or "").lower() == "check")
                             or (call.name == "ada_persona"
                                 and str((call.args or {}).get("action")
                                         or "").lower()
                                 in _PERSONA_VOICE_ACTIONS))
                            # scope='history' absorbed ada_ha_recall, which was
                            # never secondary-blocked — keep that access.
                            and not (call.name == "ada_session_recall"
                                     and (call.args or {}).get("scope") == "history")
                            and self.tool_runner is not None
                            and self.tool_runner._is_secondary_turn()
                        ):
                            # Provider-dispatched tools bypass the runner gate —
                            # enforce the secondary-speaker policy here too.
                            self.conversation.log_event(
                                "tool_denied", tool=str(call.name),
                                speaker=self.current_speaker_ha_person)
                            result = {"error": (
                                "Reserved for the session owner — propose it to "
                                "them aloud and let them ask in their own voice.")}
                        elif (call.name == "ada_session_recall"
                              and (call.args or {}).get("scope") != "history"):
                            # scope='history' (absorbed ada_ha_recall) falls
                            # through to the runner for an inline answer.
                            # `force=true` is the user's explicit push
                            # ("dig deeper", "check again") — it overrides
                            # the redundant-recall gate.
                            force = (call.args or {}).get("force")
                            if self._recall_gated() and not force:
                                result = {"output": (
                                    "ada_memory_search already returned a confident match "
                                    "moments ago — answer from those hits. Session recall "
                                    "skipped (redundant). If the user insists, retry with force=true.")}
                            else:
                                question = (call.args or {}).get("question", "What did we discuss in the previous session?")
                                group = (call.args or {}).get("group")
                                bank = (call.args or {}).get("bank")
                                recall_status = self.conversation.start_recall(
                                    str(question), self._on_recall_complete,
                                    group=str(group) if group else None,
                                    bank=str(bank) if bank else None,
                                    on_slow=self._on_recall_slow,
                                )
                                result = {"output": recall_status}
                        elif (call.name == "ada_persona"
                              and str((call.args or {}).get("action")
                                      or "").lower() in _PERSONA_VOICE_ACTIONS):
                            # absorbed ada_set_voice — provider-side so the
                            # idle-gated reconnect can run.
                            result = {"output": self._set_voice(dict(call.args or {}))}
                        elif call.name == "ada_camera_snapshot":
                            _cargs = dict(call.args or {})
                            # cctv_snapshot's absorbed behavior was
                            # capture-gated for secondary voices — the
                            # gate keys off the RESOLVED name and runs
                            # only when the call pushes to a display.
                            _wants_display = (
                                _cargs.get("screen")
                                or str(_cargs.get("target") or "")
                                .strip().lower() in ("tv", "screen"))
                            if _wants_display and self.tool_runner is not None:
                                if _cargs.get("confirmed") and not \
                                        self._user_confirmed(input_transcript):
                                    logger.warning(
                                        "session=%s ada_camera_snapshot "
                                        "self-asserted confirmed=true "
                                        "without user affirmation — "
                                        "stripping", self.session_id)
                                    _cargs.pop("confirmed", None)
                                    self._emit_ops_event(
                                        "confirm_strip",
                                        "Stripped model-asserted "
                                        "confirmed=true on "
                                        "ada_camera_snapshot — no user "
                                        "affirmation found.",
                                        tool="ada_camera_snapshot")
                                try:
                                    self.tool_runner._check_capture_confirmed(
                                        "ada_camera_snapshot", _cargs,
                                        _cargs.get("confirmed"))
                                except PermissionError as exc:
                                    result = {"error": str(exc)}
                                    _cargs = None
                            if _cargs is not None:
                                result, frame = await self._camera_snapshot(
                                    _cargs)
                                if frame:
                                    camera_frames.append(frame)
                        elif call.name == "vcast_snapshot":
                            result, frame = await self._vcast_snapshot(
                                dict(call.args or {}))
                            if frame:
                                camera_frames.append(frame)
                        elif (call.name == "ada_ops"
                              and str((call.args or {}).get("action")
                                      or "").lower() == "check"
                              and self.tool_runner is not None):
                            # absorbed ada_decision_check — provider-side so
                            # the verdict can arrive as an injected turn.
                            result = {"output": self._start_decision_check(dict(call.args or {}))}
                        elif (call.name == "ada_ops"
                              and str((call.args or {}).get("action")
                                      or "").lower() == "research"
                              and self.tool_runner is not None):
                            # absorbed ada_deep_research — provider-side so
                            # findings can arrive as an injected turn.
                            result = {"output": self._start_deep_research(dict(call.args or {}))}
                        elif (call.name == "ada_remember" and budget_hit
                              and str((call.args or {}).get("kind") or "")
                              .lower() not in ("vocab", "habit", "guest")):
                            self._emit_ops_event(
                                "remember_block",
                                "ada_remember refused — turn hit the tool "
                                "budget, content may be confabulated.",
                                tool="ada_remember")
                            # The plan/summary the model wants to save never
                            # survived the storm — refuse rather than persist
                            # confabulated content.
                            result = {"error": (
                                "The tool budget was exhausted this turn, so the earlier "
                                "part of the answer may be missing — restate the content "
                                "aloud and save it on the next turn.")}
                        else:
                            if self.tool_runner is not None:
                                try:
                                    call_args = dict(call.args or {})
                                    if (str(call.name) in _CONFIRM_GATED_TOOLS
                                            # non-bank ada_remember kinds
                                            # (vocab/guest/habit) absorbed
                                            # ungated tools — skip the probe.
                                            and not (str(call.name) == "ada_remember"
                                                     and str(call_args.get("kind")
                                                             or "").lower()
                                                     in ("vocab", "habit", "guest"))
                                            # tools-merge-meta-voice: only
                                            # action='outcome' sits in the
                                            # memory-write seat; the other
                                            # ada_ops actions absorbed
                                            # ungated tools.
                                            and not (str(call.name) == "ada_ops"
                                                     and str(call_args.get("action")
                                                             or "").lower()
                                                     != "outcome")
                                            # who='guest' absorbed
                                            # guest_register — a plain
                                            # name registration that was
                                            # never confirm-gated.
                                            and not (str(call.name) == "ada_enroll_speaker"
                                                     and str(call_args.get("who")
                                                             or "").lower()
                                                     == "guest")):
                                        # Resolved source text shared by the
                                        # gate and the Jev advisory probe —
                                        # probe fires on every gated call,
                                        # not only self-asserted confirms.
                                        _src = self._confirm_source_text(input_transcript)
                                        _affirmed = self._user_confirmed(input_transcript)
                                        self._jev_confirm_probe(
                                            _src, _affirmed, str(call.name))
                                    if call_args.get("confirmed"):
                                        _src = self._confirm_source_text(input_transcript)
                                        _affirmed = self._user_confirmed(input_transcript)
                                        if not _affirmed and self._confirm_retry_pending(
                                                str(call.name), call_args):
                                            # The denial already proposed
                                            # this exact call and minted a
                                            # bound token — the resend is
                                            # the replay, just spelled
                                            # confirmed=true instead of
                                            # confirm_token (the model's
                                            # actual behavior, 2026-10-07
                                            # flake). The gate still forced
                                            # the propose → deny → retry
                                            # protocol; let it through.
                                            logger.info(
                                                "session=%s %s confirmed=true "
                                                "retry of a pending proposal — "
                                                "accepting (token pending, no "
                                                "voice affirmation detected)",
                                                self.session_id, call.name)
                                            self._emit_ops_event(
                                                "confirm_retry_accept",
                                                f"Accepted confirmed=true "
                                                f"retry of the pending "
                                                f"{call.name} proposal — "
                                                "identical args already had a "
                                                "live confirm_token; no user "
                                                "affirmation detected.",
                                                tool=str(call.name))
                                        elif not _affirmed:
                                            logger.warning(
                                                "session=%s %s self-asserted confirmed=true "
                                                "without user affirmation — stripping",
                                                self.session_id, call.name,
                                            )
                                            call_args.pop("confirmed", None)
                                            self._emit_ops_event(
                                                "confirm_strip",
                                                f"Stripped model-asserted "
                                                f"confirmed=true on {call.name} — "
                                                f"no user affirmation found.",
                                                tool=str(call.name))
                                        else:
                                            # User actually affirmed — the
                                            # pending-register step is
                                            # redundant friction (observed
                                            # 2026-09-29: 'approve' → denied
                                            # 'no pending' → double-ask loop).
                                            call_args["_verified_affirm"] = True
                                    if call.name == "ada_memory_search":
                                        q = call_args.get("query")
                                        if q:
                                            call_args["query"] = self.conversation.expand_query(str(q))
                                    # Authorization identity is the session
                                    # OWNER pinned at connect — the speaking
                                    # voice only personalizes, it never
                                    # upgrades permissions mid-session (P1).
                                    owner = self.session_owner
                                    output = await self.tool_runner.execute(
                                        str(call.name), call_args,
                                        identity=(owner
                                                  or self.current_speaker_ha_person),
                                        speaker=self.current_speaker_ha_person,
                                        speaker_session=self.speaker_session,
                                        owner=owner,
                                        session=self.session_id)
                                    # Contract (tool-error-contract):
                                    # execute() already returns the
                                    # canonical {ok: bool, ...} dict —
                                    # pass it through at top level so an
                                    # ok:false is never buried under an
                                    # "output" wrapper.
                                    result = (output if isinstance(output, dict)
                                              else {"ok": True, "output": output})
                                    if call.name == "ada_memory_search":
                                        self._note_search_result(output)
                                except Exception as exc:
                                    result = {"error": f"{call.name} failed: {exc}"}
                            else:
                                result = {"error": "Unsupported or unavailable function"}
                        # Uniform result contract: every tool_result the
                        # model sees carries top-level ok: bool — the
                        # honesty checks (event log below, scenario-live
                        # no_failed_result) rely on it.
                        result = normalize_tool_result(result)
                        if result.get("ok") is False:
                            failed_results_this_turn += 1
                            if _claim_backing_call(
                                    str(call.name),
                                    dict(call.args or {})):
                                # Phantom-claim gate (card
                                # ada-phantom-claim-gate): a claimable
                                # action that failed/denied must reach the
                                # model with the honesty contract attached
                                # — kk-dev probe 1 narrated "being sent"
                                # over a denied chat_send photo:pick.
                                result.setdefault(
                                    "speech_rule",
                                    _CLAIM_FAILED_SPEECH_RULE)
                        elif _claim_backing_call(
                                str(call.name), dict(call.args or {})):
                            backing_calls_this_turn += 1
                        turn_tool_results.append((str(call.name), result))
                        if str(call.name) in _FIDELITY_WRITE_TOOLS:
                            # Args as the model sent them — the narration
                            # is compared against THIS text at turn end;
                            # the persisted bytes match it verbatim
                            # (cms_publish_page read-back verifies that).
                            turn_fidelity_calls.append((
                                str(call.name), dict(call.args or {}),
                                result))
                        # Ground-truth pin (card ada-trace-full-result):
                        # the ws tool_result stays a stump, but id +
                        # result_hash join it to the call-log result_full
                        # line (and to tool_result_full under
                        # ADA_TRACE_FULL) keyed by session+call id.
                        result_hash = _result_hash(result)
                        # tool= is the resolved canonical; emitted= keeps
                        # the as-called name (alias or typo) for the
                        # alias-hit rollup (card ada-alias-telemetry).
                        self.conversation.log_event(
                            "tool_call", tool=str(call.name),
                            emitted=_call_name,
                            dur_ms=int((time.monotonic() - tool_t0) * 1000),
                            ok=bool(result.get("ok", True)),
                            result_hash=result_hash)
                        yield ProviderEvent("tool_result", {
                            "id": str(call.id or ""),
                            "name": str(call.name),
                            "result": _safe_args(result) if isinstance(result, dict) else {"value": str(result)[:500]},
                            "result_hash": result_hash,
                        })
                        if self._trace_full:
                            yield ProviderEvent("tool_result_full", {
                                "id": str(call.id or ""),
                                "name": str(call.name),
                                "result": _json_safe(result),
                                "result_hash": result_hash,
                            })
                        self._tools_in_flight = max(0, self._tools_in_flight - 1)
                        function_responses.append(types.FunctionResponse(
                            id=call.id,
                            name=_call_name or "set_facial_expression",
                            response=_cap_tool_result(result),
                            # The expression tool often arrives before audio.
                            # WHEN_IDLE lets Gemini continue the spoken reply;
                            # SILENT would add the result to context without
                            # triggering generation, leaving Ada mute.
                            scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
                        ))
                        logger.info(
                            "session=%s function_call result id=%s name=%s result=%r",
                            self.session_id, call.id, call.name, result,
                        )
                        self._write_call_log({
                            "event": "function_result",
                            "id": call.id, "name": call.name,
                            "result_hash": result_hash,
                            "result": _safe_args(result)
                            if isinstance(result, dict)
                            else {"value": str(result)[:500]},
                            "result_full": _full_result_payload(result)})
                    await self._session.send_tool_response(
                        function_responses=function_responses
                    )
                    logger.info(
                        "session=%s function_call responses sent count=%d",
                        self.session_id, len(function_responses),
                    )
                    # Camera frames ride as follow-up client content (same
                    # pattern as send_habit_alert) — FunctionResponse.parts
                    # crashes send_tool_response's json.dumps on bytes.
                    for cam_name, cam_img, cam_mime in camera_frames:
                        # Vision sidecar (ADA_VISION_MODE): a cheap model
                        # describes the frame so the Live session doesn't
                        # have to — quota storms stall hardest on image
                        # turns (2026-09-30 empty-response issue).
                        desc = None
                        if vision_describe.configured():
                            try:
                                desc = await vision_describe.describe_frame(
                                    cam_img, cam_mime, cam_name)
                            except Exception:
                                desc = None
                        try:
                            if desc and vision_describe.mode() == "describe":
                                parts = [types.Part.from_text(text=(
                                    f"Camera frame from '{cam_name}' was "
                                    "analyzed by the vision helper (the live "
                                    "session skipped the image to save quota). "
                                    f"Helper says: {desc}\nRelay this to the "
                                    "user naturally — if the description seems "
                                    "thin, you may look again on request."))]
                            else:
                                verb = ("is the STALE last-known frame — "
                                        "describe it but say it is old" if
                                        "STALE" in cam_name else
                                        "just arrived — describe to the "
                                        "user what it shows.")
                                parts = [types.Part.from_text(text=(
                                    f"Camera frame from '{cam_name}' "
                                    f"{verb}"
                                    + (f" (helper agrees: {desc})"
                                       if desc else "")))]
                                parts.append(types.Part.from_bytes(
                                    data=cam_img, mime_type=cam_mime))
                            async with self._send_lock:
                                await self._session.send_client_content(
                                    turns=types.Content(
                                        role="user", parts=parts),
                                    turn_complete=True,
                                )
                        except Exception as exc:
                            logger.warning(
                                "session=%s camera frame send failed: %s",
                                self.session_id, exc,
                            )
                    if budget_hit and not budget_nudged:
                        # Refused results alone don't stop a storm — the model
                        # keeps emitting calls. Inject an explicit user-turn
                        # nudge so the turn has to produce an answer.
                        budget_nudged = True
                        try:
                            async with self._send_lock:
                                await self._session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[types.Part.from_text(text=(
                                            "[system] Tool-call limit reached for this turn — "
                                            "stop calling tools and answer the user now, "
                                            "briefly, from what you already have."))],
                                    ),
                                    turn_complete=True,
                                )
                        except Exception:
                            logger.debug("budget nudge send failed", exc_info=True)

                content = message.server_content
                if content is None:
                    continue

                if content.interrupted:
                    elapsed = time.monotonic() - response_started_at if response_started_at else 0.0
                    logger.warning(
                        "session=%s assistant interrupted response_active=%s age_ms=%d audio_chunks=%d "
                        "audio_bytes=%d pending_input_transcript=%r",
                        self.session_id, self._response_active, elapsed * 1000, response_audio_chunks,
                        response_audio_bytes, input_transcript.strip(),
                    )
                    self._response_active = False
                    tool_calls_this_turn = 0
                    expr_calls_this_turn = 0
                    failed_results_this_turn = 0
                    backing_calls_this_turn = 0
                    actuations_this_turn = 0
                    budget_hit = False
                    budget_nudged = False
                    turn_tool_results = []
                    turn_fidelity_calls = []
                    segment_tool_pending = False
                    self._dead_retried = False
                    self._revoice_in_flight = False
                    self._leak_active = False
                    self._artifact_hold = ""
                    self._artifact_logged = False
                    self._turn_open = False
                    barge_pending = True
                    self.conversation.log_event(
                        "barge_in",
                        speaker=(self.current_speaker
                                 or self.current_speaker_ha_person),
                        response_age_ms=int(elapsed * 1000))
                    yield ProviderEvent("response_interrupted", {})
                    # Gemini 3.1 can include several content parts in one event.
                    # Any audio/transcript accompanying an interruption belongs
                    # to the cancelled response and must not restart playback.
                    continue

                transcription = content.input_transcription
                if transcription and transcription.text:
                    input_transcript += transcription.text
                    self._live_input_transcript = input_transcript
                    input_done_ts = time.monotonic()
                    self._last_user_at = time.monotonic()
                    self._turn_open = True

                output_transcription = content.output_transcription
                if output_transcription and output_transcription.text:
                    if not self._response_active:
                        self._response_active = True
                        response_started_at = time.monotonic()
                        response_audio_chunks = 0
                        response_audio_bytes = 0
                        yield ProviderEvent("response_started", {})
                    clean = self._strip_tool_leak(output_transcription.text)
                    if clean:
                        clean = self._strip_speech_artifacts(clean)
                    if clean:
                        assistant_turn_text += clean
                        yield ProviderEvent(
                            "assistant_transcript_delta",
                            {"text": clean},
                        )

                model_turn = content.model_turn
                if model_turn:
                    for part in model_turn.parts or []:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            # TEXT-modality sessions (telegram/line relays):
                            # reply text arrives inline — same delta path as
                            # output_audio_transcription on voice sessions.
                            if not self._response_active:
                                self._response_active = True
                                response_started_at = time.monotonic()
                                response_audio_chunks = 0
                                response_audio_bytes = 0
                                yield ProviderEvent("response_started", {})
                            clean = self._strip_tool_leak(part_text)
                            if clean:
                                clean = self._strip_speech_artifacts(clean)
                            if clean:
                                assistant_turn_text += clean
                                yield ProviderEvent(
                                    "assistant_transcript_delta",
                                    {"text": clean},
                                )
                        inline_data = part.inline_data
                        if inline_data and inline_data.data:
                            if not self._response_active:
                                self._response_active = True
                                response_started_at = time.monotonic()
                                response_audio_chunks = 0
                                response_audio_bytes = 0
                                yield ProviderEvent("response_started", {})
                            response_audio_chunks += 1
                            response_audio_bytes += len(inline_data.data)
                            yield ProviderEvent("audio", {"pcm16": inline_data.data})

                if content.turn_complete:
                    tools_pending = segment_tool_pending
                    segment_tool_pending = False
                    # Forced re-voice (card ada-phantom-claim-gate): True
                    # only at the boundary of the correction turn itself —
                    # captured then cleared so a re-claim inside the
                    # correction can't re-arm the nudge (one per claim).
                    _in_revoice = self._revoice_in_flight
                    self._revoice_in_flight = False
                    _unbacked_claim = ""
                    _claim_failed: list[tuple[str, dict]] = []
                    transcript = input_transcript.strip()
                    if transcript:
                        if barge_pending and _looks_like_noise(transcript):
                            # Obvious noise that tripped the VAD — keep it
                            # out of the transcript; the browser pump nudges
                            # Ada to resume (policy P6: ignore gibberish).
                            self.conversation.log_event(
                                "barge_noise", text=transcript[:80])
                            yield ProviderEvent(
                                "barge_noise", {"text": transcript})
                        else:
                            self.conversation.add_user(transcript)
                            # Language lock (card ada-output-hygiene-tags):
                            # only a clearly-dominant script flips the
                            # lock — a stray "ok" or loanword doesn't
                            # re-key the session language.
                            _uth, _ula = _count_scripts(transcript)
                            if _uth >= 3 and _uth > _ula:
                                self._last_user_thai = True
                            elif _ula >= 8 and _ula > 2 * _uth:
                                self._last_user_thai = False
                            yield ProviderEvent("user_transcript",
                                                {"text": transcript})
                            # Local command lane (card
                            # use-local-model-for-short-voice-commands):
                            # advisory probe — scores whether the local
                            # model would have served this short turn;
                            # corpus rows are the promotion evidence.
                            # Voice stays shadow in every mode: the audio
                            # already streamed to Gemini.
                            command_lane.probe(
                                transcript, surface="voice",
                                session_id=self.session_id,
                                tools=[n for n, _ in turn_tool_results],
                                emit=self._emit_ops_event)
                    barge_pending = False
                    input_transcript = ""
                    self._live_input_transcript = ""
                    n_tools_this_turn = tool_calls_this_turn
                    # A write claim is honest only when a call that can
                    # back it returned ok this turn — claiming "saved"
                    # over a NOT-EXECUTED result is the same phantom
                    # class as claiming one with no call at all
                    # (2026-10-07 flake), and a passed READ doesn't back
                    # a write (card ada-phantom-card-claims: narrated
                    # board cards while only reads/zero calls ran).
                    _claim_hit = _phantom_claim(assistant_turn_text)
                    if _claim_hit and (
                            backing_calls_this_turn == 0
                            or failed_results_this_turn):
                        _claim_basis = (
                            "no backing write call this turn"
                            if backing_calls_this_turn == 0 else
                            f"{failed_results_this_turn} failed tool "
                            "result(s) this turn")
                        self._emit_ops_event(
                            "phantom_write_claim",
                            f"assistant claimed a write with {_claim_basis}: "
                            f"{assistant_turn_text.strip()[:160]!r}",
                        )
                    # Re-voice arm (card ada-phantom-claim-gate): only a
                    # claim with NO ok claim-capable call behind it is
                    # provably false — a failed sibling call elsewhere in
                    # the turn must not make us deny a write that DID
                    # land. Recall-framed claims ("I sent it yesterday")
                    # likewise stand down — correcting them would narrate
                    # a falsehood of our own. tools_pending mid-task
                    # boundaries arm too but the tail gate stands down —
                    # a "sending it now" intent line can still earn its
                    # backing call in the next segment.
                    if _claim_hit and backing_calls_this_turn == 0 and not (
                            _RECALL_CLAIM_RE.search(assistant_turn_text)):
                        _unbacked_claim = assistant_turn_text.strip()[:160]
                        _claim_failed = [
                            (n, r) for n, r in turn_tool_results
                            if isinstance(r, dict)
                            and r.get("ok") is False]
                    # Media narration guard (card
                    # ada-tv-action-hallucinated-input, 2026-10-09): a
                    # named title or playing claim must literally appear
                    # in a tool result from THIS turn — 'Dancing In The
                    # Street' was narrated over a cast_verify=paused.
                    _media_flags = _media_grounding_problems(
                        assistant_turn_text, turn_tool_results)
                    if _media_flags:
                        self._emit_ops_event(
                            "phantom_media_claim",
                            "media claim with no supporting tool result "
                            f"this turn ({'; '.join(_media_flags)}): "
                            f"{assistant_turn_text.strip()[:160]!r}",
                        )
                    # Write-fidelity guard (card ada-cms-write-fidelity):
                    # a publish call's persisted content can contradict
                    # BOTH the narration and the user seed — 'Tuesdays'
                    # stored while 'Mondays' was said/asked (2026-10-12
                    # kk-cms-only bench). Fact-token contradiction →
                    # advisory ops event; the session-audit/QC loop sees
                    # it, the model sees nothing (no retry channel).
                    if turn_fidelity_calls and not tools_pending:
                        # Last 3 user turns — the seed ask + the confirm
                        # are what the stored page should agree with;
                        # older turns' facts would only add false hits.
                        # Mid-task boundaries are skipped — calls persist
                        # until the final segment so the check sees the
                        # whole narration and fires at most once.
                        _seed = " ".join(
                            str(t.get("text") or "") for t in
                            [t for t in self.conversation.turns()[-12:]
                             if t.get("role") == "user"][-3:])
                        _fidelity_flags = _write_fidelity_problems(
                            assistant_turn_text, _seed,
                            turn_fidelity_calls)
                        for _ff in _fidelity_flags:
                            self._emit_ops_event("write_fidelity", _ff)
                    if not tools_pending:
                        # Claim-gate counters reset at the FINAL boundary
                        # only — a mid-task reset orphaned the call from
                        # the reply segment's narration, so an ok send in
                        # segment 1 could never back a "sent it" in
                        # segment 2 (card ada-phantom-claim-gate). Same
                        # lifecycle as turn_tool_results below.
                        tool_calls_this_turn = 0
                        expr_calls_this_turn = 0
                        failed_results_this_turn = 0
                        backing_calls_this_turn = 0
                    actuations_this_turn = 0
                    budget_hit = False
                    budget_nudged = False
                    if self._artifact_hold and not self._leak_active:
                        tail = self._flush_artifact_hold()
                        if tail:
                            assistant_turn_text += tail
                            yield ProviderEvent(
                                "assistant_transcript_delta",
                                {"text": tail},
                            )
                    else:
                        self._artifact_hold = ""
                    self._leak_active = False
                    self._artifact_logged = False
                    # Dead-turn guard (card ada-dead-turn-guard, session
                    # 56d2e4d167): the model twice ended a turn on the
                    # literal "response:" while memory_search already held
                    # the plate — the user heard silence and repeated the
                    # question. A degenerate turn (bare scaffold prefix,
                    # whitespace, lone structural tags — or nothing at all
                    # when zero audio played) is never written to the
                    # transcript as the answer: emit a dead_turn ops event,
                    # deterministically retry ONCE with the tool-result
                    # digest handed to the model, and if the retry also
                    # lands dead, synthesize the fallback that speaks the
                    # result itself.
                    _dead_evidence = (
                        transcript or self._turn_open
                        or n_tools_this_turn or self._response_active
                        or turn_tool_results)
                    # A boundary that dispatched tool calls is mid-task —
                    # the model is waiting on results, not silent. Judging
                    # it "dead" there synthesized "Ada is now neutral" from
                    # every set_facial_expression result and poisoned the
                    # transcript (scenario run 2026-10-07).
                    _dead = _dead_evidence and not tools_pending and (
                        _dead_turn_text(assistant_turn_text)
                        if assistant_turn_text.strip()
                        else response_audio_bytes == 0)
                    if _dead:
                        self._emit_ops_event(
                            "dead_turn",
                            f"turn produced no speakable answer "
                            f"(raw={assistant_turn_text.strip()[:80]!r}, "
                            f"audio_bytes={response_audio_bytes}, tools="
                            f"{[n for n, _ in turn_tool_results]}) — "
                            + ("retrying once"
                               if not self._dead_retried
                               else "synthesizing fallback"))
                        if (not self._dead_retried
                                and await self._send_dead_turn_retry(
                                    turn_tool_results, transcript)):
                            # The retry opens a fresh model turn. Close
                            # the dead bubble first — the retry's answer
                            # must not append onto the visible scaffold
                            # fragment. Keep turn_tool_results for the
                            # fallback if the retry lands dead too.
                            if self._response_active:
                                yield ProviderEvent("response_completed", {})
                            self._response_active = False
                            response_audio_chunks = 0
                            response_audio_bytes = 0
                            response_started_at = 0.0
                            self._dead_retried = True
                            assistant_turn_text = ""
                            input_done_ts = 0.0
                            continue
                        fallback = self._dead_turn_fallback(
                            turn_tool_results)
                        logger.warning(
                            "session=%s dead turn — synthesized fallback %r",
                            self.session_id, fallback[:120])
                        if not self._response_active:
                            # Pure-silence dead turn — the fallback still
                            # gets the normal started/delta/completed frame
                            # so the UI renders it like any reply.
                            self._response_active = True
                            response_started_at = time.monotonic()
                            yield ProviderEvent("response_started", {})
                        yield ProviderEvent(
                            "assistant_transcript_delta", {"text": fallback})
                        self.conversation.add_assistant(fallback)
                        assistant_turn_text = ""
                    self._dead_retried = False
                    if not tools_pending:
                        # Keep results across a mid-task boundary so a dead
                        # FINAL segment can still speak them as the fallback.
                        turn_tool_results = []
                        turn_fidelity_calls = []
                    _lang_drift = ""
                    if assistant_turn_text.strip():
                        self.conversation.add_assistant(assistant_turn_text)
                        if (self._last_user_thai
                                and _assistant_drifted_english(
                                    assistant_turn_text)):
                            _lang_drift = assistant_turn_text.strip()[:160]
                        assistant_turn_text = ""
                    if self._response_active:
                        elapsed = time.monotonic() - response_started_at if response_started_at else 0.0
                        logger.info(
                            "session=%s assistant response completed duration_ms=%d audio_chunks=%d audio_bytes=%d",
                            self.session_id, elapsed * 1000, response_audio_chunks, response_audio_bytes,
                        )
                        # TTFT ≈ last user-speech transcription chunk → first
                        # model output. Feeds the latency benchmark rollup.
                        self.conversation.log_event(
                            "turn_latency",
                            ttft_ms=(int((response_started_at - input_done_ts)
                                         * 1000) if input_done_ts else None),
                            dur_ms=int(elapsed * 1000),
                            audio_bytes=response_audio_bytes,
                            tools=n_tools_this_turn)
                        yield ProviderEvent("response_completed", {})
                    input_done_ts = 0.0
                    self._response_active = False
                    if not tools_pending:
                        # The user's turn stays open across a mid-task
                        # boundary — the final segment still needs it as
                        # dead-turn evidence.
                        self._turn_open = False
                    # Language drift (card ada-output-hygiene-tags —
                    # session e9837ae922 flipped to English 3x mid-Thai):
                    # flag it for the audit, then re-lock silently — a
                    # turn_complete=False note lands in context without
                    # prompting a reply. Capped per session (context
                    # budget): two locks in, a drifting model needs a
                    # session rotate, not more notes.
                    if _lang_drift:
                        logger.warning(
                            "session=%s language drift — English reply in "
                            "a Thai-locked session: %r",
                            self.session_id, _lang_drift)
                        self._emit_ops_event(
                            "language_drift",
                            "assistant replied in English while the user "
                            f"was speaking Thai: {_lang_drift!r}")
                        if self._lang_locks_sent < 2:
                            self._lang_locks_sent += 1
                            try:
                                await self._send_context_note(
                                    "(system) Language lock — the speaker "
                                    "is using Thai. Keep every reply in "
                                    "Thai end to end: an English name or "
                                    "technical term inside a Thai sentence "
                                    "is fine, a whole English clause is a "
                                    "slip.")
                            except Exception:
                                logger.debug(
                                    "language-lock note send failed",
                                    exc_info=True)
                    # Forced re-voice (card ada-phantom-claim-gate): the
                    # turn closed on a success claim nothing backs —
                    # reopen the model turn so it corrects itself aloud.
                    # Mid-task boundaries stand down (a pending call can
                    # still back the claim), the correction's own
                    # boundary stands down (no re-voice loops), and a
                    # failed send gets the whole turn's failures handed
                    # over so the correction can name the real reason.
                    if (_unbacked_claim and not tools_pending
                            and not _in_revoice
                            and await self._send_claim_correction(
                                _unbacked_claim, _claim_failed)):
                        self._revoice_in_flight = True
                        logger.warning(
                            "session=%s phantom claim re-voice sent: %r",
                            self.session_id, _unbacked_claim[:120])
                        self._emit_ops_event(
                            "phantom_claim_revoice",
                            f"forced correction after unbacked claim: "
                            f"{_unbacked_claim!r}")
                    # A notification queued mid-turn — deliver it now that
                    # the response finished, as its own turn.
                    if self._pending_notifications:
                        await self._drain_notifications()
                    # Durable-write outbox dead letters — a parked write
                    # exhausted its retry window; Ada tells the user on
                    # this next turn rather than letting it stay silent.
                    for _notice in write_outbox.drain_notices():
                        await self.notify_or_defer(
                            _notice, urgent=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Stash any in-flight user utterance before the session dies — a
        # close mid-input (rotate, stall, drop) otherwise loses the text
        # because turn_complete never fired and add_user never ran. The
        # reconnect path appends it to the session prime's tail.
        if self._live_input_transcript.strip():
            self.pending_user_turn = self._live_input_transcript.strip()
        if self.usage_input_tokens or self.usage_output_tokens:
            logger.info(
                "session=%s usage summary %s", self.session_id, self.usage_summary()
            )
        # Persist conversation to NotebookLM in the background so we don't block disconnect.
        if self.conversation:
            _ = asyncio.create_task(self.conversation.persist())
        if self._session_context is not None:
            try:
                await asyncio.wait_for(
                    self._session_context.__aexit__(None, None, None), timeout=3
                )
            except (TimeoutError, Exception):
                logger.debug("Gemini session close did not complete cleanly", exc_info=True)
        if self._client is not None:
            try:
                await self._client.aio.aclose()
            except Exception:
                logger.debug("Gemini client close did not complete cleanly", exc_info=True)


def create_provider(instructions: str | None = None, tool_runner: Any = None,
                    home_assistant_client: Any = None, habit_state_getter: Any = None,
                    session_id: str | None = None,
                    conversation: ConversationMemory | None = None,
                    caller_name: str | None = None,
                    caller_person: str | None = None,
                    channel: str | None = None) -> RealtimeProvider:
    return GeminiLiveProvider(
        instructions=instructions,
        tool_runner=tool_runner,
        home_assistant_client=home_assistant_client,
        habit_state_getter=habit_state_getter,
        session_id=session_id,
        conversation=conversation,
        caller_name=caller_name,
        caller_person=caller_person,
        channel=channel,
    )

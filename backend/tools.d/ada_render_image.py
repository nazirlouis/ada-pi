"""ada_render_image — drop-in tool (cards nano-banana-image, ada-render-image-v2).

Generate an image from an assembled prompt — or edit/compose the ref_url
image — through the Gemini image model (default gemini-3.1-flash-image,
"Nano Banana 2": one model does text-to-image AND image+instruction
editing). The call goes through gemini_pool.generate(): on gateway lanes
(ADA_GEMINI_REST_BASE_URL set) the raw name maps to the 'image' model
class — LiteLLM routing + per-lane virtual-key metering; direct-Gemini
dev keeps the raw model name.

provider='local' (card local-image-lane) renders on tony-omen's
sd-server (stable-diffusion.cpp, SDXL-Lightning 4-step Q4, CPU-only)
via the native async sdcpp API: POST /sdcpp/v1/img_gen returns a job id
instantly, GET /sdcpp/v1/jobs/{id} polls it. CPU renders take ~10min at
1024px and the server is a single slot with a queue — so a render that
outlives the short foreground wait returns ok:false + pending + job,
and a background task keeps polling and pushes the finished photo to the
originating chat through chat_send. job= re-attaches to an in-flight
job after a restart, mirroring ada_render_video's operation= resume —
no second render is queued. Local is text-to-image only: ref_url edits
stay on the default provider.

The PNG lands under ADA_RENDER_DIR and publishes through the
input-bridge /frame asset hop ada_render_video uses — the returned
image_url is fetchable by the relays (LINE imageMessage / TG sendPhoto
auto-attach it via _images_from).

ref_url (edit/compose a source photo — 'make the car red', 'put me on
a beach') accepts the same three shapes as ada_render_video:
  - an intake key 'doc/...' from a [document uploaded] session note —
    resolved in-process via the shared document_check engine;
  - a /api/documents/<key>/{image,preview} path (relative or absolute);
  - any other fetchable http(s) image URL — including the relay's own
    /relay-img/arch/<chat>/<message>.jpg archives of photos users sent.
Plus two quality-pass additions (card render-flow-quality-pass):
  - ref_url='last' resolves the session's previous render through
    backend/render_state (on-disk artifact first, published URL as
    fallback) so 'make it warmer' iterates instead of rerolling — and
    a short prompt carrying an edit cue auto-feeds it even when
    ref_url is omitted (gemini path only; local stays t2i);
  - ref_urls[] takes up to 3 sources total for 'combine these photos'
    asks — the model emits multiple reference images natively.

Post-render quality pass: one cheap vision check
(ADA_RENDER_CHECK_MODEL, default gemini-3.5-flash-lite — gateway
'summary' class, ~$0.004/turn) compares output vs ask; an obvious
miss (wrong subject, garbled requested text) re-renders once with
the critique appended. ADA_RENDER_IMAGE_SELFCHECK=0 disables.
Gateway/5xx/timeout failures get one automatic retry
(ADA_RENDER_RETRY_S backoff); quota-shaped errors go through the
pool's 429-vs-quota split, not the retry. 2K/4K renders also publish
the original bytes to the relay media archive and return full_url —
LINE re-encodes the attached photo, so the link is the full-res file
(needs OPS-0651's arch route serving). Page-ish asks ('landing page',
'1920px…') attach a 'notice' — gen is aspect-capped and a render is
a picture OF a page, not a page.

Env:
    GEMINI_API_KEY / GOOGLE_API_KEY — required (pool bookkeeping; the
        actual request key is ADA_GEMINI_REST_API_KEY when the gateway
        env is set)
    ADA_RENDER_IMAGE_MODEL — default gemini-3.1-flash-image
    ADA_GATES_DIR          — owner kill-switch dir (default
        ~/.local/share/ada/gates); touch ada_render_image.closed or
        ada_render_image.<lane>.closed to refuse renders instantly
    ADA_RENDER_DIR         — artifact dir, default ~/.local/share/ada/renders
    ADA_SELF_URL           — self API base for relative ref paths
                             (default http://127.0.0.1:$ADA_PORT or :8000)
    ADA_SD_SERVER_URL      — local provider base, default
                             http://100.75.102.88:8399 (tony-omen tailnet
                             bind — reachable from idc03 lanes and
                             on-host hairpin; never on LAN)
    ADA_RENDER_LOCAL_WAIT_S — foreground poll budget before pending,
                             default 45
    ADA_RENDER_LOCAL_POLL_S — poll interval, default 5
    ADA_RENDER_LOCAL_DEADLINE_S — background delivery gives up after
                             this, default 1800
    ADA_RENDER_CHECK_MODEL     — self-check vision model, default
                             gemini-3.5-flash-lite
    ADA_RENDER_IMAGE_SELFCHECK — '0' disables the post-render QA pass
    ADA_RENDER_RETRY_S         — transient-failure retry backoff,
                             default 2
    LINE_MEDIA_ARCHIVE_DIR / ADA_RELAY_PUBLIC_BASE — arch publish hop
                             for full_url on 2K/4K renders
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import secrets
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("tools.ada_render_image")

DECLARATION = {
    "name": "ada_render_image",
    "description": (
        "Generate an image from a prompt, or edit source photos "
        "(ref_url/ref_urls, 'last' = the previous render) with "
        "instructions, and return a shareable image_url the chat "
        "relays post as a photo."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The assembled image description — "
                               "subject, style, mood, detail; with "
                               "ref_url it is the edit instruction. "
                               "Required.",
            },
            "ref_url": {
                "type": "string",
                "description": "Optional source image to edit: 'last' "
                               "(the previous render this session — use "
                               "it for 'make it warmer' style follow-ups), "
                               "a 'doc/...' intake key, a "
                               "/api/documents/... path, or any "
                               "fetchable image URL.",
            },
            "ref_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional extra source images, same "
                               "shapes as ref_url ('last' allowed) — "
                               "up to 3 total, for 'combine these "
                               "photos' asks.",
            },
            "aspect": {
                "type": "string",
                "description": "square (1:1, default) | landscape "
                               "(16:9) | portrait (9:16) | an explicit "
                               "ratio.",
            },
            "size": {
                "type": "string",
                "description": "1K (default, ~1024px, cheapest) | 2K "
                               "| 4K (~4096px, sharpest, costs most). "
                               "Only go above 1K when detail matters.",
            },
            "provider": {
                "type": "string",
                "description": "gemini (default — fast cloud render, "
                               "can edit ref_url) | local — tony-omen "
                               "sd-server, free and private but ~10 "
                               "minutes on CPU; returns pending and the "
                               "photo arrives via chat_send when done.",
            },
            "job": {
                "type": "string",
                "description": "Local provider only: re-attach to an "
                               "in-flight render — the 'job' id a "
                               "pending call returned. No new render "
                               "is queued.",
            },
        },
        "required": ["prompt"],
    },
}

_DEFAULT_MODEL = "gemini-3.1-flash-image"
# LiteLLM's image alias can return HTTP 200 with an empty candidates list
# when the primary upstream has no usable image response. Try the explicit
# paid-lite and OpenRouter image aliases before reporting failure.
_IMAGE_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite-image",
    "or-gemini-3.1-flash-image",
)
_DEFAULT_DIR = "~/.local/share/ada/renders"
_PROMPT_MAX = 2000
_REF_MAX_BYTES = 15 * 1024 * 1024
_IMG_MAX_BYTES = 20 * 1024 * 1024

_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4"}
_ASPECT_ALIASES = {
    "square": "1:1", "landscape": "16:9", "wide": "16:9",
    "portrait": "9:16", "tall": "9:16", "vertical": "9:16",
}

_SIZES = {"1K", "2K", "4K"}
_SIZE_ALIASES = {"small": "1K", "default": "1K", "medium": "2K",
                 "hd": "2K", "large": "4K", "big": "4K", "ultra": "4K"}

_DOC_KEY_RE = re.compile(r"^doc/[a-z0-9][a-z0-9._-]*$")
_DOC_PATH_RE = re.compile(
    r"/api/documents/([^?#]+?)/(image|preview|pdf)(?:[?#].*)?$")
_JOB_RE = re.compile(r"^job_[A-Za-z0-9_-]+$")

# Multi-ref (card render-flow-quality-pass): gemini-3.1-flash-image
# accepts several reference images per call; cap keeps a prompt from
# becoming an image dump.
_REFS_MAX = 3

# Iterate-on-last auto-feed: a short prompt that references the previous
# artifact ("make it warmer", "change the text") and no explicit
# fresh-render cue feeds the session's last render as the edit source.
# Strong cues are pronouns/comparatives/edit verbs; phrase cues like
# "the text" also fire — but not when the prompt opens with an
# indefinite article, which reads as a fresh subject ("a poster with
# the text SALE" is a new render, not an edit of the last one).
_EDIT_CUE_RE = re.compile(
    r"\b(it|this|that|them|those|same|again|instead|the image|"
    r"the picture|the photo|the render|the last one|warmer|cooler|"
    r"brighter|darker|bigger|smaller|softer|sharper|changes?|swap|"
    r"remove|keep|fix|adjust|tweak|recolou?r|re-?word|crop|zoom)\b",
    re.I)
_EDIT_PHRASE_RE = re.compile(
    r"\bthe (text|caption|wording|background|colou?rs?|font|subject|"
    r"logo|title)\b", re.I)
_FRESH_LEAD_RE = re.compile(
    r"^\s*(?:a|an|some|two|three|one|several|draw|generate|create|"
    r"make me a|render)\s", re.I)
_FRESH_CUE_RE = re.compile(
    r"\b(new|another|different|fresh|from scratch|re-?roll|"
    r"something else|brand new)\b", re.I)
_AUTO_REF_PROMPT_MAX = 400


def _edit_cued(prompt: str) -> bool:
    """True when the assembled prompt reads as an edit of the previous
    artifact rather than a fresh scene."""
    if _FRESH_CUE_RE.search(prompt):
        return False
    if _EDIT_CUE_RE.search(prompt):
        return True
    return bool(_EDIT_PHRASE_RE.search(prompt)
                and not _FRESH_LEAD_RE.search(prompt))

# Transient retry (same card): one retry on transport/gateway wobble —
# quota-shaped errors are NOT transient (the pool already rotated keys
# and a drained pool is a policy answer, not a blip).
_TRANSIENT_RE = re.compile(
    r"timed?\s?out|timeout|connect|reset|eof|bad gateway|unavailable|"
    r"overloaded|deadline|internal|temporar|\b50[0-4]\b", re.I)

# Vision self-check model — the cheap text/vision lane ('summary' class
# on the gateway), not the image model itself.
_CHECK_MODEL = "gemini-3.5-flash-lite"

# Aspect-intent notices — page-ish and pixel-exact asks get honest
# capability notes attached to the result (never a refusal).
_PAGE_ASK_RE = re.compile(
    r"landing\s*page|web\s*site|web\s*page|home\s*page|full[\s-]*page|"
    r"one[\s-]*pager|splash\s*page|site\s+mockup|wireframe", re.I)
_PIXEL_ASK_RE = re.compile(r"\b\d{3,5}\s*(?:px|pixels?)\b", re.I)

# provider='local' — sd-server on tony-omen (stable-diffusion.cpp,
# SDXL-Lightning 4-step Q4). SDXL-native geometry per aspect; lightning
# wants few steps at low CFG.
_LOCAL_BASE = "http://100.75.102.88:8399"
_LOCAL_WAIT_S = 45.0
_LOCAL_POLL_S = 5.0
_LOCAL_DEADLINE_S = 1800.0
_LOCAL_STEPS = 4
_LOCAL_CFG = 1.0
_LOCAL_SAMPLER = "euler_a"
_LOCAL_DIMS = {
    "1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
    "4:3": (1152, 864), "3:4": (864, 1152),
}
# Job ids with an armed background-delivery task — resume calls re-arm
# only when the earlier task died (e.g. a restart between processes).
_LOCAL_DELIVERING: set[str] = set()


def _model() -> str:
    return os.environ.get("ADA_RENDER_IMAGE_MODEL") or _DEFAULT_MODEL


def _self_base() -> str:
    return (os.environ.get("ADA_SELF_URL")
            or f"http://127.0.0.1:{os.environ.get('ADA_PORT') or '8000'}"
            ).rstrip("/")


def _aspect(raw: Any) -> tuple[str | None, str | None]:
    """-> (ratio, error). Blank defaults to square."""
    val = str(raw or "").strip().lower()
    if not val:
        return "1:1", None
    if val in _ASPECT_RATIOS:
        return val, None
    if val in _ASPECT_ALIASES:
        return _ASPECT_ALIASES[val], None
    m = re.fullmatch(r"(\d{1,2})\s*[:x×]\s*(\d{1,2})", val)
    if m and f"{m.group(1)}:{m.group(2)}" in _ASPECT_RATIOS:
        return f"{m.group(1)}:{m.group(2)}", None
    return None, (
        f"unknown aspect {raw!r} — image renders square (1:1), "
        "landscape (16:9), portrait (9:16), 4:3 or 3:4")


def _size(raw: Any) -> tuple[str | None, str | None]:
    """-> (size, error). Blank defaults to 1K."""
    val = str(raw or "").strip()
    if not val:
        return "1K", None
    if val.upper() in _SIZES:
        return val.upper(), None
    if val.lower() in _SIZE_ALIASES:
        return _SIZE_ALIASES[val.lower()], None
    return None, (
        f"unknown size {raw!r} — image renders at 1K (default), "
        "2K or 4K")


def _gate_closed() -> str | None:
    """Owner kill-switch — a file in ADA_GATES_DIR closes the tool
    instantly, no restart: <tool>.closed gates every lane,
    <tool>.<ADA_INSTANCE_ID>.closed gates one lane."""
    d = Path(os.path.expanduser(
        os.environ.get("ADA_GATES_DIR") or "~/.local/share/ada/gates"))
    lane = os.environ.get("ADA_INSTANCE_ID") or ""
    names = ["ada_render_image.closed"]
    if lane:
        names.append(f"ada_render_image.{lane}.closed")
    if any((d / n).exists() for n in names):
        return ("image rendering is switched off by the owner right "
                "now — tell the user it's temporarily closed")
    return None


def _ref_from_engine(key: str) -> tuple[bytes, str] | None:
    """Resolve a held intake key in-process — same singleton the video
    tool uses."""
    try:
        from backend import document_check
        held = document_check.engine().held(key)
    except Exception as exc:
        logger.info("ref resolve %r: engine lookup failed: %s", key, exc)
        return None
    if held is None:
        return None
    jpg = (held.meta or {}).get("archive_jpg")
    if isinstance(jpg, bytes) and jpg:
        return jpg, "image/jpeg"
    if isinstance(held.preview, bytes) and held.preview:
        return held.preview, "image/jpeg"
    return None


async def _resolve_ref(client: httpx.AsyncClient, ref: str
                       ) -> tuple[tuple[bytes, str] | None, str | None]:
    """-> ((bytes, mime), None) or (None, error)."""
    ref = ref.strip()
    doc = _DOC_PATH_RE.search(ref) if "/" in ref else None
    if doc:
        key = urllib.parse.unquote(doc.group(1))
        hit = _ref_from_engine(key)
        if hit:
            return hit, None
        if not ref.startswith(("http://", "https://")):
            return None, (
                f"no held document for key {key!r} — intake results live "
                "in RAM only, ask the user to upload it again")
    elif _DOC_KEY_RE.match(ref):
        hit = _ref_from_engine(ref)
        if hit:
            return hit, None
        return None, (
            f"no held document for key {ref!r} — intake results live in "
            "RAM only, ask the user to upload it again")
    elif ref.startswith("/"):
        ref = _self_base() + ref
    if not ref.startswith(("http://", "https://")):
        return None, (
            "ref_url must be a 'doc/...' intake key, a /api/documents/... "
            "path, or an http(s) image URL")
    headers = {}
    host = urllib.parse.urlparse(ref).hostname or ""
    key = os.environ.get("ADA_API_KEY", "")
    if key and host in ("127.0.0.1", "localhost", "::1"):
        headers["x-api-key"] = key
    try:
        resp = await client.get(
            ref, headers=headers, follow_redirects=True, timeout=30)
    except (httpx.HTTPError, TimeoutError) as exc:
        return None, (f"I couldn't fetch the reference image "
                      f"({exc.__class__.__name__})")
    if resp.status_code >= 400:
        return None, f"the reference image URL returned HTTP {resp.status_code}"
    mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        return None, f"the reference URL isn't an image (got {mime or 'unknown'})"
    if len(resp.content) > _REF_MAX_BYTES:
        return None, "the reference image is too large (15MB max)"
    return (resp.content, mime), None


def _gen_config(ratio: str, size: str) -> Any:
    """responseModalities + aspect/size — types first, dict fallback for
    old SDK builds."""
    try:
        from google.genai import types
        return types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=ratio, image_size=size))
    except Exception:
        return {"response_modalities": ["TEXT", "IMAGE"],
                "image_config": {"aspect_ratio": ratio,
                                 "image_size": size}}


async def _render(prompt: str, refs: list[tuple[bytes, str]],
                  ratio: str, size: str, model: str | None = None
                  ) -> tuple[bytes | None, str | None, str | None]:
    """-> (image bytes, model text, None) or (None, None, error)."""
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for data, mime in refs:
        parts.append({"inline_data": {
            "mime_type": mime, "data": base64.b64encode(data).decode()}})
    contents = [{"role": "user", "parts": parts}]
    from backend import gemini_pool
    resp = await gemini_pool.generate(
        model or _model(), contents, config=_gen_config(ratio, size),
        tool="render_image")
    text_bits: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        for part in (getattr(cand, "content", None) and
                     getattr(cand.content, "parts", None)) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data, " ".join(text_bits).strip() or None, None
            t = getattr(part, "text", None)
            if t:
                text_bits.append(str(t))
    reason = None
    try:
        reason = getattr(resp, "prompt_feedback", None)
    except Exception:
        pass
    return None, (" ".join(text_bits).strip() or None), (
        "the image model answered with no image — rephrase the prompt "
        "and try once more" + (f" (feedback: {reason})" if reason else ""))


def _save(data: bytes, mime: str) -> Path:
    ext = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
    d = Path(os.path.expanduser(
        os.environ.get("ADA_RENDER_DIR") or _DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    path = d / (f"render-img-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                f"-{secrets.token_hex(3)}{ext}")
    path.write_bytes(data)
    return path


def _transient(exc: BaseException) -> bool:
    """True when the model call failed on transport/gateway wobble and
    earns one retry. Quota-shaped errors are excluded — the pool already
    rotated keys, and a drained pool is a policy answer, not a blip."""
    try:
        from backend import gemini_pool
        if gemini_pool.is_quota_error(exc):
            return False
    except Exception:
        pass
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and v >= 500:
            return True
        if str(v or "").upper() in ("UNAVAILABLE", "INTERNAL",
                                    "DEADLINE_EXCEEDED"):
            return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return bool(_TRANSIENT_RE.search(str(exc)))


async def _render_retry(prompt: str, refs: list[tuple[bytes, str]],
                        ratio: str, size: str
                        ) -> tuple[bytes | None, str | None, str | None]:
    """Render with a bounded retry and explicit image-model fallbacks."""
    models = [_model()] + [
        model for model in _IMAGE_FALLBACK_MODELS if model != _model()
    ]
    last_error = "the image model returned no usable image"
    last_text: str | None = None
    for model in models:
        for attempt in range(2):
            try:
                data, text, error = await _render(
                    prompt, refs, ratio, size, model=model)
                if data:
                    return data, text, None
                last_text = text or last_text
                last_error = error or last_error
                if attempt == 0 and not error:
                    continue
                break
            except Exception as exc:
                last_error = f"the image model call failed: {exc}"
                if attempt == 0 and _transient(exc):
                    logger.info(
                        "image render transient on %s (%s) — one retry",
                        model, exc)
                    await asyncio.sleep(
                        _env_float("ADA_RENDER_RETRY_S", 2.0))
                    continue
                if not _transient(exc):
                    return None, None, last_error
                break
        if model != models[-1]:
            logger.info(
                "image model %s returned no usable image; trying fallback",
                model)
    return None, last_text, last_error


def _check_model() -> str:
    return os.environ.get("ADA_RENDER_CHECK_MODEL") or _CHECK_MODEL


async def _selfcheck(data: bytes, mime: str, prompt: str
                     ) -> tuple[str, str]:
    """One cheap vision QA pass comparing the render against the ask.
    -> ('miss', detail) only on an obvious miss; every failure mode —
    the check model itself erroring, an unparseable verdict — fails
    open to 'pass'/'skip' so QA can never sink a good render."""
    if os.environ.get("ADA_RENDER_IMAGE_SELFCHECK", "1").lower() in (
            "0", "false", "no", "off"):
        return "skip", ""
    ask = prompt[:600]
    q = ("One-line QA on an AI image render. The request was:\n"
         f'"{ask}"\n'
         "Reply with exactly one line: PASS when the image plausibly "
         "matches the request, or 'MISS: <under 10 words>' only for an "
         "obvious miss — wrong subject, requested text garbled or "
         "illegible, blank or broken image. Judge content, not style.")
    parts = [{"text": q}, {"inline_data": {
        "mime_type": mime, "data": base64.b64encode(data).decode()}}]
    try:
        from backend import gemini_pool
        resp = await gemini_pool.generate(
            _check_model(), [{"role": "user", "parts": parts}],
            tool="render_check", timeout=25)
    except Exception as exc:
        logger.info("render selfcheck failed-open: %s", exc)
        return "skip", ""
    text_bits: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        for part in (getattr(cand, "content", None) and
                     getattr(cand.content, "parts", None)) or []:
            t = getattr(part, "text", None)
            if t:
                text_bits.append(str(t))
    head = (" ".join(text_bits).strip().splitlines() or [""])[0]
    m = re.match(r"(?i)\s*(?:miss|fail|no)\b[:\- ]*(.*)", head)
    if m:
        return "miss", (m.group(1).strip()[:120] or "obvious miss")
    return "pass", ""


def _arch_publish(data: bytes, name: str) -> str | None:
    """Durable original-quality URL via the LINE relay media archive —
    the same hop ada_render_video uses. The /frame URL + a LINE
    imageMessage both re-encode, so 2K/4K renders land here and the
    member grabs the full-res file via full_url."""
    arch = Path(os.path.expanduser(
        os.environ.get("LINE_MEDIA_ARCHIVE_DIR")
        or "~/.local/share/ada/line-media"))
    try:
        d = arch / "renders"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    except OSError as exc:
        logger.warning("arch publish failed: %s", exc)
        return None
    base = (os.environ.get("ADA_RELAY_PUBLIC_BASE")
            or os.environ.get("LINE_PUBLIC_IMG_BASE")
            or "https://157.85.102.125.sslip.io/relay-img").rstrip("/")
    return f"{base}/arch/renders/{name}"


def _ask_notices(prompt: str) -> str:
    """Honest-capability notes attached to the result — the model relays
    them; they never block a render."""
    notes = []
    if _PAGE_ASK_RE.search(prompt):
        notes.append(
            "this is a *picture* of a page at one fixed aspect — not a "
            "working page; if the user wants a usable page, offer to "
            "render the hero/section art here and assemble the page "
            "with the page/CMS tools instead")
    if _PIXEL_ASK_RE.search(prompt):
        notes.append(
            "generation is aspect-based (1:1/16:9/9:16/4:3/3:4), not "
            "pixel-exact — 1K is ~1024px on the short side; say so if "
            "they needed an exact size")
    return " ".join(notes)


def _remember(runner: Any, url: str, path: Path | None, prompt: str,
              ratio: str, size: str, provider: str) -> None:
    """Store this render as the caller's last artifact — feeds the
    'last' ref alias and the edit-cue auto-feed on later turns."""
    try:
        from backend import render_state
        render_state.remember(runner, {
            "kind": "image", "url": url, "file": str(path or ""),
            "prompt": prompt[:200], "aspect": ratio, "size": size,
            "provider": provider})
    except Exception as exc:
        logger.info("render remember skipped: %s", exc)


def _provider(raw: Any) -> tuple[str | None, str | None]:
    """-> ('gemini'|'local', error). Blank defaults to gemini."""
    val = str(raw or "").strip().lower()
    if not val:
        return "gemini", None
    if val in ("gemini", "nano-banana", "nanobanana", "cloud"):
        return "gemini", None
    if val in ("local", "sd", "sd-server", "sdcpp", "omen"):
        return "local", None
    return None, (f"unknown provider {raw!r} — image renders on "
                  "'gemini' (default, cloud) or 'local' (tony-omen "
                  "sd-server, ~10min CPU render)")


def _sd_base() -> str:
    return (os.environ.get("ADA_SD_SERVER_URL")
            or _LOCAL_BASE).rstrip("/")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


async def _sd_submit(client: httpx.AsyncClient, prompt: str,
                     ratio: str) -> tuple[str | None, int, str | None]:
    """POST /sdcpp/v1/img_gen -> (job id, queue position, None)|(None,0,err)."""
    w, h = _LOCAL_DIMS.get(ratio, _LOCAL_DIMS["1:1"])
    body = {
        "prompt": prompt,
        "width": w,
        "height": h,
        "seed": -1,
        "sample_params": {
            "sample_steps": _LOCAL_STEPS,
            "sample_method": _LOCAL_SAMPLER,
            "guidance": {"txt_cfg": _LOCAL_CFG},
        },
        "output_format": "png",
    }
    try:
        resp = await client.post(f"{_sd_base()}/sdcpp/v1/img_gen",
                                 json=body, timeout=30)
    except (httpx.HTTPError, TimeoutError) as exc:
        return None, 0, (f"the local render server didn't answer "
                         f"({exc.__class__.__name__}) — sd-server on "
                         "tony-omen may be down")
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    if resp.status_code >= 400:
        msg = (payload.get("error") or {}).get("message") \
            if isinstance(payload, dict) else None
        return None, 0, (f"the local render server refused the job "
                         f"(HTTP {resp.status_code}"
                         f"{': ' + str(msg)[:120] if msg else ''})")
    job_id = payload.get("id") if isinstance(payload, dict) else None
    if not job_id:
        return None, 0, ("the local render server accepted the job but "
                         "returned no job id — can't track it")
    try:
        qpos = int(payload.get("queue_position") or 0)
    except (TypeError, ValueError):
        qpos = 0
    return str(job_id), qpos, None


async def _sd_job(client: httpx.AsyncClient, job_id: str
                  ) -> tuple[dict[str, Any] | None, str | None]:
    """GET /sdcpp/v1/jobs/{id} -> (payload, None)|(None, err)."""
    try:
        resp = await client.get(f"{_sd_base()}/sdcpp/v1/jobs/{job_id}",
                                timeout=20)
    except (httpx.HTTPError, TimeoutError) as exc:
        return None, (f"lost contact with the local render "
                      f"({exc.__class__.__name__})")
    if resp.status_code == 404:
        return None, ("the local render server doesn't know job "
                      f"{job_id!r} — it may have restarted and dropped "
                      "the queue; submit a fresh render")
    if resp.status_code >= 400:
        return None, (f"the local render server answered HTTP "
                      f"{resp.status_code} while polling {job_id}")
    try:
        payload = resp.json()
    except Exception:
        return None, "the local render server returned a malformed job"
    return payload, None


def _sd_result_bytes(payload: dict[str, Any]
                     ) -> tuple[bytes | None, str | None]:
    """Completed-job payload -> (png bytes, None)|(None, error)."""
    result = payload.get("result")
    candidates: list[Any] = []
    if isinstance(result, dict):
        candidates.extend(result.get("data") or [])
        candidates.extend(result.get("images") or [])
        candidates.append(result.get("image"))
    candidates.extend(payload.get("data") or [])
    for item in candidates:
        if isinstance(item, str) and item:
            try:
                return base64.b64decode(item), None
            except Exception:
                continue
        if isinstance(item, dict):
            b64 = item.get("b64_json") or item.get("b64") \
                or item.get("image")
            if isinstance(b64, str) and b64:
                try:
                    return base64.b64decode(b64), None
                except Exception:
                    continue
    return None, ("the local render finished but the job carried no "
                  "image payload")


def _local_pending(job_id: str, queue_position: int,
                   resumed: bool = False) -> dict[str, Any]:
    eta = "~10 minutes" if queue_position <= 0 else (
        f"~{10 * (queue_position + 1)} minutes — {queue_position} "
        f"render{'s' if queue_position != 1 else ''} ahead in the "
        "single-slot queue")
    return {
        "ok": False,
        "pending": True,
        "job": job_id,
        "queue_position": queue_position,
        "provider": "local",
        "error": (
            f"the image is still rendering locally ({eta} — tony-omen "
            "is CPU-only). Tell the user it's rendering and the photo "
            "will arrive in the chat when done — a background task is "
            "already watching the job and pushes it via chat_send. "
            f"Call again with job={job_id!r} only if asked for a status "
            "check; never resubmit the same prompt."),
    }


def _arm_local_delivery(runner: Any, job_id: str, prompt: str) -> None:
    """Background task: poll to completion, publish, chat_send the photo."""
    if job_id in _LOCAL_DELIVERING:
        return
    _LOCAL_DELIVERING.add(job_id)
    task = asyncio.create_task(_deliver_local(runner, job_id, prompt))

    def _done(t: asyncio.Task) -> None:
        _LOCAL_DELIVERING.discard(job_id)
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            logger.warning("local render delivery %s failed: %s",
                           job_id, exc)
    task.add_done_callback(_done)


async def _deliver_local(runner: Any, job_id: str, prompt: str) -> None:
    """Poll a queued/generating local job; on success push the published
    image to the originating chat, on failure push a short apology."""
    deadline = asyncio.get_running_loop().time() + _env_float(
        "ADA_RENDER_LOCAL_DEADLINE_S", _LOCAL_DEADLINE_S)
    channel = "line"
    try:
        from backend.tool_runner.common import session_chat_id
        if session_chat_id(runner, "telegram"):
            channel = "telegram"
    except Exception:
        pass
    status = ""
    timed_out = True
    payload: dict[str, Any] | None = None
    poll_s = max(
        _env_float("ADA_RENDER_LOCAL_POLL_S", _LOCAL_POLL_S) * 3, 15.0)
    async with httpx.AsyncClient() as client:
        while asyncio.get_running_loop().time() < deadline:
            payload, err = await _sd_job(client, job_id)
            if payload is None:
                logger.info("local job %s poll: %s", job_id, err)
            else:
                status = str(payload.get("status") or "")
                if status in ("completed", "failed", "cancelled"):
                    timed_out = False
                    break
            await asyncio.sleep(poll_s)
    if timed_out:
        status = ""
    text = f"local render {job_id}: "
    image_url = ""
    if status == "completed":
        data, derr = _sd_result_bytes(payload or {})
        if data:
            path: Path | None = None
            try:
                path = _save(data, "image/png")
            except OSError as exc:
                logger.warning("local render save failed: %s", exc)
            slug = re.sub(r"[^a-z0-9]+", "-",
                          (prompt or "image").lower())[:40] \
                .strip("-") or "image"
            from backend import traffic_camera
            url_out = await asyncio.to_thread(
                traffic_camera.publish_relay, data, slug,
                "image/png", "render")
            if url_out:
                image_url = url_out
                _remember(runner, url_out, path, prompt, "", "1K",
                          "local")
                text += f"done — {(prompt or 'your image')[:80]}"
            else:
                text += ("finished, but publishing a shareable URL "
                         "failed")
        else:
            text += derr or "finished without an image"
    elif status == "cancelled":
        text += "was cancelled"
    elif status == "failed":
        jerr = ((payload or {}).get("error") or {})
        jmsg = jerr.get("message") if isinstance(jerr, dict) else None
        text += f"failed{': ' + str(jmsg)[:120] if jmsg else ''}"
    else:
        text += ("stopped responding — the render may still finish "
                 "server-side; ask me to check the job id")
    try:
        await runner.chat_send(channel=channel, text=text,
                               image_url=image_url)
    except Exception as exc:
        logger.warning("local render chat_send %s failed: %s",
                       job_id, exc)


async def _finish_local(runner: Any, client: httpx.AsyncClient,
                        job_id: str, prompt: str, ratio: str,
                        size: str, resumed: bool = False
                        ) -> dict[str, Any]:
    """Bounded foreground poll; pending + armed delivery when it
    outlives the wait budget."""
    deadline = asyncio.get_running_loop().time() + _env_float(
        "ADA_RENDER_LOCAL_WAIT_S", _LOCAL_WAIT_S)
    while True:
        payload, err = await _sd_job(client, job_id)
        if err:
            return {"ok": False, "error": err, "job": job_id,
                    "pending": True}
        assert payload is not None
        status = str(payload.get("status") or "")
        qpos = int(payload.get("queue_position") or 0)
        if status == "failed":
            jerr = (payload.get("error") or {})
            jmsg = jerr.get("message") if isinstance(jerr, dict) else None
            return {"ok": False, "job": job_id, "error": (
                f"the local render failed"
                f"{': ' + str(jmsg)[:160] if jmsg else ''}")}
        if status == "cancelled":
            return {"ok": False, "job": job_id,
                    "error": "the local render was cancelled"}
        if status == "completed":
            break
        if asyncio.get_running_loop().time() >= deadline:
            _arm_local_delivery(runner, job_id, prompt)
            return _local_pending(job_id, qpos, resumed=resumed)
        await asyncio.sleep(
            _env_float("ADA_RENDER_LOCAL_POLL_S", _LOCAL_POLL_S))
    data, derr = _sd_result_bytes(payload)
    if not data:
        return {"ok": False, "job": job_id,
                "error": derr or "the local render returned no image"}
    if len(data) > _IMG_MAX_BYTES:
        return {"ok": False, "error": "the rendered image is too large"}
    try:
        path = _save(data, "image/png")
    except OSError as exc:
        logger.warning("render save failed: %s", exc)
        return {"ok": False,
                "error": f"the image rendered but saving it failed: {exc}"}
    slug = re.sub(r"[^a-z0-9]+", "-", (prompt or "image").lower())[:40] \
        .strip("-") or "image"
    from backend import traffic_camera
    url_out = await asyncio.to_thread(
        traffic_camera.publish_relay, data, slug, "image/png", "render")
    if not url_out:
        return {"ok": False,
                "file": str(path),
                "job": job_id,
                "error": "the image rendered and was saved, but publishing "
                         "a shareable URL failed (input-bridge /frame) — "
                         "the artifact is kept at 'file'"}
    _remember(runner, url_out, path, prompt, ratio, "1K", "local")
    out: dict[str, Any] = {
        "ok": True,
        "image_url": url_out,
        "title": prompt[:60],
        "file": str(path),
        "aspect": ratio,
        "size": "1K",
        "provider": "local",
        "model": "sd-cpp-local (sdxl-lightning-4step, tony-omen)",
        "job": job_id,
        "bytes": len(data),
        "note": "image_url is the shareable render — relays post it as a "
                "photo automatically; 'file' is the local artifact.",
    }
    if size != "1K":
        out["note"] += (f" (asked for {size} — the local provider tops "
                        "out at 1K/1024px)")
    notice = _ask_notices(prompt)
    if notice:
        out["notice"] = notice
    return out


async def run(runner: Any, **args: Any) -> dict[str, Any]:
    closed = _gate_closed()
    if closed:
        return {"ok": False, "error": closed}
    prompt = str(args.get("prompt") or "").strip()
    prov, err = _provider(args.get("provider"))
    if err:
        return {"ok": False, "error": err}
    assert prov is not None
    job = str(args.get("job") or "").strip()
    if job:
        if prov != "local":
            return {"ok": False,
                    "error": "job= only applies to provider='local' — "
                             "gemini renders finish inside the call"}
        if not _JOB_RE.match(job):
            return {"ok": False,
                    "error": "job must be the 'job_...' id a pending "
                             "local render returned"}
    elif not prompt:
        return {"ok": False,
                "error": "a prompt describing the image is required"}
    prompt = prompt[:_PROMPT_MAX]
    ratio, err = _aspect(args.get("aspect"))
    if err:
        return {"ok": False, "error": err}
    assert ratio is not None
    size, err = _size(args.get("size"))
    if err:
        return {"ok": False, "error": err}
    assert size is not None
    refs_in: list[str] = []
    ref_url = str(args.get("ref_url") or "").strip()
    if ref_url:
        refs_in.append(ref_url)
    extra = args.get("ref_urls")
    if isinstance(extra, str):
        extra = [extra]
    if isinstance(extra, (list, tuple)):
        refs_in += [str(r or "").strip() for r in extra]
    refs_in = [r for r in refs_in if r]
    if len(refs_in) > _REFS_MAX:
        return {"ok": False, "error": (
            f"up to {_REFS_MAX} reference images per render — got "
            f"{len(refs_in)}")}
    from backend import render_state
    auto_ref = False
    if not refs_in and prov == "gemini" and prompt:
        ent = render_state.last(runner, kinds={"image"},
                                max_age_s=render_state.AUTO_TTL_S)
        if ent and len(prompt) <= _AUTO_REF_PROMPT_MAX \
                and _edit_cued(prompt):
            refs_in = ["last"]
            auto_ref = True
    for r in refs_in:
        if r.lower() in render_state.LAST_ALIASES:
            continue
        if not (_DOC_KEY_RE.match(r) or _DOC_PATH_RE.search(r)
                or r.startswith(("/", "http://", "https://"))):
            return {"ok": False,
                    "error": "ref_url/ref_urls entries must be 'last', "
                             "a 'doc/...' intake key, a "
                             "/api/documents/... path, or an http(s) "
                             "image URL"}
    if prov == "local":
        if refs_in:
            return {"ok": False,
                    "error": "the local provider is text-to-image only "
                             "— it can't edit a ref photo. Drop "
                             "provider='local' to edit with the default "
                             "model instead"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            if not job:
                job, qpos, err = await _sd_submit(client, prompt, ratio)
                if err:
                    return {"ok": False, "error": err}
                assert job is not None
                logger.info("local render submitted: %s", job)
            return await _finish_local(runner, client, job, prompt,
                                       ratio, size, resumed=True)
    try:
        from backend import gemini_pool
        if not (gemini_pool.configured_keys()
                or os.environ.get("ADA_GEMINI_REST_BASE_URL")):
            return {"ok": False,
                    "error": "image rendering isn't configured — "
                             "GEMINI_API_KEY is missing"}
    except Exception:
        pass

    refs: list[tuple[bytes, str]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for raw in refs_in:
            if raw.lower() in render_state.LAST_ALIASES:
                resolved, rerr = render_state.resolve_last(
                    runner, kinds={"image"})
                if rerr:
                    return {"ok": False, "error": rerr}
                assert resolved is not None
                if resolved[0] == "bytes":
                    refs.append(resolved[1])
                    continue
                raw = str(resolved[1])
            ref, rerr = await _resolve_ref(client, raw)
            if rerr:
                return {"ok": False, "error": rerr}
            assert ref is not None
            refs.append(ref)

    data, model_text, rerr = await _render_retry(prompt, refs,
                                               ratio, size)
    if rerr:
        return {"ok": False, "error": rerr,
                "model_text": model_text or ""}
    assert data is not None

    selfcheck: dict[str, Any] = {"verdict": "skip"}
    verdict, detail = await _selfcheck(data, "image/png", prompt)
    selfcheck["verdict"] = verdict
    if verdict == "miss":
        fix = prompt[:_PROMPT_MAX - 160] + (
            f" — previous attempt missed: {detail}; fix that" if detail
            else " — previous attempt missed the request; fix it")
        d2, m2, e2 = await _render_retry(fix, refs, ratio, size)
        selfcheck["retried"] = True
        selfcheck["detail"] = detail
        if d2:
            data, model_text = d2, m2
            selfcheck["retry"] = "re-rendered with the critique"
        else:
            selfcheck["retry"] = ("retry failed — keeping the first "
                                  "render; warn the user it may be off")
            if e2:
                logger.info("selfcheck retry render failed: %s", e2)
    if len(data) > _IMG_MAX_BYTES:
        return {"ok": False, "error": "the rendered image is too large"}
    mime = "image/png"
    try:
        path = _save(data, mime)
    except OSError as exc:
        logger.warning("render save failed: %s", exc)
        return {"ok": False,
                "error": f"the image rendered but saving it failed: {exc}"}
    slug = re.sub(r"[^a-z0-9]+", "-", (prompt or "image").lower())[:40] \
        .strip("-") or "image"
    from backend import traffic_camera
    url_out = await asyncio.to_thread(
        traffic_camera.publish_relay, data, slug, mime, "render")
    if not url_out:
        return {"ok": False,
                "file": str(path),
                "error": "the image rendered and was saved, but publishing "
                         "a shareable URL failed (input-bridge /frame) — "
                         "the artifact is kept at 'file'; a retry renders "
                         "a fresh image"}
    _remember(runner, url_out, path, prompt, ratio, size, "gemini")
    out: dict[str, Any] = {
        "ok": True,
        "image_url": url_out,
        "title": prompt[:60],
        "file": str(path),
        "aspect": ratio,
        "size": size,
        "model": _model(),
        "bytes": len(data),
        "selfcheck": selfcheck,
        "note": "image_url is the shareable render — relays post it as a "
                "photo automatically; 'file' is the local artifact. To "
                "adjust it, call again with ref_url='last' — an edit "
                "keeps the composition, a fresh call rerolls.",
    }
    if auto_ref:
        out["auto_ref"] = "last"
    if size in ("2K", "4K"):
        full = _arch_publish(data, path.name)
        if full:
            out["full_url"] = full
            out["note"] += (" full_url serves the original bytes — "
                            "LINE re-encodes the attached photo, so "
                            "share the link for the full-res file.")
    notice = _ask_notices(prompt)
    if notice:
        out["notice"] = notice
    if model_text:
        out["model_text"] = model_text[:400]
    return out

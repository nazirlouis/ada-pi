"""Headless Devin session dispatch.

Two launch paths:

- no playbook — the tuned path: files an armed kanban card through
  board-api POST /card (action dispatch, queue:true); kanban-dispatch
  claims it and runs the session on a runner. No ssh fallback: the
  board IS the entry point, so an unreachable board is an error, not
  a bypass.
- playbook=<name> — ssh `devin-dispatch start` on tony-dell
  (BatchMode); each task runs as a systemd --user unit in a dedicated
  git worktree, gated + stamped per the playbook contract.

Completion is reported back by the dispatch watcher (chaba-admin event +
iPhone notify + devin-bank outcome doc), so these tools only deal in
launch/status/followup — never wait for a session to finish.

SSOT: docs/ssot/jobs/ada/2026-09-22-ada-devin-dispatch.yml
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend import board_client

logger = logging.getLogger("tools.devin_dispatch")

# Tailnet name, not the LAN IP: idc03 (Ada's current home) cannot reach
# 192.168.2.67, while `ssh tony-dell` resolves via MagicDNS to the tailscale
# address where plain sshd answers — tony-dell's tailscaled runs with
# RunSSH:false so it no longer intercepts :22, and the old re-auth concern
# that forced the LAN IP (2026-09-24) is moot. ssh configs on
# mn01/idc03/tony-omen already map tony-dell -> IdentityFile.
# Supersedes docs/ssot/jobs/infrastructure/2026-10-10-dispatch-ssh-tailnet.yml
HOST = os.environ.get("ADA_DEVIN_DISPATCH_HOST", "tony-dell")
BIN = os.environ.get("ADA_DEVIN_DISPATCH_BIN", "~/.local/bin/devin-dispatch")
TIMEOUT_S = float(os.environ.get("ADA_DEVIN_DISPATCH_TIMEOUT_S", "45"))

# List-status output cap: the full registry is noise in a live-voice context
# and every tool result stays in the Gemini session, inflating turn latency.
STATUS_MAX_LINES = int(os.environ.get("ADA_DEVIN_DISPATCH_STATUS_MAX", "8"))

# In-process dedup: the model retries a failed/slow dispatch with rephrased
# task text (seen 2026-09-24: 3 calls in ~2min spawned 3 sessions). Recent
# dispatches are cached; a near-identical task inside the window reuses the
# existing task_id instead of starting a duplicate session.
DEDUP_WINDOW_S = float(os.environ.get("ADA_DEVIN_DISPATCH_DEDUP_S", "600"))
# Containment |∩|/min(|a|,|b|): a retry is often a *shorter* rephrase of the
# same task, which Jaccard under-scores.
DEDUP_MIN_SIM = float(os.environ.get("ADA_DEVIN_DISPATCH_DEDUP_SIM", "0.6"))

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "been", "into", "then", "than", "them", "they", "what", "when",
    "check", "look", "please", "investigate", "find", "tell", "make",
}


def _task_tokens(text: str) -> frozenset[str]:
    return frozenset(
        w for w in _TOKEN_RE.findall(text.lower())
        if len(w) > 3 and w not in _STOP
    )


# repo -> [(tokens, ref_id, is_card, monotonic_ts)] — ref_id is the
# devin-dispatch task id for playbook sessions, the board card id for
# voice-dispatch cards.
_recent_dispatches: dict[
    str, list[tuple[frozenset[str], str, bool, float]]] = {}

# devin-dispatch task ids: YYYYMMDD-HHMMSS-<30-char slug>
_TASK_ID_RE = re.compile(r"^(\d{8})-(\d{6})-([a-z0-9-]+)$")


def _dedup_result(ref_id: str, repo: str, *,
                  card_id: str | None = None) -> dict:
    if card_id:
        return {
            "ok": True,
            "card_id": card_id,
            "repo": repo,
            "deduplicated": True,
            "note": (
                "A matching card was already queued on the board "
                "recently; reusing it instead of filing a duplicate. "
                "kanban-dispatch will claim and run it."
            ),
        }
    return {
        "task_id": ref_id,
        "repo": repo,
        "deduplicated": True,
        "note": (
            "A matching task was already dispatched recently; reusing "
            "that session instead of starting a duplicate. Use "
            "devin_read action='status' to check progress."
        ),
    }


async def _remote_dupe(repo: str, toks: frozenset[str]) -> str | None:
    """Check the dispatch host's task list for a same-intent task started
    inside the dedup window. Covers retries after a backend restart, where
    the in-process cache is empty."""
    try:
        out = await _run("status")
    except Exception as exc:
        logger.info("dedup remote check skipped: %s", exc)
        return None
    now = datetime.now(timezone.utc)
    for line in out.splitlines():
        fields = line.split()
        if not fields or f"repo={repo}" not in fields:
            continue
        m = _TASK_ID_RE.match(fields[0])
        if not m:
            continue
        try:
            started = datetime.strptime(
                m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if (now - started).total_seconds() > DEDUP_WINDOW_S:
            continue
        slug_toks = _task_tokens(m.group(3).replace("-", " "))
        if toks and slug_toks:
            sim = len(toks & slug_toks) / min(len(toks), len(slug_toks))
            if sim >= DEDUP_MIN_SIM:
                return fields[0]
    return None


async def _run(*args: str) -> str:
    """Run `devin-dispatch <args>` on the dispatch host. Returns stdout."""
    # ssh re-joins the command into a remote `bash -c` string — every
    # argument must be shell-quoted or task text containing ( ) ; ' etc.
    # breaks the parse (seen 2026-09-24). BIN stays unquoted so the remote
    # shell expands its leading `~`.
    remote_cmd = " ".join([BIN, *(shlex.quote(a) for a in args)])
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        HOST, remote_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"devin-dispatch {args[0]} timed out after {TIMEOUT_S}s")
    if proc.returncode != 0:
        msg = err.decode(errors="replace").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"devin-dispatch {args[0]} failed: {msg}")
    return out.decode(errors="replace").strip()


# -- Playbook registry (backend/playbooks/*.yml) ------------------------------
#
# A playbook is a named dispatch contract: declared params feed a
# task_template, a gate block states the confirm/identity requirements,
# `requires` names bank-side preconditions (today: the build-tool spec
# doc must carry status=pass), verify lists the commands the session runs
# before finishing, and report_to / max_runtime_min bound the job. A
# dispatched playbook job is stamped into the devin-handoff ledger as
# job/<task_id>/contract so devin-dispatch-watch can pick up the verify
# list (kind=job-contract — inside the job/ namespace so the handoff
# inbox skips it, and not kind=job so report feeds don't count it as a
# job row).

PLAYBOOKS_DIR = Path(
    os.environ.get("ADA_PLAYBOOKS_DIR")
    or Path(__file__).resolve().parent / "playbooks"
)
HANDOFF_COLLECTION = os.environ.get(
    "ADA_DEVIN_HANDOFF_COLLECTION", "ada-ha-bank-devin-handoff")

_PLAYBOOK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_VALID_GATE_IDENTITIES = {"owner", "identified", "any"}


class PlaybookError(ValueError):
    """Bad playbook file or bad playbook call — reported to the caller."""


def list_playbooks(playbooks_dir: Path | None = None) -> list[str]:
    root = Path(playbooks_dir) if playbooks_dir else PLAYBOOKS_DIR
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.yml")
                  if _PLAYBOOK_NAME_RE.match(p.stem))


def _template_fields(template: str) -> set[str]:
    return {
        field for _, field, _, _ in string.Formatter().parse(template)
        if field
    }


def _validate_playbook(path: Path, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise PlaybookError(f"{path.name}: playbook file must be a mapping")
    pb = dict(data)
    name = str(pb.get("name") or path.stem)
    if name != path.stem:
        raise PlaybookError(
            f"{path.name}: name {name!r} != filename {path.stem!r}")
    for field in ("description", "task_template", "report_to"):
        if not isinstance(pb.get(field), str) or not pb[field].strip():
            raise PlaybookError(f"{path.name}: missing non-empty '{field}'")
    params = pb.get("params")
    if not isinstance(params, dict) or not params:
        raise PlaybookError(f"{path.name}: 'params' must be a mapping")
    for pname, cfg in params.items():
        if not isinstance(cfg, dict):
            raise PlaybookError(
                f"{path.name}: param {pname!r} must be a mapping "
                "(required/description/line)")
    gate = pb.get("gate")
    if not isinstance(gate, dict):
        raise PlaybookError(f"{path.name}: 'gate' must be a mapping")
    identity = str(gate.get("identity") or "any").lower()
    if identity not in _VALID_GATE_IDENTITIES:
        raise PlaybookError(
            f"{path.name}: gate.identity {identity!r} not in "
            f"{sorted(_VALID_GATE_IDENTITIES)}")
    pb["gate"] = {"confirm": bool(gate.get("confirm", True)),
                  "identity": identity}
    verify = pb.get("verify") or []
    if not isinstance(verify, list) or not all(
            isinstance(v, str) and v.strip() for v in verify):
        raise PlaybookError(
            f"{path.name}: 'verify' must be a list of commands ([] allowed)")
    pb["verify"] = [v.strip() for v in verify]
    domains = pb.get("memory_domains") or []
    if not isinstance(domains, list) or not all(
            isinstance(d, str) and d.strip() for d in domains):
        raise PlaybookError(
            f"{path.name}: 'memory_domains' must be a list of bank names")
    pb["memory_domains"] = [d.strip() for d in domains]
    try:
        pb["max_runtime_min"] = int(pb.get("max_runtime_min") or 0)
    except (TypeError, ValueError):
        pb["max_runtime_min"] = 0
    if pb["max_runtime_min"] <= 0:
        raise PlaybookError(
            f"{path.name}: 'max_runtime_min' must be a positive int")
    declared = {str(p) for p in params}
    allowed = declared | {f"{p}_line" for p in params}
    unknown = _template_fields(pb["task_template"]) - allowed
    if unknown:
        raise PlaybookError(
            f"{path.name}: task_template placeholder(s) {sorted(unknown)} "
            "not declared in params")
    return pb


def load_playbook(name: str,
                  playbooks_dir: Path | None = None) -> dict[str, Any]:
    """Load + validate backend/playbooks/<name>.yml. Raises PlaybookError."""
    if not _PLAYBOOK_NAME_RE.match(name or ""):
        raise PlaybookError(f"invalid playbook name {name!r}")
    root = Path(playbooks_dir) if playbooks_dir else PLAYBOOKS_DIR
    path = root / f"{name}.yml"
    if not path.is_file():
        raise PlaybookError(
            f"unknown playbook {name!r} (available: "
            f"{', '.join(list_playbooks(root)) or 'none'})")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlaybookError(f"{name}: yaml parse failed: {exc}") from exc
    return _validate_playbook(path, data)


def _resolve_params(pb: dict[str, Any],
                    params: dict[str, Any] | None) -> dict[str, str]:
    """Validate params against the playbook and build template values.

    Every declared param maps to a string value; a param with a 'line'
    sub-template also yields '<name>_line' — the rendered line when the
    param was given, else "" (clean optional-section interpolation).
    """
    declared = pb["params"]
    params = dict(params or {})
    unknown = sorted(str(k) for k in params if k not in declared)
    if unknown:
        raise PlaybookError(
            f"{pb['name']}: unknown param(s) {', '.join(unknown)} "
            f"(declared: {', '.join(sorted(declared))})")
    values: dict[str, str] = {}
    missing: list[str] = []
    for pname, cfg in declared.items():
        raw = params.get(pname)
        v = "" if raw is None else str(raw).strip()
        if not v and cfg.get("default") is not None:
            v = str(cfg["default"]).strip()
        if cfg.get("required") and not v:
            missing.append(str(pname))
        values[str(pname)] = v
        values[str(pname) + "_line"] = (
            str(cfg["line"]).format(value=v)
            if v and cfg.get("line") else "")
    if missing:
        raise PlaybookError(
            f"{pb['name']}: missing required param(s) {', '.join(missing)}")
    return values


def _render_task(pb: dict[str, Any], values: dict[str, str],
                 task: str | None = None) -> str:
    """task_template + params + the contract trailer the session sees."""
    body = pb["task_template"].format_map(values).strip()
    trailer = [f"Playbook: {pb['name']}"]
    if pb["verify"]:
        trailer.append(
            "Verify before finishing — run each command and record the "
            "results in dispatch-outcome.md:")
        trailer += [f"  - {c}" for c in pb["verify"]]
    else:
        trailer.append("Verify: none — this is a report-only playbook.")
    trailer.append(f"Report to: {pb['report_to']}")
    if pb["memory_domains"]:
        trailer.append("Memory domains you may consult via the mddb MCP: "
                       + ", ".join(pb["memory_domains"]))
    trailer.append(f"Timebox: {pb['max_runtime_min']} minutes.")
    rendered = body + "\n\n" + "\n".join(trailer)
    if task and str(task).strip():
        rendered += "\n\nRequest context:\n" + str(task).strip()
    return rendered


def _meta_first(meta: dict[str, Any], field: str) -> str:
    v = (meta or {}).get(field)
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v) if v is not None else ""


async def _enforce_gate(name: str, pb: dict[str, Any],
                        values: dict[str, str], mddb: Any,
                        caller: str | None, is_owner: bool) -> dict | None:
    """Playbook gate: identity + declared bank-side requirements.

    Returns a refusal dict the tool surfaces to the model, or None when
    the call may proceed. Fails closed: an unverifiable requirement is a
    refusal, not a pass.
    """
    identity_req = pb["gate"]["identity"]
    if identity_req == "owner" and not is_owner:
        return {"ok": False, "playbook": name, "gate": "identity",
                "error": f"playbook '{name}' is owner-gated — only the "
                         "owner's identities may dispatch it."}
    if identity_req == "identified" and not (caller or is_owner):
        return {"ok": False, "playbook": name, "gate": "identity",
                "error": f"playbook '{name}' requires an identified caller."}

    want_status = str(
        (pb.get("requires") or {}).get("spec_status") or "").lower()
    if not want_status:
        return None
    spec_key = values.get("spec_key") or ""
    if mddb is None:
        return {"ok": False, "playbook": name, "gate": "spec_status",
                "spec_key": spec_key,
                "error": f"playbook '{name}' cannot verify spec "
                         f"'{spec_key}' — the memory store is unavailable, "
                         "so the spec's review status cannot be checked. "
                         "Refusing to dispatch unverified."}
    doc = await mddb.get_document(HANDOFF_COLLECTION, spec_key)
    if doc is None and spec_key and not spec_key.startswith("spec/"):
        spec_key = "spec/" + spec_key
        doc = await mddb.get_document(HANDOFF_COLLECTION, spec_key)
    meta = (doc or {}).get("meta") or {}
    kinds = meta.get("kind") or []
    status = _meta_first(meta, "status").lower()
    base = {"ok": False, "playbook": name, "gate": "spec_status",
            "spec_key": spec_key, "spec_status": status or None,
            "offer": "ada_devteam_review"}
    if doc is None or "spec-review" not in kinds:
        return {**base, "error": (
            f"playbook '{name}' refused: no spec-review doc '{spec_key}' "
            "in the devin-handoff bank. Do NOT dispatch anyway — offer to "
            "run ada_devteam_review first so a reviewed spec exists.")}
    if status != want_status:
        return {**base, "error": (
            f"playbook '{name}' refused: spec doc '{spec_key}' has status "
            f"'{status or 'unknown'}', needs '{want_status}'. Do NOT "
            "dispatch anyway — offer to run ada_devteam_review so the "
            "spec can be (re)reviewed to a passing verdict.")}
    values["spec_key"] = spec_key  # resolved key (bare slug -> spec/<slug>)
    return None


async def _stamp_contract(mddb: Any, task_id: str, name: str,
                          pb: dict[str, Any],
                          values: dict[str, str]) -> str | None:
    """job/<task_id>/contract doc — the playbook + verify list the watch
    timer picks up alongside the job/<id> ledger entry it owns.

    Companion doc, not meta on job/<id>: the watcher's own /add rewrites
    that doc's meta wholesale on every transition, so a stamp there would
    die at exactly the moment the verify list is needed.
    """
    if mddb is None:
        return None
    key = f"job/{task_id}/contract"
    verify = pb["verify"]
    body = (f"Playbook contract for job {task_id}: {name}.\n\n"
            + ("Verify:\n" + "\n".join(f"- {c}" for c in verify)
               if verify else "Verify: none (report-only playbook).")
            + f"\n\nReport to: {pb['report_to']}"
            + f"\nMax runtime: {pb['max_runtime_min']} min"
            + (f"\nSpec: {values['spec_key']}" if values.get("spec_key") else ""))
    meta: dict[str, list[str]] = {
        "kind": ["job-contract"], "status": ["stamped"],
        "job_id": [task_id], "playbook": [name],
        "report_to": [str(pb["report_to"])],
        "max_runtime_min": [str(pb["max_runtime_min"])],
        "ts": [datetime.now(timezone.utc).isoformat()],
        "subject": [f"{name}-{task_id}"],
        "source": ["devin_dispatch"], "written_by": ["ada"],
        "scope": ["tony"], "bank": ["devin-handoff"],
    }
    if verify:
        meta["verify"] = verify
    if pb["memory_domains"]:
        meta["memory_domains"] = pb["memory_domains"]
    if values.get("spec_key"):
        meta["spec_key"] = [values["spec_key"]]
    ok = await mddb.add_document(
        HANDOFF_COLLECTION, key=key, lang="en", content_md=body, meta=meta)
    return key if ok else None


# -- Board dispatch (no-playbook path) -------------------------------------
#
# Files an armed card through board-api POST /card — the same contract
# the CMS spawn route uses (pwa_server /api/cms/spawn): action{type:
# dispatch, repo} + queue:true puts it on kanban-dispatch's claim rail;
# the runner picks it up and the board tracks the session. The card's
# spec field carries the full task text plus a provenance line; board-api
# validates action.repo against its dispatch whitelist and 400s on a repo
# it won't run — surfaced verbatim.

# Card writes can sit behind the board page rebuild under flock (~45s on
# the Caddy route per tools.d/kanban.py) — a read-y timeout would report
# unreachable on a write that landed.
_BOARD_TIMEOUT_S = float(os.environ.get("ADA_BOARD_DISPATCH_TIMEOUT_S", "75"))
_CARD_TITLE_MAX = 80
_CARD_PROVENANCE = "Dispatched by Ada voice/chat on behalf of the user."
_CARD_CREATED_RE = re.compile(r"card\s+([a-z0-9][a-z0-9._-]*)",
                              re.IGNORECASE)


def _created_card_id(data: dict[str, Any] | None) -> str | None:
    """Pull the created card's id out of a POST /card response — check the
    structured fields first, then the 'card <id> ...' message text (same
    parse kanban.py's action=file uses)."""
    data = data or {}
    for key in ("id", "card_id"):
        cid = str(data.get(key) or "").strip()
        if cid:
            return cid
    card = data.get("card")
    if isinstance(card, dict) and card.get("id"):
        return str(card["id"])
    m = _CARD_CREATED_RE.search(str(data.get("message") or ""))
    return m.group(1) if m else None


async def _board_dispatch(repo: str, task: str) -> dict:
    """Queue the task as an armed kanban card; returns the tool result."""
    card_id = f"voice-dispatch-{int(time.time())}"
    status, data = await board_client.post("/card", {
        "id": card_id,
        "title": " ".join(task.split())[:_CARD_TITLE_MAX],
        "spec": task.strip() + "\n\n" + _CARD_PROVENANCE,
        "column": "backlog",
        "tags": ["voice-dispatch"],
        "priority": "medium",
        "action": {"type": "dispatch", "repo": repo},
        "queue": True,
        "on_exists": "queue",
        "from": "ada",
        "text": _CARD_PROVENANCE,
    }, timeout=_BOARD_TIMEOUT_S)
    if status == 0:
        # No ssh fallback: the board is the dispatch rail — bypassing it
        # would launch untracked work.
        return {"ok": False,
                "error": ("board-api is unreachable "
                          f"({data.get('error') or 'no response'}) — the "
                          "dispatch card was not filed. Say the dispatch "
                          "failed; do not retry immediately.")}
    if status >= 400:
        # e.g. repo not in board-api's dispatch whitelist — the server's
        # 'error' field is the message to relay.
        return {"ok": False, "status": status,
                "error": str(data.get("error")
                             or f"board-api HTTP {status}")}
    return {
        "ok": True,
        "card_id": _created_card_id(data) or card_id,
        "note": ("queued on the board — kanban-dispatch will claim and "
                 "run it"),
    }


async def dispatch(repo: str, task: str | None = None, *,
                   playbook: str | None = None,
                   params: dict[str, Any] | None = None,
                   mddb: Any = None,
                   caller: str | None = None,
                   is_owner: bool = False) -> dict:
    """Start a headless Devin session.

    With playbook=<name> the task is rendered from
    backend/playbooks/<name>.yml: params feed the template, the declared
    gate is enforced before anything is dispatched (build-tool refuses
    when its spec doc is missing or not status=pass), and the finished
    task is stamped into the job ledger as job/<task_id>/contract carrying
    the playbook + verify list for devin-dispatch-watch. Returns the
    devin-dispatch task id + registry info.

    Without a playbook the task is filed as an armed kanban card via
    board-api and kanban-dispatch claims it — returns the card id.
    """
    pb: dict[str, Any] | None = None
    values: dict[str, str] = {}
    if playbook:
        try:
            pb = load_playbook(playbook)
            values = _resolve_params(pb, params)
        except PlaybookError as exc:
            return {"ok": False, "playbook": playbook, "error": str(exc)}
        refusal = await _enforce_gate(
            playbook, pb, values, mddb, caller, is_owner)
        if refusal is not None:
            logger.info("dispatch refused: playbook=%s gate=%s",
                        playbook, refusal.get("gate"))
            return refusal
        task = _render_task(pb, values, task)
    if not task or not str(task).strip():
        return {"ok": False,
                "error": "task or playbook is required"}
    now = time.monotonic()
    toks = _task_tokens(task)
    entries = _recent_dispatches.setdefault(repo, [])
    entries[:] = [e for e in entries if now - e[3] < DEDUP_WINDOW_S]
    dupe = None
    dupe_is_card = False
    for prev_toks, prev_id, prev_is_card, _ in entries:
        if not toks or not prev_toks:
            continue
        sim = len(toks & prev_toks) / min(len(toks), len(prev_toks))
        if sim >= DEDUP_MIN_SIM:
            dupe = prev_id
            dupe_is_card = prev_is_card
            break
    if dupe is None:
        dupe = await _remote_dupe(repo, toks)
    if dupe is not None:
        logger.info("dedup: reusing %s for similar task", dupe)
        entries.append((toks, dupe, dupe_is_card, now))
        return _dedup_result(dupe, repo,
                           card_id=dupe if dupe_is_card else None)
    if pb is None:
        res = await _board_dispatch(repo, task)
        if res.get("ok"):
            entries.append((toks, res["card_id"], True, now))
        return res
    task_id = (await _run("start", repo, task)).strip()
    entries.append((toks, task_id, False, now))
    result = {
        "task_id": task_id,
        "repo": repo,
        "note": (
            "Session is running unattended on tony-dell in a dedicated "
            "worktree. Use devin_read action='status' to check progress; a completion "
            "notification is sent automatically when it finishes."
        ),
    }
    contract = await _stamp_contract(mddb, task_id, playbook, pb, values)
    result.update({
        "playbook": playbook,
        "verify": pb["verify"],
        "report_to": pb["report_to"],
        "max_runtime_min": pb["max_runtime_min"],
    })
    if contract:
        result["job_contract"] = contract
    else:
        result["note"] += (" (job-contract stamp skipped — memory "
                           "store unavailable)")
    return result


async def status(task_id: str | None = None) -> str:
    """List tasks, or show one task's unit state + transcript timestamp."""
    args = ["status"] + ([task_id] if task_id else [])
    out = await _run(*args)
    if task_id:
        return out
    lines = [line for line in out.splitlines() if line.strip()]
    if len(lines) <= STATUS_MAX_LINES:
        return out
    shown = lines[-STATUS_MAX_LINES:]
    return (
        f"showing latest {STATUS_MAX_LINES} of {len(lines)} tasks "
        "(pass a task_id for details)\n" + "\n".join(shown)
    )


# `devin-dispatch status` prints: <id> <state> <result> repo=<r> transcript=<ts>
_STATUS_LINE_RE = re.compile(
    r"^(?P<task_id>\S+)\s+(?P<state>\S+)\s+(?P<result>\S+)\s+"
    r"repo=(?P<repo>\S+)\s+transcript=(?P<transcript>.*)$"
)


async def tasks() -> list[dict[str, str]]:
    """Structured `devin-dispatch status`: one dict per local task dir.

    Lets callers verify job-ledger entries against ground truth on the
    dispatch host — e.g. a ledger doc still marked "running" whose unit is
    dead with transcript=never is a spawn failure, not an active job.
    """
    out = await _run("status")
    return [m.groupdict() for line in out.splitlines()
            if (m := _STATUS_LINE_RE.match(line.strip()))]


async def followup(task_id: str, message: str) -> str:
    """Send a follow-up message to a running (or resumable) session."""
    return await _run("followup", task_id, message)

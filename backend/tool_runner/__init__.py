"""FastAPI-style tool runner for Ada's Home Assistant, memory, and
habit tools.

Layout (tool-runner-split, 2026-10-06): shared module state lives in
common.py; each domain in a mixin module; ToolRunner composes them by
multiple inheritance. backend.tool_runner stays the import path — the
wildcard re-exports every common name so `from backend.tool_runner
import X` and patch("backend.tool_runner.X") keep working.
"""
from __future__ import annotations

from .common import *  # noqa: F401,F403

from .calendar import CalendarMixin
from .chat import ChatMixin
from .cms import CmsMixin
from .devin import DevinMixin
from .docs_drive import DocsDriveMixin
from .gev import GevMixin
from .ha import HaMixin
from .memory import MemoryMixin
from .meta import MetaMixin
from .screens import ScreensMixin
from .web import WebMixin


class ToolRunner(
    DevinMixin,
    DocsDriveMixin,
    MetaMixin,
    CalendarMixin,
    HaMixin,
    WebMixin,
    MemoryMixin,
    CmsMixin,
    ScreensMixin,
    ChatMixin,
    GevMixin,
):
    """Execute Ada tools for FastAPI and the voice provider."""

    def __init__(self, ha_client: HomeAssistantClient, habit_state_getter: Any | None = None, instance_id: str | None = None) -> None:
        self.context = ToolContext(ha_client=ha_client, habit_state_getter=habit_state_getter, runner=self)
        # CHABA_MEMORY=1 guest mode: file-backed public memory, no MDDB.
        self.chaba = chaba_memory.get_store() if chaba_memory.enabled() else None
        if self.chaba is not None:
            self.mddb = None
            self.memory = None
            self.events = None
        else:
            self.mddb = MddbClient()
            self.memory = AdaMemoryStore(ha_client, mddb_client=self.mddb, instance_id=instance_id)
            self.events = HaEventRecorder(ha_client, mddb_client=self.mddb, instance_id=instance_id)
        self._instance_id = instance_id
        # Dedup breaker: tracks identical needs_confirm denials so a model
        # that retries the same gated call instead of asking the user gets
        # a hard stop. {(tool, args-key): [timestamps]}
        self._denials: dict[tuple[str, str], list[float]] = {}
        # Per-session storm breaker (card ada-tool-retry-storm): failure
        # timestamps per (scope, tool, error_class), and open circuits per
        # (scope, tool) once a class trips — the runner is shared, so the
        # scope key keeps one session's outage from breaking another's.
        self._storm_fails: dict[tuple[str, str, str], list[float]] = {}
        self._storm_open: dict[tuple[str, str], dict[str, Any]] = {}
        # Voice session that invoked the current tool call; the realtime
        # provider sets this so memory writes carry provenance.
        self.session_id: str | None = None
        # HA person entity of the current speaker (set by the realtime
        # provider from speaker ID). Used to route personal memory to the
        # speaker's person-scoped bank instead of the instance default.
        self.current_speaker_ha_person: str | None = None
        # Issued-key/caller name for this session (set by pwa_server at ws
        # connect). Fallback memory-policy identity when speaker ID is off.
        self.session_caller_name: str | None = None
        # HA person entity bound to the session's issued key at connect
        # (auth.ha_person_for_key). Bridges key -> person.<id> identity so
        # unenrolled speakers still land in their person-scoped bank.
        self.session_caller_ha_person: str | None = None
        # Session owner identity pinned at websocket connect
        # (session_caller_ha_person or session_caller_name). Immutable for
        # the session's lifetime — ALL permission checks evaluate this;
        # a recognized guest voice can never widen authorization (P1/P3).
        self.session_owner_identity: str | None = None
        # Shared L0 doc-work log — the provider assigns this to the
        # conversation's doc_items list so ada_doc_* calls land in the
        # session report timeline. None = doc actions not recorded.
        self.doc_log: list[dict[str, Any]] | None = None
        # Shared L0 session-mechanics log — same pattern as doc_log, for
        # policy denials and security-relevant events.
        self.event_log: list[dict[str, Any]] | None = None
        # Active SpeakerSession for voice enrollment — set by pwa_server
        # when the WebSocket session opens. Used by ada_enroll_speaker to
        # capture the user's voice from the buffered audio.
        self.speaker_session: Any | None = None
        self._banks: MemoryBankRegistry | None = None
        self._decision_engine: DecisionCheckEngine | None = None
        self._calendar: CalendarService | None = None
        self._calendar_loaded = False
        self._control_calls: list[float] = []
        self._control_entity_calls: dict[str, list[float]] = {}
        # Bound confirmations minted on denial: token -> {fp, at, tool}.
        # Single-use, TTL'd; see _mint_confirm_token/_consume_confirm_token.
        self._confirm_tokens: dict[str, dict[str, Any]] = {}
        # Structured ledger of confirm proposals/grants/consumptions — the
        # audit trail for "which action did the user actually approve?".
        self._confirm_audit: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_CONFIRM_AUDIT_MAX
        )

    def _memory_identity(self) -> str | None:
        """Memory-policy identity: the in-flight call's identity first
        (ContextVar set by execute()), then the identified speaker's HA
        person, then the key's bound HA person / caller name, else None."""
        v = _CALLER_IDENTITY.get()
        if v is not None:
            return v
        return (self._current_speaker()
                or self.session_caller_ha_person
                or self.session_caller_name)

    def _current_speaker(self) -> str | None:
        """Session-scoped identified speaker — the ContextVar set by
        execute() wins (a session with NO identified speaker passes None,
        which must NOT fall back to the shared field); the shared field is
        only a fallback for non-ws callers (REST/tests)."""
        v = _CALLER_SPEAKER.get()
        if v is _IDENTITY_UNSET:
            return self.current_speaker_ha_person
        return v

    def _owner(self) -> str | None:
        """The calling session's pinned owner — per-call contextvar first
        (shared runner can't race it), shared field as fallback."""
        return _CALLER_OWNER.get() or self.session_owner_identity

    def _speaker_is_owner_tier(self) -> bool:
        """Boss rule (card line-voice-enroll-recognize): the identified
        speaker's identity carries a full-access policy — persons.yml
        role:admin renders {full: true} into person_policies and
        control_policies. A positively-identified owner voice keeps
        owner-tier on ANY lane, including a member lane whose session
        key belongs to someone else (person.tony on the kk lane).

        Voice-attributed elevation, so the bar is the same as a live
        identification: a fresh label (staleness check mirrors
        _is_secondary_turn) plus the identify threshold/margin that set
        it. Residual spoofing risk (replay, similar voices) is
        documented in the speaker_profiles tool guide."""
        speaker = self._current_speaker()
        if not speaker:
            return False
        ss = _CALLER_SPEAKER_SESSION.get(None)
        if ss is not None:
            age = getattr(ss, "speaker_age_s", lambda: None)()
            if age is not None and age > SPEAKER_STALE_S:
                return False
        try:
            banks = self.banks
            for policies in (banks.person_policies,
                             banks.control_policies):
                if (policies.get(speaker) or {}).get("full"):
                    return True
        except Exception:
            pass
        return False

    def policy_identity(self) -> str | None:
        """Authorization identity: the session owner pinned at connect.
        Falls back to the live speaker identity only when no session owner
        exists (REST calls, anonymous/test sessions). An identified
        owner-tier voice on a member lane resolves to the speaker's own
        identity — the boss rule."""
        if self._speaker_is_owner_tier():
            return self._current_speaker()
        return self._owner() or self._memory_identity()

    def _is_secondary_turn(self) -> bool:
        """True while the identified speaker is not the session owner.

        Only a positively-identified different voice counts — an
        unrecognized voice can never be proven non-owner, so it keeps the
        owner's rights (unenrolled owners must not lock themselves out).
        """
        speaker = self._current_speaker()
        owner = self._owner()
        if not speaker or not owner or speaker == owner:
            return False
        # Stale label check: when the buffer has seen voiced chunks since
        # the label last re-confirmed, an aged-out identification is just
        # a stale guess — drop it (SPEAKER_STALE_S rationale above).
        ss = _CALLER_SPEAKER_SESSION.get(None)
        if ss is not None:
            age = getattr(ss, "speaker_age_s", lambda: None)()
            if age is not None and age > SPEAKER_STALE_S:
                logger.info(
                    "speaker label %r expired (%.0fs unconfirmed) — "
                    "treating turn as unrecognized", speaker, age)
                return False
        if self._speaker_is_owner_tier():
            return False
        caller = self.session_caller_name
        aliases = {owner, caller, f"person.{_slug(caller)}" if caller else None}
        return speaker not in aliases

    def _secondary_blocked_tools(self) -> set[str]:
        """Blocked tool set for secondary turns. Rendered SSOT config
        (`session_security.secondary_blocked`) overrides the default when a
        list is present; absent/malformed config fails closed to the
        default. Entries may be group names or literal tool names."""
        try:
            spec = self.banks.session_security.get("secondary_blocked")
        except Exception:
            spec = None
        if not isinstance(spec, list):
            return SECONDARY_BLOCKED_TOOLS | {"persona_write"}
        blocked: set[str] = set()
        for tok in spec:
            tok = str(tok)
            blocked |= _SECONDARY_BLOCKED_GROUPS.get(tok, {tok})
        return blocked

    def _log_session_event(self, kind: str, **fields: Any) -> None:
        if self.event_log is not None:
            self.event_log.append({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": str(kind), **fields})

    @property
    def banks(self) -> MemoryBankRegistry:
        """Memory-bank registry, loaded on first use so a missing
        ADA_INSTANCE_ID only fails when memory tools are actually invoked."""
        if self._banks is None:
            if self._instance_id:
                self._banks = MemoryBankRegistry(instance=self._instance_id)
            else:
                self._banks = get_registry()
        return self._banks

    @property
    def decision_engine(self) -> DecisionCheckEngine:
        """Shared purchase-check pipeline for POST /api/decision/check and the
        ada_decision_check voice tool — same persist + event side-effects."""
        if self._decision_engine is None:
            self._decision_engine = DecisionCheckEngine(
                self.mddb, self.banks, self.memory.instance,
                ha_client=self.context.ha_client,
            )
        return self._decision_engine

    async def execute(self, name: str, args: dict[str, Any] | None = None,
                      *, identity: Any = _IDENTITY_UNSET,
                      speaker: Any = _IDENTITY_UNSET,
                      speaker_session: Any = _IDENTITY_UNSET,
                      owner: Any = _IDENTITY_UNSET,
                      session: Any = _IDENTITY_UNSET) -> Any:
        # Retired names from the tool consolidation resolve to their
        # canonical action= tool first — the alias table is the contract,
        # phonetic normalization below is just typo repair. The alias's
        # implied args (its action=) go under the caller's, not over.
        _implied: dict[str, Any] = {}
        _alias_src = name
        if isinstance(name, str):
            name, _implied = _resolve_alias(name)
        # Gemini sometimes spells underscore-heavy tool names phonetically
        # (c_c_t_v_wall) — normalize back so the call lands.
        method = getattr(self, name, None)
        if not method and isinstance(name, str) and "_" in name:
            bare = name.replace("_", "")
            for attr in dir(self):
                if attr.replace("_", "") == bare:
                    logger.info("tool name normalized %s -> %s",
                                name, attr)
                    name = attr
                    method = getattr(self, name, None)
                    break
        dyn_spec = None
        if not method:
            dyn_spec = self._dynamic_tools.tools.get(name)
            if dyn_spec is None:
                raise KeyError(f"Unknown tool: {name}")
            method = self._bind_dynamic(dyn_spec)
        call_args = {**_implied, **dict(args or {})}
        if _alias_src != name:
            call_args = _alias_call_args(str(_alias_src), call_args)
        # Resume the write outbox after a restart — entries queued before the
        # process died retry on the first tool call of the new process.
        if self.mddb is not None:
            write_outbox.ensure_running(self.mddb)
        # Per-call identity override: the provider passes the SESSION's
        # resolved identity — the runner is shared across sessions, so its
        # mutable identity fields can race when sessions overlap.
        ident = self._memory_identity() if identity is _IDENTITY_UNSET else identity
        _ident_token = _CALLER_IDENTITY.set(ident)
        # Same for the identified speaker: another session's voice ID must
        # not flip this session's secondary-owner gate mid-call.
        _spk_token = (
            _CALLER_SPEAKER.set(speaker)
            if speaker is not _IDENTITY_UNSET else None
        )
        # Same for the speaker session: the provider passes its own so a
        # concurrent ws connect can't swap the buffer out from under an
        # in-flight enroll call.
        _ss_token = (
            _CALLER_SPEAKER_SESSION.set(speaker_session)
            if speaker_session is not _IDENTITY_UNSET else None
        )
        # And the pinned owner: another session's connect must not re-pin
        # the owner this call authorizes against.
        _owner_token = (
            _CALLER_OWNER.set(owner)
            if owner is not _IDENTITY_UNSET else None
        )
        # The provider's own session_id for the storm breaker scope —
        # same shared-runner race class as the identity vars above.
        _sess_token = (
            _CALLER_SESSION.set(session)
            if session is not _IDENTITY_UNSET else None
        )
        try:
            return await self._execute_gated(
                name, call_args, ident, method=method, dyn_spec=dyn_spec,
                from_alias=_alias_src in _ALIASES)
        except Exception as exc:
            # Gate denials and arg errors raise instead of returning a
            # dict — append the error-path guide to the exception text so
            # the provider's {"error": ...} wrapper still carries it.
            hint = _tool_guide().get(str(name))
            if hint:
                exc.args = (f"{exc}\nusage: {hint}",)
            raise
        finally:
            _CALLER_IDENTITY.reset(_ident_token)
            _CALLER_VERIFIED_AFFIRM.set(False)
            if _spk_token is not None:
                _CALLER_SPEAKER.reset(_spk_token)
            if _ss_token is not None:
                _CALLER_SPEAKER_SESSION.reset(_ss_token)
            if _owner_token is not None:
                _CALLER_OWNER.reset(_owner_token)
            if _sess_token is not None:
                _CALLER_SESSION.reset(_sess_token)

    @property
    def _dynamic_tools(self) -> tools_loader.ToolRegistry:
        return tools_loader.registry()

    def _bind_dynamic(self, spec: "tools_loader.DynamicTool") -> Any:
        """Wrap a tools.d run(runner, **args) as a method-like callable with
        the manifest's timeout enforced."""
        async def _bound(**kwargs: Any) -> Any:
            import asyncio
            return await asyncio.wait_for(
                spec.run(self, **kwargs), timeout=spec.timeout_s)
        return _bound

    async def _execute_gated(self, name: str, call_args: dict[str, Any],
                             ident: str | None, method: Any = None,
                             dyn_spec: Any = None,
                             from_alias: bool = False) -> Any:
        if method is None:
            method = getattr(self, name, None)
        confirm: tuple[Any, Any] = (
            call_args.pop("confirmed", None),
            call_args.pop("confirm_token", None),
        )
        # Provider-verified user affirmation: the user actually said yes in
        # this session, so a confirmed call may satisfy a register-then-
        # confirm gate in one step (the pending key is still recorded).
        _affirm_token = _CALLER_VERIFIED_AFFIRM.set(
            bool(call_args.pop("_verified_affirm", None)))
        # Per-session storm breaker: a tool that already failed
        # TOOL_BREAKER_TRIP times with the same error class in this
        # session is short-circuited — the model gets a synthesized
        # 'do not retry' result and neither the gates nor the tool run.
        synth = self._storm_check(name)
        if synth is not None:
            return synth
        # Lazy-schema validation (card ada-lazy-tool-schemas): cold tools
        # ship name+blurb+empty params on the wire, so args are checked
        # against the real declaration HERE — before any gate can mint a
        # confirm token or register a pending request on a malformed call.
        # A miss returns ok:false and _usage_hint attaches the tool_guide
        # entry — the model repairs the shape and retries.
        # Absorbed-name calls (the _ALIASES back-compat surface) skip the
        # check: their args were spelled for the retired decls, and the
        # implied defaults / _alias_call_args remap are server-internal
        # contract (e.g. ada_daily_summary's period='digest' is deliberately
        # not in plan_day's declared enum) — not model-supplied input.
        if not from_alias:
            bad = _validate_call_args(name, call_args)
            if bad is not None:
                bad = self._schema_retry_bound(name, bad)
                return await self._capture_reminder(
                    name, self._usage_hint(name, bad))
        policy_ident = self.policy_identity()
        # tools-merge-display (2026-10-05): cast_to_screen's absorbed
        # seats — action='list'/'status' were ungated reads (vcast_list /
        # the phantom vcast_status) and 'say' (vcast_say) was actuating
        # but never gated or secondary-blocked. The merged actions keep
        # exactly the access they had; screen-ownership ACL still runs
        # inside the method where it always did.
        cast_ungated = (
            name == "cast_to_screen" and str(
                call_args.get("action") or "").lower()
            in {"list", "status", "say"})
        if self._is_secondary_turn():
            action = str(call_args.get("action") or "").lower()
            blocked = self._secondary_blocked_tools()
            # Dynamic (tools.d) tools default to owner-only on secondary
            # turns — the manifest must explicitly set secondary_allowed.
            if dyn_spec is not None and not dyn_spec.secondary_allowed:
                blocked = blocked | {name}
            # ada_remember's non-bank kinds (vocab/guest/habit) absorbed
            # tools that were never secondary-blocked — the carve-out keeps
            # a guest's own notes and vocab log reachable on their turn.
            nonbank_remember = (
                name == "ada_remember" and str(
                    call_args.get("kind") or "").lower()
                in _REMEMBER_NONBANK_KINDS)
            # scope='guest' absorbed guest_recall — a public-store read that
            # was never secondary-blocked.
            guest_scope_search = (
                name == "ada_memory_search" and str(
                    call_args.get("scope") or "").lower() == "guest")
            # cms_edit action='note' absorbed cms_note_update — a merge-only
            # append that was never gated or secondary-blocked.
            cms_note_edit = (
                name == "cms_edit" and str(
                    call_args.get("action") or "").lower() == "note")
            # ha_confidence without status/safety absorbed
            # ada_ha_get_device_confidence — a read that was never
            # secondary-blocked; only the set path stays gated.
            ha_confidence_read = (
                name == "ha_confidence" and not (
                    call_args.get("status") or call_args.get("safety")))
            # ada_ops absorbed five meta tools (tools-merge-meta-voice) with
            # mixed postures — the name carries ada_outcome's blocked seat,
            # but action='usage' was an open read and 'research' was never
            # secondary-blocked, so those stay reachable on a guest turn.
            ops_read = (
                name == "ada_ops" and action in ("usage", "research"))
            # tools-merge-tasks-status carve-outs: tasks action='list' is
            # the absorbed tasks_list (a free read) and a plain chat_send
            # (no photo=/doc= flow) was never secondary-blocked — only the
            # absorbed photos picker / doc-upload flows are owner-tier.
            tasks_list_read = name == "tasks" and action == "list"
            chat_doc_flow = name == "chat_send" and (
                call_args.get("photo") or call_args.get("doc"))
            # ada_enroll_speaker is exempt here — its own check is smarter:
            # the owner can re-enroll even while a secondary voice is
            # identified, as long as the buffer voice isn't the secondary's
            # (2026-09-29 deadlock: Tony locked out of his own session
            # because KK's identification was sticky and enroll refused).
            if (name in blocked and name != "ada_enroll_speaker"
                    and not nonbank_remember
                    and not guest_scope_search
                    and not cms_note_edit
                    and not ha_confidence_read
                    and not cast_ungated
                    and not ops_read
                    and not tasks_list_read
                    and (name != "chat_send" or chat_doc_flow)) or (
                name == "ada_persona" and "persona_write" in blocked
                # *_voice actions absorbed ada_set_voice — that tool was
                # never runner-side secondary-blocked (the provider gate
                # covers live turns), so only set/reset keep the seat.
                and action in ("set", "reset")
            ):
                self._log_session_event(
                    "tool_denied", tool=name,
                    speaker=self._current_speaker(),
                    owner=self.session_owner_identity)
                logger.warning(
                    "denied %s: secondary speaker %r (owner %r)",
                    name, self._current_speaker(),
                    self.session_owner_identity)
                self._storm_record(name, PermissionError(
                    "secondary speaker"))
                raise PermissionError(
                    "Only the session owner can run this action — propose it "
                    "to them aloud and let them confirm in their own voice.")
        try:
            if name in CAPTURE_CONFIRMED_TOOLS:
                # camera-capture gate first (uplink/wall-start/cctv-snapshot);
                # cast_to_screen then still runs the control gate (read-only,
                # rate limit) — except the absorbed read/narrate actions,
                # which were never control-gated.
                self._check_capture_confirmed(name, call_args, confirm[0])
                if name == "cast_to_screen" and not cast_ungated:
                    await self._check_control_allowed(name, call_args, *confirm)
            elif name in CONTROL_TOOLS:
                await self._check_control_allowed(name, call_args, *confirm)

            elif name in MEMORY_WRITE_TOOLS:
                # ada_remember kind=vocab/guest/habit writes outside the
                # curated banks (own vocab log, chaba guest store, habit
                # telemetry) — the absorbed tools were never bank-gated.
                if not (name == "ada_remember" and str(
                        call_args.get("kind") or "").lower()
                        in _REMEMBER_NONBANK_KINDS) and not (
                        # ada_ops holds ada_outcome's write seat — only
                        # action='outcome' is a bank write; usage/health/
                        # check/research carry no bank arg and skip the gate.
                        name == "ada_ops" and str(
                            call_args.get("action") or "").lower()
                        != "outcome"):
                    self._check_memory_write_allowed(name, call_args, *confirm)
            elif name in CALENDAR_WRITE_TOOLS:
                self._check_calendar_write_allowed(name, call_args, *confirm)
            elif name in CMS_WRITE_TOOLS:
                self._check_cms_write_allowed(name, call_args, *confirm)
            elif name in DEVIN_CONFIRMED_TOOLS:
                self._check_devin_confirmed(name, call_args, *confirm)
            elif dyn_spec is not None:
                self._check_dynamic_allowed(name, dyn_spec, call_args, *confirm)
            elif name in DOC_TOOLS:
                if not self.banks.bank_allowed(DOC_BANK, policy_ident):
                    logger.warning(
                        "denied %s for identity %r: documents bank policy",
                        name, ident)
                    raise PermissionError(
                        "document tools are outside this session's access policy")
                if name in DOC_CONFIRMED_TOOLS:
                    self._check_doc_confirmed(name, call_args, *confirm)
            elif name in DRIVE_TOOLS:
                # Whole-Drive + Photos access is owner-tier: same bank
                # policy as the document tools. chat_send absorbed the
                # photos picker + doc-upload card flow (tools-merge-
                # tasks-status) — only photo=/doc= calls take this gate;
                # a plain text/image send was never bank-scoped.
                if name == "chat_send" and not (
                        call_args.get("photo") or call_args.get("doc")):
                    pass
                else:
                    if not self.banks.bank_allowed(DOC_BANK, policy_ident):
                        logger.warning(
                            "denied %s for identity %r: documents bank policy",
                            name, ident)
                        raise PermissionError(
                            "drive/photos tools are outside this session's access policy")
                    if name in DRIVE_CONFIRMED_TOOLS:
                        self._check_drive_confirmed(name, call_args, *confirm)
        except PermissionError as exc:
            # Phantom-save guard (2026-09-28): the model papered over refused
            # writes and claimed success aloud. Every gate denial now carries
            # a blunt prefix the narration layer cannot miss.
            self._storm_record(name, exc)
            raise PermissionError(
                "NOT EXECUTED — the action did not happen and must not be "
                f"described as done/saved/published: {exc}"
            ) from exc
        # confirmed/confirm_token were popped for the gates above — hand them
        # back to methods that declare them (e.g. devin_job_report re-checks
        # confirmation internally on its publish path).
        # gate_args = the args exactly as the confirm gates saw them
        # (envelope keys popped, pre-normalize) — a retry arrives spelled
        # the same way, so a re-armed token must bind THIS fingerprint.
        gate_args = dict(call_args)
        _params = inspect.signature(method).parameters
        if "confirmed" in _params and confirm[0] is not None:
            call_args["confirmed"] = confirm[0]
        if "confirm_token" in _params and confirm[1] is not None:
            call_args["confirm_token"] = confirm[1]
        call_args = self._normalize_args(name, method, call_args)
        logger.info("tool %s args=%r", name, call_args)
        try:
            result = await method(**call_args)
        except Exception as exc:
            # The tools.d contract ("never raise into the model") applies
            # to native methods too — a tool that fails mid-execution
            # reports {ok: False, error} like every other failure shape.
            # Dispatch-layer denials (the _check_* gates above, an
            # unknown tool name) still raise: the tool never ran, so
            # there is no tool result.
            logger.warning("tool %s raised: %s", name, exc)
            result = {
                "ok": False,
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        result = normalize_tool_result(result)
        self._storm_record(name, result)
        result = self._rearm_confirm_on_infra_failure(
            name, gate_args, confirm, result,
            dyn_confirmed=(dyn_spec is not None
                           and getattr(dyn_spec, "policy", None) == "confirmed"))
        self._log_change_request(name, call_args, result, ident)
        result = self._denial_breaker(name, call_args, result)
        result = self._usage_hint(name, result)
        return await self._capture_reminder(name, result)

    def _usage_hint(self, name: str, result: Any) -> Any:
        """Attach the tool's error-path guide (backend/tool_guide.yml) to
        a failed or denied call — the slim schema description carries
        only routing, so on error the model gets the detailed contract
        here (same shape as the needs_confirm/verify_warn payloads)."""
        if not isinstance(result, dict) or "usage" in result:
            return result
        if not (result.get("error") or result.get("needs_confirm")
                or result.get("ok") is False):
            return result
        hint = _tool_guide().get(name)
        if hint:
            result["usage"] = hint
        return result

    # User-visible state changes Ada applies herself. Transient home
    # controls (lights, media) and memory-bank writes are deliberately
    # excluded — noisy, and already tracked elsewhere.
    _CHANGE_LOG_TOOLS = {
        "cms_publish_page", "cms_edit", "docs", "ada_forget",
        "ha_confidence", "devin", "devin_read",
    }

    def _log_change_request(self, name: str, args: dict[str, Any],
                            result: Any, ident: Any) -> None:
        """Mirror every applied system change into the Ada events feed
        (events.md -> ada-review -> Devin memory render) so changes Tony
        asks Ada for vocally reach Devin sessions — the team-sync lane."""
        if name not in self._CHANGE_LOG_TOOLS:
            return
        # ada_doc_archive's old seat — only the archive action is a
        # state change worth logging, not the read actions.
        if (name == "docs" and str(args.get("action") or "").lower()
                != "archive"):
            return
        # ha_confidence holds the absorbed ada_ha_set_device_confidence
        # seat — reads (no status/safety) aren't changes.
        if name == "ha_confidence" and not (
                args.get("status") or args.get("safety")):
            return
        # devin_read is mostly reads — only its write actions are changes
        # (devin_job_report logged every call pre-merge; review files a
        # spec doc into the handoff bank).
        if name == "devin_read" and str(
                args.get("action") or "") not in ("report", "review"):
            return
        if isinstance(result, dict) and result.get("ok") is False:
            return
        detail = args.get("slug") or args.get("title") or \
            args.get("task_id") or args.get("key") or \
            str(args.get("task") or "")[:80] or name
        try:
            from backend.event_log import log_event
            actor = getattr(ident, "name", None) or str(ident or "?")
            log_event("change-request", actor, name, str(detail)[:120])
        except Exception:
            logger.debug("change-request event log failed", exc_info=True)

    def _denial_breaker(self, name: str, args: dict[str, Any],
                        result: Any) -> Any:
        """The model sometimes retries an identical needs_confirm-gated
        call over and over (self-asserted confirmed=true gets stripped, so
        it can never pass) instead of asking the user aloud and waiting a
        turn. After the 3rd identical denial in 3min, return a hard stop."""
        if not isinstance(result, dict) or "needs_confirm" not in result:
            return result
        import hashlib
        key_args = json.dumps(
            {k: args.get(k) for k in ("screen", "action", "url")},
            sort_keys=True, default=str)
        key = (name, hashlib.md5(key_args.encode()).hexdigest())
        now = time.time()
        hits = [t for t in self._denials.get(key, []) if now - t < 180]
        hits.append(now)
        self._denials[key] = hits
        if len(hits) >= 3:
            self._denials[key] = []
            return {"ok": False, "error": (
                "STOP RETRYING — this exact call was refused " +
                str(len(hits)) +
                " times: the screen is busy and self-asserted confirmed=true "
                "does not count. Do NOT call this tool again right now — "
                "tell the user aloud what is running on the screen, ask if "
                "they want it replaced, and wait for their spoken yes in "
                "the next turn.")}
        return result

    def _schema_retry_bound(self, name: str, result: dict[str, Any]
                            ) -> dict[str, Any]:
        """Bound the malformed-call self-repair loop to ONE corrected
        retry per window (card ada-error-truthfulness-narration).
        SchemaError misses skip storm accounting (the tool never ran —
        nothing to break), so without this a model could burn the whole
        per-turn call budget guessing arg spellings. The first miss
        returns the repair payload as-is (expected_args + usage hint);
        the second inside the window is flagged no_retry so the model
        narrates the real error instead of trying a third shape. A clean
        success on the tool clears the count (see _storm_record)."""
        scope = self._storm_scope()
        fails = self.__dict__.setdefault("_schema_fails", {})
        key = (scope, name)
        now = time.monotonic()
        hits = [t for t in fails.get(key, [])
                if now - t < TOOL_BREAKER_WINDOW_S]
        hits.append(now)
        fails[key] = hits
        if len(hits) >= 2:
            result = dict(result)
            result["no_retry"] = True
            result["error"] = str(result.get("error") or "") + (
                " — DO NOT RETRY this turn: one corrected call already "
                "failed; narrate the real error to the user instead.")
        return result

    # -- per-session storm breaker ---------------------------------------

    def _storm_scope(self) -> str:
        """Session bucket for the breaker — the runner is shared, so one
        session's outage must not trip the circuit for another. The
        provider passes its session_id per call (contextvar, race-free);
        voice sessions without one fall back to the SpeakerSession
        object, then the pinned owner/caller identity."""
        sess = _CALLER_SESSION.get(None)
        if sess:
            return f"sess-{sess}"
        ss = _CALLER_SPEAKER_SESSION.get(_IDENTITY_UNSET)
        if ss is _IDENTITY_UNSET:
            ss = getattr(self, "speaker_session", None)
        if ss is not None:
            return f"ws-{id(ss)}"
        return "owner-" + str(
            self._owner() or getattr(self, "session_caller_name", None)
            or self._memory_identity() or "-")

    def _storm_check(self, name: str) -> dict[str, Any] | None:
        """Open circuit → synthesized 'do not retry' result, no execution.
        Past TOOL_BREAKER_OPEN_S the rec is dropped and the next call is a
        half-open probe: success resets the counts, another failure
        re-trips the breaker."""
        key = (self._storm_scope(), name)
        # getattr: __new__-built runners (tools_loader tests) skip __init__.
        rec = getattr(self, "_storm_open", {}).get(key)
        if rec is None:
            return None
        now = time.monotonic()
        if now >= rec["until"]:
            self._storm_open.pop(key, None)
            return None
        already = bool(rec["announced"])
        rec["announced"] = True
        if already:
            text = (
                f"{name} is still down ({rec['class']}). DO NOT RETRY — "
                "you already told the user it is broken; do not announce "
                "it again unless they ask.")
        else:
            text = (
                f"OUTAGE — {name} is failing for this session "
                f"({rec['class']}, {rec['hits']} failures in "
                f"{int(TOOL_BREAKER_WINDOW_S)}s). DO NOT RETRY this tool "
                "— tell the user ONCE, plainly, that it is not working "
                "right now, then move on.")
        return {
            "ok": False,
            "error": text,
            "error_type": "CircuitOpen",
            "error_class": rec["class"],
            "circuit_open": True,
            "already_announced": already,
            "retry_after_s": max(0, int(rec["until"] - now)),
        }

    def _storm_record(self, name: str, outcome: Any) -> None:
        """Feed one call outcome into the breaker: a clean success clears
        this session's failure counts for the tool; a counted failure may
        trip it open."""
        scope = self._storm_scope()
        cls = _storm_error_class(outcome)
        # setdefault: __new__-built runners (tools_loader tests) skip __init__.
        fails = self.__dict__.setdefault("_storm_fails", {})
        open_map = self.__dict__.setdefault("_storm_open", {})
        if cls is None:
            if isinstance(outcome, dict) and outcome.get("ok") is True:
                for k in [k for k in fails
                          if k[0] == scope and k[1] == name]:
                    fails.pop(k, None)
                # A clean call also clears the malformed-call repair
                # budget — the next miss gets a fresh corrected retry.
                sfails = self.__dict__.get("_schema_fails") or {}
                sfails.pop((scope, name), None)
            return
        key = (scope, name, cls)
        now = time.monotonic()
        hits = [t for t in fails.get(key, [])
                if now - t < TOOL_BREAKER_WINDOW_S]
        hits.append(now)
        fails[key] = hits
        if len(hits) >= TOOL_BREAKER_TRIP:
            open_map[(scope, name)] = {
                "class": cls,
                "hits": len(hits),
                "until": now + TOOL_BREAKER_OPEN_S,
                "announced": False,
            }
            logger.warning(
                "storm breaker OPEN tool=%s scope=%s: %d %s failures "
                "in %ds — calls short-circuit until +%ds",
                name, scope, len(hits), cls,
                int(TOOL_BREAKER_WINDOW_S), int(TOOL_BREAKER_OPEN_S))

    # Tools that already carry capture state — a reminder on them would
    # be noise (the capture IS the subject of these calls). The absorbed
    # vcast_list/vcast_say names ride in as cast_to_screen actions, so
    # the canonical name covers them (tools-merge-display).
    _CAPTURE_AWARE_TOOLS = {
        "cast_to_screen", "cctv_wall", "ada_camera_snapshot",
    }
    _capture_reminded_ts = 0.0

    async def _capture_reminder(self, name: str, result: Any) -> Any:
        """While a camera capture/uplink/wall is live, tag unrelated tool
        results with a one-line reminder so Ada can't silently walk away
        from a running capture mid-conversation (2026-09-29 transcript:
        topic-shift to calendar left the zone-a wall unacknowledged).
        Throttled to once a minute."""
        if (name in self._CAPTURE_AWARE_TOOLS
                or not isinstance(result, dict)
                or result.get("capture_reminder")
                or time.time() - self._capture_reminded_ts < 60):
            return result
        try:
            import asyncio
            caps = await asyncio.to_thread(self._vcast_api, "/capture")
            live = {s: c for s, c in (caps.get("captures") or {}).items()
                    if c.get("active")}
            if not live:
                return result
            self._capture_reminded_ts = time.time()
            desc = ", ".join(
                f"screen {s}: {c.get('source') or 'cam'}"
                + (f" ({c['ch']})" if c.get("ch") else "")
                for s, c in live.items())
            result["capture_reminder"] = (
                f"heads-up — camera capture still live ({desc}); "
                "acknowledge it to the user or ask before ending "
                "the topic.")
        except Exception:
            pass
        return result

    @staticmethod
    def _normalize_args(name: str, method: Any, call_args: dict[str, Any]) -> dict[str, Any]:
        params = inspect.signature(method).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return call_args
        for alias, canonical in _ARG_ALIASES.items():
            if alias in call_args and canonical in params and canonical not in call_args:
                call_args[canonical] = call_args.pop(alias)
        extra = [k for k in call_args if k not in params]
        if extra:
            logger.warning("tool %s: ignoring unexpected args %r", name, extra)
            for key in extra:
                call_args.pop(key, None)
        return call_args

    # -- confirmation gate machinery -------------------------------------

    def _audit_confirmation(
        self, event: str, tool: str, fingerprint: str, via: str
    ) -> None:
        self._confirm_audit.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "tool": tool,
            "fingerprint": fingerprint[:12],
            "via": via,
            "identity": self._memory_identity(),
        })

    def confirmation_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest-last ledger of confirm proposals, grants, consumptions,
        and denials — the auditable answer to 'approved which action?'."""
        entries = list(self._confirm_audit)
        return entries[-max(1, int(limit)):]

    def _mint_confirm_token(self, tool: str, args: dict[str, Any]) -> str:
        """Mint a single-use token bound to this exact tool + args."""
        now = time.monotonic()
        for tok, rec in list(self._confirm_tokens.items()):
            if now - rec["at"] > CONFIRM_TOKEN_TTL_S:
                self._confirm_tokens.pop(tok, None)
        while len(self._confirm_tokens) >= _CONFIRM_TOKEN_MAX:
            self._confirm_tokens.pop(next(iter(self._confirm_tokens)))
        token = "cfm-" + secrets.token_hex(5)
        fp = _confirm_fingerprint(tool, args)
        self._confirm_tokens[token] = {"fp": fp, "at": now, "tool": tool}
        self._audit_confirmation("proposed", tool, fp, "token")
        return token

    def _consume_confirm_token(
        self, tool: str, args: dict[str, Any], token: Any
    ) -> str | None:
        """Validate and consume a minted token; None = accepted, else reason."""
        if not isinstance(token, str) or not token.strip():
            return "not a string"
        rec = self._confirm_tokens.get(token)
        if rec is None:
            return "unknown or expired confirm_token"
        if time.monotonic() - rec["at"] > CONFIRM_TOKEN_TTL_S:
            self._confirm_tokens.pop(token, None)
            return "confirm_token expired"
        if rec["tool"] != tool:
            return "confirm_token was issued for a different tool"
        if rec["fp"] != _confirm_fingerprint(tool, args):
            return "confirm_token was issued for different arguments"
        self._confirm_tokens.pop(token, None)  # single-use
        return None

    def pending_confirm(self, tool: str, args: dict[str, Any]) -> bool:
        """True when a live confirm token already covers this exact
        tool+args — an earlier denial proposed this call and the caller
        is retrying within the token window. `args` must exclude the
        confirmed/confirm_token envelope keys (the provider filters them
        before calling). Used to accept the model's actual retry form —
        confirmed=true on the identical call — which never carries the
        token spelling (2026-10-07 ada_remember flake: 6/8 writes
        NOT-EXECUTED because the model resends confirmed=true)."""
        fp = _confirm_fingerprint(tool, args)
        now = time.monotonic()
        for rec in self._confirm_tokens.values():
            if (rec["tool"] == tool and rec["fp"] == fp
                    and now - rec["at"] <= CONFIRM_TOKEN_TTL_S):
                return True
        return False

    def _burn_pending_confirms(self, tool: str, fingerprint: str) -> None:
        """Drop live tokens for this exact call — a confirmed=true grant
        already armed it, so an outstanding token must not replay it."""
        for tok, rec in list(self._confirm_tokens.items()):
            if rec["tool"] == tool and rec["fp"] == fingerprint:
                self._confirm_tokens.pop(tok, None)

    def _require_confirmation(
        self,
        name: str,
        args: dict[str, Any],
        confirmed: Any,
        token: Any,
        instruction: str,
    ) -> None:
        """Shared confirm gate. Accepts a truthy `confirmed` (user said yes
        before the call) or a bound `confirm_token` minted by an earlier
        denial. Otherwise raises PermissionError carrying a fresh token."""
        fp = _confirm_fingerprint(name, args)
        if _confirmed_truthy(confirmed):
            if token is not None:
                # Burn the token even though the flag sufficed — a token left
                # unconsumed here could otherwise be replayed later.
                self._consume_confirm_token(name, args, token)
            else:
                # Same hygiene for the flag-only grant: any outstanding
                # token minted for this exact call is dead — replaying it
                # must not re-arm the same write.
                self._burn_pending_confirms(name, fp)
            self._audit_confirmation("granted", name, fp, "confirmed")
            return
        rejected = ""
        if token is not None:
            reason = self._consume_confirm_token(name, args, token)
            if reason is None:
                self._audit_confirmation("consumed", name, fp, "token")
                return
            rejected = f" ({reason})"
            self._audit_confirmation("denied", name, fp, f"token:{reason}")
        else:
            self._audit_confirmation("denied", name, fp, "unconfirmed")
        fresh = self._mint_confirm_token(name, args)
        logger.warning("denied %s %r: confirmation required%s", name, args, rejected)
        raise PermissionError(
            f"{instruction}{rejected} TO EXECUTE: replay the SAME call "
            f"adding the argument confirm_token='{fresh}' (single use, "
            f"expires in {int(CONFIRM_TOKEN_TTL_S)}s) — or resend the "
            "identical call with confirmed=true."
        )

    def _rearm_confirm_on_infra_failure(
        self, name: str, gate_args: dict[str, Any],
        confirm: tuple[Any, Any], result: Any,
        dyn_confirmed: bool = False,
    ) -> Any:
        """Re-issue a bound confirm_token inside an infra-class failure
        result. A confirmed call that dies on transport (ssh timeout,
        refused, unreachable) never ran — consuming the grant and then
        denying the retry forces a SECOND voice confirmation for the same
        request (confirm-token-retry-expiry: sessions 38b18589b0 +
        7a74536b32, "โทเค็นหมดอายุ อีกแล้ว"). The fresh token is bound to
        the same tool+args fingerprint, so only the approved identical
        retry can spend it — confirmed=true resends also pass via
        pending_confirm(). Policy refusals (gate= key) and denials
        (NOT EXECUTED / "requires confirmation" text) never re-arm."""
        if not isinstance(result, dict) or result.get("ok") is not False:
            return result
        if result.get("gate") or result.get("confirm_token"):
            return result
        if not (_confirmed_truthy(confirm[0]) or confirm[1]):
            return result
        if name not in CONFIRM_GATED_TOOLS and not dyn_confirmed:
            return result
        err = str(result.get("error") or result.get("err") or "")
        etype = str(result.get("error_type") or "")
        if ("NOT EXECUTED" in err or "requires confirmation" in err
                or not (_INFRA_ERROR_RE.search(err)
                        or etype in _INFRA_ERROR_TYPES)):
            return result
        fresh = self._mint_confirm_token(name, gate_args)
        fp = _confirm_fingerprint(name, gate_args)
        self._audit_confirmation("rearmed", name, fp, "infra_failure")
        self._log_session_event(
            "confirm_rearm", tool=name,
            speaker=self._current_speaker(),
            owner=self.session_owner_identity)
        logger.info(
            "re-armed confirm token for %s after infra failure: %s",
            name, err[:120])
        result["confirm_token"] = fresh
        result["retry_hint"] = (
            "Transient infrastructure failure — the action did NOT run. "
            "The confirmation already given still stands: retry the "
            f"identical call adding confirm_token='{fresh}' (single use, "
            f"expires in {int(CONFIRM_TOKEN_TTL_S)}s), or resend it with "
            "confirmed=true. Do NOT ask the user to confirm again — a "
            "downstream failure is not a new confirmation question.")
        return result

    async def _check_control_allowed(
        self, name: str, args: dict[str, Any], confirmed: Any,
        confirm_token: Any = None,
    ) -> None:
        """Server-side gate for actuating tools. Raises PermissionError on denial."""
        if os.environ.get("ADA_READ_ONLY") == "true":
            logger.warning("denied %s %r: ADA_READ_ONLY", name, args)
            raise PermissionError("control tools are disabled (ADA_READ_ONLY=true)")
        now = time.monotonic()
        self._control_calls = [t for t in self._control_calls if now - t < CONTROL_RATE_WINDOW_S]
        if len(self._control_calls) >= CONTROL_MAX_GLOBAL:
            logger.warning("denied %s %r: global rate limit", name, args)
            raise PermissionError(
                f"rate limit exceeded: more than {CONTROL_MAX_GLOBAL} control calls in {int(CONTROL_RATE_WINDOW_S)}s"
            )
        entity_id = str(args.get("entity_id") or "")
        if entity_id:
            ident = self.policy_identity()
            decision = self.banks.acl_decision(ident, entity_id=entity_id)
            if not decision["allowed"]:
                logger.warning(
                    "denied %s on %r for identity %r: control policy (%s)",
                    name, entity_id, ident, decision.get("decided_by"),
                )
                raise PermissionError(
                    f"'{entity_id}' is outside this session's control policy"
                )
            calls = self._control_entity_calls.setdefault(entity_id, [])
            calls[:] = [t for t in calls if now - t < CONTROL_RATE_WINDOW_S]
            if len(calls) >= CONTROL_MAX_PER_ENTITY:
                logger.warning("denied %s %r: per-entity rate limit", name, args)
                raise PermissionError(
                    f"rate limit exceeded for {entity_id}: more than {CONTROL_MAX_PER_ENTITY} "
                    f"control calls in {int(CONTROL_RATE_WINDOW_S)}s"
                )
            if self.chaba is not None:
                # Guest mode has no MDDB safety map — deny anything that can
                # move or open: locks, covers, buttons, gates, garages, doors.
                if re.search(r"(?:lock|cover|button|gate|garage|door|siren|alarm)", entity_id):
                    logger.warning("denied %s on %r: chaba guest deny-pattern", name, entity_id)
                    raise PermissionError(
                        f"guests cannot control '{entity_id}' (locks/covers/gates are off-limits)"
                    )
            else:
                await self.memory._ensure_confidence()
            safety = self.memory._safety.get(entity_id) if self.memory is not None else None
            if safety != "dangerous" and self.memory is not None:
                # A related entity can inherit danger: button.gate_motor_my_position
                # physically jogs the dangerous cover.gate_motor.
                object_id = entity_id.split(".", 1)[-1]
                for eid, level in self.memory._safety.items():
                    if level == "dangerous" and object_id.startswith(eid.split(".", 1)[-1] + "_"):
                        safety = "dangerous"
                        break
            if safety == "dangerous":
                self._require_confirmation(
                    name, args, confirmed, confirm_token,
                    f"{entity_id} is marked dangerous. Call again with "
                    "confirmed=true only after explicit user confirmation.",
                )
            calls.append(now)
        self._control_calls.append(now)

    def _check_capture_confirmed(
        self, name: str, args: dict[str, Any], confirmed: Any,
    ) -> None:
        """Server-side gate for camera captures/casts (Tony's rule):
        starting a camera capture requires an explicit user yes."""
        gated = (
            (name == "cast_to_screen"
             and str(args.get("action") or "").lower() == "uplink")
            or (name == "cctv_wall"
                and str(args.get("action") or "").lower()
                not in {"stop", "settings", "status"})
            # cctv_snapshot's seat: gated only when the call puts the
            # frame on a display — a describe-only snapshot stays open.
            or (name == "ada_camera_snapshot"
                and (args.get("screen")
                     or str(args.get("target") or "").strip().lower()
                     in {"tv", "screen"}))
        )
        if gated and not _confirmed_truthy(confirmed) and self._is_secondary_turn():
            logger.warning(
                "denied %s %r: camera capture by secondary speaker "
                "without confirmed=true", name, args)
            raise PermissionError(
                f"{name} starts a camera capture — a guest asked for it. "
                "Confirm with the session owner first, then call again "
                "with confirmed=true."
            )

    def _check_memory_write_allowed(
        self, name: str, args: dict[str, Any], confirmed: Any,
        confirm_token: Any = None,
    ) -> None:
        """Server-side gate for memory-bank writes. Raises PermissionError on denial."""
        if os.environ.get("ADA_READ_ONLY") == "true":
            logger.warning("denied %s %r: ADA_READ_ONLY", name, args)
            raise PermissionError("memory writes are disabled (ADA_READ_ONLY=true)")
        bank_name = str(args.get("bank") or "")
        try:
            bank = self.banks.bank(bank_name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        # Unified ACL resolver: person_policies + person-scope + writable +
        # allowed_tools decided in one walk (registry.acl_decision); the
        # messages below keep their per-layer shape for the model.
        ident = self.policy_identity()
        decision = self.banks.acl_decision(
            ident, bank=bank.name, tool=name, tool_aliases=_ALIASES)
        if not decision["allowed"]:
            layer = str(decision.get("decided_by") or "")
            if layer == "bank.person_scope":
                # Person-scope write privacy stays enforced where it
                # always was — memory_ops._check_person_scope_write runs
                # inside the tool and returns an {ok: False} result, not
                # a dispatch-layer raise. The resolver reports the layer
                # (acl_explain shows it); the gate defers enforcement.
                pass
            else:
                logger.warning(
                    "denied %s on %r for identity %r: %s",
                    name, bank.name, ident, layer,
                )
                if layer in ("person_scope", "person_policies"):
                    allowed = ", ".join(self.banks.banks_for_person(ident))
                    raise PermissionError(
                        f"memory bank '{bank.name}' is not available for "
                        "this speaker"
                        + (f" — allowed banks: {allowed}" if allowed else "")
                    )
                if layer == "bank.writable":
                    raise PermissionError(str(
                        decision.get("reason")
                        or f"memory bank '{bank_name}' is read-only"))
                if layer == "bank.allowed_tools":
                    raise PermissionError(
                        f"tool {name} is not allowed on memory bank "
                        f"'{bank_name}'")
                raise PermissionError(str(
                    decision.get("reason")
                    or f"{name} denied on '{bank_name}'"))
        if bank.write_policy == "confirmed":
            self._require_confirmation(
                name, args, confirmed, confirm_token,
                f"memory bank '{bank_name}' requires confirmation. Ask the user "
                "explicitly first, then call again with confirmed=true only "
                "after they say yes — do not write it to a different bank "
                "to get around the confirmation step.",
            )

    def _check_calendar_write_allowed(
        self, name: str, args: dict[str, Any], confirmed: Any,
        confirm_token: Any = None,
    ) -> None:
        """Server-side gate for calendar/task writes. Raises PermissionError on denial."""
        if name == "tasks":
            # Per-action split after the tools-merge-tasks-status collapse
            # — alias resolution ran before this gate, so name is
            # canonical: action='list' absorbed tasks_list, a free read
            # that was never confirm-gated (or READ_ONLY-blocked).
            if str(args.get("action") or "").strip().lower() == "list":
                return
        if os.environ.get("ADA_READ_ONLY") == "true":
            logger.warning("denied %s %r: ADA_READ_ONLY", name, args)
            raise PermissionError("calendar writes are disabled (ADA_READ_ONLY=true)")
        self._require_confirmation(
            name, args, confirmed, confirm_token,
            f"{name} requires confirmation. Restate the details to the user, "
            "get an explicit yes, then call again with confirmed=true.",
        )

    def _check_cms_write_allowed(
        self, name: str, args: dict[str, Any], confirmed: Any,
        confirm_token: Any = None,
    ) -> None:
        """Stateful gate for miniapp page writes. First attempt registers a
        pending confirmation; a resubmit with confirmed=true must match it —
        the model cannot jump straight to confirmed without the ask-step.
        Raises PermissionError on denial."""
        if name == "cms_edit":
            # Per-action split after the tools-merge-cms collapse — alias
            # resolution ran before this gate, so name is canonical:
            #   note     — absorbed cms_note_update: a merge-only timeline
            #              append that was never confirm-gated
            #   automate — absorbed cms_automation: op list/get read the
            #              registry, free like before; set/enable/disable/
            #              run fall through to confirmation
            #   delete   — absorbed cms_delete_page: confirmed
            eaction = str(args.get("action") or "").lower()
            if eaction == "note":
                return
            if (eaction == "automate" and str(args.get("op") or "").lower()
                    in CMS_AUTOMATION_READ_ACTIONS):
                return
        if os.environ.get("ADA_READ_ONLY") == "true":
            logger.warning("denied %s %r: ADA_READ_ONLY", name, args)
            raise PermissionError("CMS writes are disabled (ADA_READ_ONLY=true)")
        if name == "cms_publish_page":
            # Report meta contract (ssot.apps.ada-cms-reports.yml): reject
            # BEFORE the confirm handshake registers so the model fixes the
            # call instead of asking the user to approve a write that would
            # land without the fields reports-index/verify rely on.
            missing = missing_publish_fields(args)
            if missing:
                raise ValueError(
                    "cms_publish_page: report meta contract — missing or "
                    f"invalid: {', '.join(missing)}. Pass summary (one-line "
                    "brief, ~240 chars), domain (grouping tag), fresh_for "
                    "('30m'/'1h'/'6h'/'1d'), and confidence "
                    "(high|medium|low|unverified); updated and timeline are "
                    "stamped automatically.")
            # Normalize the slug before keying — the model may resubmit the
            # confirm call with different casing/spacing than the register
            # call, and a raw-args key would miss the pending request.
            try:
                slug = self._cms_slug(str(args.get("slug") or ""))
            except ValueError:
                # Register/deny must not raise on an invalid slug — the
                # publish path itself validates and rejects with ValueError.
                slug = str(args.get("slug") or "")
            # Keyed by slug:lang — confirming an EN publish does not unlock
            # a different-language variant of the same slug.
            pkey = f"{slug}:{args.get('lang') or 'en'}"
            pending = getattr(self, "_cms_pending", None)
            if pending is None:
                pending = self._cms_pending = {}
            if _confirmed_truthy(confirmed):
                if pending.get(pkey) is not None or _CALLER_VERIFIED_AFFIRM.get():
                    # A provider-verified user affirmation already covers the
                    # ask-step — the register round-trip is friction, not
                    # safety (2026-09-29: user said 'อนุมัติ', model called
                    # confirmed=true, denied 'no pending' twice).
                    pending.pop(pkey, None)
                    return
                logger.warning("denied %s %r: confirmed without pending request", name, args)
                raise PermissionError(
                    f"{name}: confirmed=true has no pending request for '{slug}'. "
                    "First call without confirmed to register the request, ask the "
                    "user to confirm, then resubmit with confirmed=true."
                )
            pending[pkey] = True
            logger.info("cms pending-confirm registered: %s", pkey)
            raise PermissionError(
                f"{name} requires confirmation — request registered. Now tell the "
                "user the page slug and title, ask for an explicit yes, then "
                "call again with the SAME args plus confirmed=true."
            )
        self._require_confirmation(
            name, args, confirmed, confirm_token,
            f"{name} requires confirmation. Restate the page slug, title, and "
            "what will change, get an explicit yes, then call again with confirmed=true.",
        )

    def _check_dynamic_allowed(
        self, name: str, spec: Any, args: dict[str, Any],
        confirmed: Any, confirm_token: Any = None,
    ) -> None:
        """Policy gate for tools.d drop-in tools. Raises PermissionError."""
        if spec.policy == "owner_only":
            ident = self.policy_identity()
            policy = self.banks.policy_for(ident) or {}
            if not policy.get("full"):
                logger.warning("denied %s for identity %r: owner_only tool",
                               name, ident)
                raise PermissionError(
                    f"{name} is restricted to the owner's identities")
        if spec.policy == "confirmed":
            self._require_confirmation(
                name, args, confirmed, confirm_token,
                f"{name} requires confirmation. Describe what it will do, "
                "get an explicit yes, then call again with confirmed=true.",
            )

    def _check_devin_confirmed(
        self, name: str, args: dict[str, Any], confirmed: Any,
        confirm_token: Any = None,
    ) -> None:
        """Server-side gate for devin session control (the canonical
        `devin` tool — dispatch/followup/answer). PermissionError on denial."""
        if os.environ.get("ADA_READ_ONLY") == "true":
            logger.warning("denied %s %r: ADA_READ_ONLY", name, args)
            raise PermissionError("devin tools are disabled (ADA_READ_ONLY=true)")
        self._require_confirmation(
            name, args, confirmed, confirm_token,
            f"{name} requires confirmation. Restate the repo, task, and "
            "that an unattended Devin session will make code changes, get an "
            "explicit yes, then call again with confirmed=true.",
        )

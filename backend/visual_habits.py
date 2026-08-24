"""Durable visual habit monitors sharing the camera and pose pipelines."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import statistics
import time
import uuid
from collections import deque
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("voice.visual_habits")

DEFAULT_SETTINGS = {
    "sitting_too_long": {"enabled": True, "maximum_sitting_minutes": 60, "break_reset_minutes": 5},
    "phone_distraction": {"enabled": True, "confirmation_minutes": 2, "reset_minutes": 2},
    "desk_clutter": {"enabled": True, "check_interval_seconds": 60, "sustained_change_minutes": 10, "reset_minutes": 5},
    "working_too_late": {"enabled": True, "cutoff_time": "22:00", "morning_reset_time": "06:00"},
    "not_drinking_enough_water": {"enabled": True, "reminder_interval_minutes": 60, "response_window_seconds": 15},
    "junk_food": {"enabled": True, "observation_window_seconds": 15, "reset_minutes": 30},
}

def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).astimezone().isoformat()


def desk_descriptor(jpeg: bytes) -> list[float]:
    """Return a compact, lighting-normalized descriptor; never retains pixels."""
    with Image.open(io.BytesIO(jpeg)) as source:
        image = source.convert("RGB").resize((32, 24), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        # Normalize global illumination while retaining local color/layout changes.
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        std = np.maximum(rgb.std(axis=(0, 1), keepdims=True), 0.05)
        normalized = np.clip((rgb - mean) / std, -2.5, 2.5)
        blocks = normalized.reshape(6, 4, 8, 4, 3).mean(axis=(1, 3)).reshape(-1)
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        gray = (gray - gray.mean()) / max(float(gray.std()), 0.05)
        structure = gray.reshape(6, 4, 8, 4).mean(axis=(1, 3)).reshape(-1)
        edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        edge_blocks = edges.reshape(6, 4, 8, 4).mean(axis=(1, 3)).reshape(-1)
    return np.concatenate((blocks, structure, edge_blocks)).round(5).tolist()


def descriptor_change(current: list[float], baseline: list[float]) -> float:
    if not current or len(current) != len(baseline):
        return 1.0
    return float(np.mean(np.abs(np.asarray(current) - np.asarray(baseline))))


def phone_near_user(detections: list[dict[str, Any]], pose: dict[str, Any] | None) -> bool:
    if not pose:
        return False
    anchors = [p for p in pose.get("keypoints", []) if p.get("name") in {
        "left_wrist", "right_wrist", "nose", "left_ear", "right_ear", "left_eye", "right_eye"
    } and float(p.get("score", 0)) >= .25]
    for item in detections:
        if item.get("label") != "cell phone" or float(item.get("score", 0)) < .45:
            continue
        x, y = float(item["x"]), float(item["y"])
        w, h = float(item["width"]), float(item["height"])
        cx, cy = x + w / 2, y + h / 2
        margin = max(.10, min(.22, max(w, h) * 1.25))
        if any(abs(float(p["x"]) - cx) <= w / 2 + margin and abs(float(p["y"]) - cy) <= h / 2 + margin for p in anchors):
            return True
    return False


def hand_near_face(pose: dict[str, Any] | None) -> bool:
    """Return a conservative local hand-to-mouth cue for Gemini review."""
    if not pose:
        return False
    points = {p.get("name"): p for p in pose.get("keypoints", []) if float(p.get("score", 0)) >= .25}
    faces = [points[name] for name in ("nose", "left_ear", "right_ear", "left_eye", "right_eye") if name in points]
    wrists = [points[name] for name in ("left_wrist", "right_wrist") if name in points]
    return any(((float(w["x"])-float(f["x"]))**2 + (float(w["y"])-float(f["y"]))**2) ** .5 <= .16 for w in wrists for f in faces)


def _valid_clock(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":": return False
    try: hour, minute = map(int, value.split(":"))
    except ValueError: return False
    return 0 <= hour < 24 and 0 <= minute < 60


class VisualHabitService:
    """Latest-frame-only coordinator for sitting, phone, clutter and alerts."""

    def __init__(self, camera: Any, pose_service: Any, detector: Any, store: Any,
                 notifier: Any = None, clutter_verifier: Any = None) -> None:
        self.camera, self.pose_service, self.detector, self.store = camera, pose_service, detector, store
        self.notifier, self.clutter_verifier = notifier, clutter_verifier
        self.timezone = ZoneInfo(os.environ.get("ADA_TIMEZONE", "America/New_York"))
        self.settings = {key: {**defaults, **store.monitor_state(f"{key}_settings")} for key, defaults in DEFAULT_SETTINGS.items()}
        self.states = {key: store.monitor_state(key) for key in DEFAULT_SETTINGS}
        for key in ("not_drinking_enough_water", "junk_food"):
            if self.states[key].pop("challenge", None):
                self.states[key].update(state="inconclusive", retry_after=time.time() + 300)
                store.save_monitor_state(key, self.states[key])
        self.latest_detection: dict[str, Any] | None = None
        self.latest_detection_frame: bytes | None = None
        self._detection_generation = 0
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._challenge_task: asyncio.Task[None] | None = None
        self._calibrating = False
        self._calibration_samples: list[list[float]] = []
        self._calibration_error: str | None = None
        self._last_clutter_check = 0.0

    async def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._challenge_task:
            self._challenge_task.cancel()
            with suppress(asyncio.CancelledError): await self._challenge_task
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError): await self._task
        self._task = None
        self._challenge_task = None

    def update_settings(self, key: str, values: dict[str, Any]) -> dict[str, Any]:
        if key not in DEFAULT_SETTINGS or not values or set(values) - set(DEFAULT_SETTINGS[key]):
            raise ValueError("Unsupported monitor or setting")
        merged = {**self.settings[key], **values}
        if not isinstance(merged["enabled"], bool): raise ValueError("enabled must be boolean")
        for name in ("cutoff_time", "morning_reset_time"):
            if name in merged and not _valid_clock(merged[name]): raise ValueError(f"{name} must use HH:MM")
        bounds = {"maximum_sitting_minutes": (1, 720), "break_reset_minutes": (1, 120),
                  "confirmation_minutes": (1, 60), "reset_minutes": (1, 120),
                  "check_interval_seconds": (10, 3600), "sustained_change_minutes": (1, 120),
                  "reminder_interval_minutes": (5, 240), "response_window_seconds": (5, 60),
                  "observation_window_seconds": (5, 60)}
        for name, value in merged.items():
            if name in {"enabled", "cutoff_time", "morning_reset_time"}: continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not bounds[name][0] <= value <= bounds[name][1]:
                raise ValueError(f"{name} must be between {bounds[name][0]} and {bounds[name][1]}")
        self.settings[key] = merged
        self.store.save_monitor_state(f"{key}_settings", merged)
        return merged

    def begin_calibration(self) -> dict[str, Any]:
        self._calibrating, self._calibration_samples, self._calibration_error = True, [], None
        return self.snapshot()["desk_clutter"]

    async def trigger_check(self, key: str, now: float | None = None) -> dict[str, Any]:
        """Run a user-requested check without changing its saved schedule."""
        now = time.time() if now is None else now
        if key == "not_drinking_enough_water":
            if self._challenge_task and not self._challenge_task.done():
                raise RuntimeError("Another Gemini observation is already running")
            self._start_challenge(key, now)
        elif key == "working_too_late":
            night = self._night_key(now)
            if not self._presence():
                await self.notifier.send_text_turn("Manual late-work check: explain briefly that Ada cannot confirm desk work because recent visual presence is unavailable. Do not record a habit.")
            elif night is None:
                await self.notifier.send_text_turn("Manual late-work check: tell the user briefly that they are at the desk, but it is not past the configured nightly cutoff, so no habit occurrence was recorded.")
            elif self.states[key].get("night_key") == night:
                await self.notifier.send_text_turn("Manual late-work check: tell the user briefly that tonight's late-work occurrence was already recorded and will not be counted twice.")
            else:
                await self._record(key, now, {"manual_check": True, "night_key": night})
                self.states[key]["night_key"] = night
                self._save(key)
        else:
            raise ValueError("This monitor does not support a manual check")
        return self.snapshot()[key]

    def clear_history(self) -> None:
        self.states = {key: {} for key in DEFAULT_SETTINGS}
        if self._challenge_task: self._challenge_task.cancel()
        self._challenge_task = None

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        result = {}
        for key in DEFAULT_SETTINGS:
            state = self.states[key]
            candidate = float(state.get("candidate_since", 0) or 0)
            target = (self.settings[key].get("maximum_sitting_minutes") or self.settings[key].get("confirmation_minutes") or self.settings[key].get("sustained_change_minutes", 1)) * 60
            progress = round(min(1, max(0, now-candidate) / target), 3) if candidate else 0
            if key == "not_drinking_enough_water":
                target = self.settings[key]["reminder_interval_minutes"] * 60
                progress = round(min(1, float(state.get("accumulated_seconds", 0)) / target), 3)
            elif key == "junk_food" and state.get("challenge"):
                target = self.settings[key]["observation_window_seconds"]
                progress = round(min(1, max(0, now-float(state["challenge"].get("started_at", now))) / target), 3)
            elif key == "working_too_late":
                progress = self._late_progress(now)
            result[key] = {"habit_key": key, "settings": dict(self.settings[key]), "state": state.get("state", "idle"),
                           "latched": bool(state.get("latched")), "progress": progress,
                           "calibrated": bool(self.store.monitor_state("desk_clutter_calibration")) if key == "desk_clutter" else True}
            if state.get("challenge"): result[key]["challenge"] = dict(state["challenge"])
        result["desk_clutter"].update(calibrating=self._calibrating, calibration_samples=len(self._calibration_samples), calibration_error=self._calibration_error)
        return result

    def _save(self, key: str) -> None: self.store.save_monitor_state(key, self.states[key])

    async def _record(self, key: str, now: float, details: dict[str, Any], latch: bool = True) -> None:
        alert = self.store.record_habit_occurrence(key, _stamp(now), details)
        notification = self.store.create_notification(key, alert["event_id"], alert, voice_pending=True)
        self.states[key].update(latched=latch, state="alerted", candidate_since=None, recovery_since=None)
        self._save(key)
        if key != "desk_clutter" or self.pose_service.office_occupied() is True:
            await self._deliver(notification)

    async def _deliver(self, notification: dict[str, Any]) -> None:
        if not self.notifier or notification.get("voice_delivered"): return
        key = notification["habit_key"]
        corrections = {"sitting_too_long": "stand up and take a five-minute movement break",
                       "phone_distraction": "put the phone out of reach and return to the task",
                       "desk_clutter": "clear the changed work area before starting the next task",
                       "working_too_late": "wrap up, write down the next step, and leave the desk",
                       "not_drinking_enough_water": "drink a glass now and keep water within reach",
                       "junk_food": "put the snack away and choose water or a less processed option"}
        message = f"ADA habit alert: {key.replace('_', ' ')} was confirmed. Give one concise, restrained sarcastic observation, then suggest they {corrections[key]}."
        try:
            await self.notifier.send_text_turn(message)
            self.store.mark_notification_voice(int(notification["id"]))
        except Exception as exc: logger.warning("visual habit speech failed: %s", exc)

    def _presence(self) -> bool:
        return self.pose_service.office_occupied() is True

    def _clock(self, value: str) -> tuple[int, int]:
        return tuple(map(int, value.split(":")))  # type: ignore[return-value]

    def _night_key(self, now: float) -> str | None:
        local = datetime.fromtimestamp(now, self.timezone)
        cutoff_h, cutoff_m = self._clock(self.settings["working_too_late"]["cutoff_time"])
        reset_h, reset_m = self._clock(self.settings["working_too_late"]["morning_reset_time"])
        minute = local.hour * 60 + local.minute
        cutoff, reset = cutoff_h * 60 + cutoff_m, reset_h * 60 + reset_m
        if cutoff > reset:
            if minute >= cutoff: return local.date().isoformat()
            if minute < reset: return (local.date() - timedelta(days=1)).isoformat()
            return None
        return local.date().isoformat() if cutoff <= minute < reset else None

    def _late_progress(self, now: float) -> float:
        if self._night_key(now): return 1.0
        local = datetime.fromtimestamp(now, self.timezone)
        hour, minute = self._clock(self.settings["working_too_late"]["cutoff_time"])
        cutoff = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if cutoff <= local: cutoff += timedelta(days=1)
        return round(max(0, 1 - (cutoff-local).total_seconds() / 86400), 3)

    async def _run(self) -> None:
        await self.camera.start()
        while True:
            started, now = time.monotonic(), time.time()
            frame = self.camera.latest_frame
            presence = self._presence()
            await self._process_sitting(now, presence)
            await self._process_late_work(now, presence)
            await self._process_water(now, presence)
            if frame and (self.settings["phone_distraction"]["enabled"] or self.settings["junk_food"]["enabled"]):
                result = await asyncio.to_thread(self.detector.infer, frame)
                self.latest_detection, self.latest_detection_frame = result, frame
                async with self._condition:
                    self._detection_generation += 1; self._condition.notify_all()
                if self.settings["phone_distraction"]["enabled"]:
                    await self._process_phone(now, phone_near_user(result.get("detections", []), self.pose_service.latest_result))
            await self._process_junk_food(now, hand_near_face(self.pose_service.latest_result))
            if frame: await self._process_clutter(now, frame, presence)
            if presence:
                for note in self.store.pending_voice_notifications(): await self._deliver(note)
            await asyncio.sleep(max(0, 2 - (time.monotonic() - started)))

    async def _process_sitting(self, now: float, present: bool) -> None:
        key, state, cfg = "sitting_too_long", self.states["sitting_too_long"], self.settings["sitting_too_long"]
        if not cfg["enabled"]: return
        if present:
            state["last_present"] = now; state["recovery_since"] = None
            if not state.get("latched"):
                state["candidate_since"] = state.get("candidate_since") or now; state["state"] = "tracking"
                if now - state["candidate_since"] >= cfg["maximum_sitting_minutes"] * 60: await self._record(key, now, {"duration_seconds": now-state["candidate_since"]})
        else:
            if now - float(state.get("last_present", 0)) > 15 and state.get("candidate_since") and not state.get("latched"): state.update(candidate_since=None, state="idle")
            if state.get("latched"):
                state["recovery_since"] = state.get("recovery_since") or now
                if now-state["recovery_since"] >= cfg["break_reset_minutes"]*60: state.update(latched=False, state="idle", candidate_since=None, recovery_since=None)
        self._save(key)

    async def _process_phone(self, now: float, positive: bool) -> None:
        key, state, cfg = "phone_distraction", self.states["phone_distraction"], self.settings["phone_distraction"]
        samples = [(float(t), bool(v)) for t,v in state.get("samples", []) if now-float(t) <= cfg["confirmation_minutes"]*60]
        samples.append((now, positive)); state["samples"] = samples
        if positive: state["last_positive"], state["recovery_since"] = now, None
        if state.get("latched"):
            if not positive:
                state["recovery_since"] = state.get("recovery_since") or now
                if now-state["recovery_since"] >= cfg["reset_minutes"]*60: state.update(latched=False, state="idle", recovery_since=None, samples=[])
        elif samples:
            span = samples[-1][0]-samples[0][0]; ratio = sum(v for _,v in samples)/len(samples)
            state["candidate_since"], state["state"] = samples[0][0], "tracking"
            if span >= cfg["confirmation_minutes"]*60-2.5 and ratio >= .70 and now-float(state.get("last_positive", 0)) <= 15: await self._record(key, now, {"positive_ratio": ratio})
        self._save(key)

    async def _process_late_work(self, now: float, present: bool) -> None:
        key, state, cfg = "working_too_late", self.states["working_too_late"], self.settings["working_too_late"]
        if not cfg["enabled"]: return
        night = self._night_key(now)
        if night is None:
            state.update(state="waiting", latched=False)
        elif state.get("night_key") == night:
            state.update(state="alerted", latched=True)
        elif present:
            await self._record(key, now, {"night_key": night, "cutoff_time": cfg["cutoff_time"]})
            state["night_key"] = night
        else:
            state.update(state="late_window", latched=False)
        self._save(key)

    async def _process_water(self, now: float, present: bool) -> None:
        key, state, cfg = "not_drinking_enough_water", self.states["not_drinking_enough_water"], self.settings["not_drinking_enough_water"]
        if not cfg["enabled"]: return
        previous = float(state.get("last_tick", now)); delta = max(0, min(5, now-previous)); state["last_tick"] = now
        if present: state["last_present"] = now
        effectively_present = present or now-float(state.get("last_present", 0)) <= 15
        if effectively_present and not state.get("challenge"):
            state["accumulated_seconds"] = float(state.get("accumulated_seconds", 0)) + delta
            state["state"] = "tracking"
        elif not state.get("challenge"):
            state["state"] = "paused"
        due = float(state.get("accumulated_seconds", 0)) >= cfg["reminder_interval_minutes"] * 60
        if due and present and now >= float(state.get("retry_after", 0)):
            self._start_challenge(key, now)
        self._save(key)

    async def _process_junk_food(self, now: float, cue: bool) -> None:
        key, state, cfg = "junk_food", self.states["junk_food"], self.settings["junk_food"]
        if not cfg["enabled"]: return
        if cue: state["last_evidence"] = now
        if state.get("latched"):
            if now-float(state.get("last_evidence", now)) >= cfg["reset_minutes"]*60:
                state.update(latched=False, state="idle", cues=[])
            else: state["state"] = "alerted"
            self._save(key); return
        cues = [float(value) for value in state.get("cues", []) if now-float(value) <= 20]
        if cue: cues.append(now)
        state["cues"] = cues
        if len(cues) >= 3 and now >= float(state.get("retry_after", 0)) and not state.get("challenge"):
            self._start_challenge(key, now); state["cues"] = []; self._save(key); return
        elif cues: state["state"] = "gesture_detected"
        else: state["state"] = "idle"
        self._save(key)

    def _start_challenge(self, key: str, now: float) -> None:
        if self._challenge_task and not self._challenge_task.done(): return
        challenge = {"id": uuid.uuid4().hex, "habit_key": key, "phase": "prompting", "created_at": now}
        self.states[key]["challenge"] = challenge; self.states[key]["state"] = "prompting"; self._save(key)
        self._challenge_task = asyncio.create_task(self._run_challenge(key, challenge))
        task = self._challenge_task
        task.add_done_callback(lambda finished: setattr(self, "_challenge_task", None) if self._challenge_task is finished else None)

    async def _next_matching(self, queue: asyncio.Queue[Any], kind: str, timeout: float, challenge_id: str | None = None) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            event = await asyncio.wait_for(queue.get(), max(.01, deadline-time.monotonic()))
            if event.type != kind: continue
            if challenge_id is None or event.data.get("challenge_id") == challenge_id: return event

    async def _run_challenge(self, key: str, challenge: dict[str, Any]) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        collector: asyncio.Task[None] | None = None
        async def collect() -> None:
            async for event in self.notifier.events(): await queue.put(event)
        try:
            if not self.notifier or not all(hasattr(self.notifier, name) for name in ("events", "send_text_turn", "send_video")):
                raise RuntimeError("Gemini Live observation is unavailable")
            collector = asyncio.create_task(collect()); await asyncio.sleep(0)
            if key == "not_drinking_enough_water":
                prompt = "Hydration check: ask the user, briefly and clearly, to drink water now. Do not judge yet; an observation window will follow."
                window = int(self.settings[key]["response_window_seconds"])
            else:
                prompt = "A possible eating gesture was detected. Briefly tell the user Ada is checking the current snack, then wait for the observation window."
                window = int(self.settings[key]["observation_window_seconds"])
            await self.notifier.send_text_turn(prompt)
            await self._next_matching(queue, "response_completed", 20)
            challenge.update(phase="observing", started_at=time.time(), deadline=time.time()+window)
            self.states[key]["state"] = "observing"; self._save(key)
            frames_sent = 0
            last_generation = -1
            for _ in range(window):
                frame = self.camera.latest_frame
                generation = int(getattr(self.camera, "generation", last_generation + 1))
                frame_at = getattr(self.camera, "latest_frame_at", time.monotonic())
                if frame and generation != last_generation and frame_at is not None and time.monotonic()-frame_at <= 2:
                    await self.notifier.send_video(frame); frames_sent += 1
                    last_generation = generation
                await asyncio.sleep(1)
            if frames_sent < max(3, window//3): raise RuntimeError("Not enough camera frames for a fair verdict")
            challenge["phase"] = "reviewing"; self.states[key]["state"] = "reviewing"; self._save(key)
            criteria = ("whether the user visibly drank water from a cup or bottle" if key == "not_drinking_enough_water" else
                        "whether the user visibly consumed a specifically identifiable broadly unhealthy food such as a sugary drink, candy, chips, cookies, pastries, dessert, fast food, or a heavily processed snack or meal. The hand-to-face gesture that triggered this review is not evidence. Face touching, scratching, nail biting, an empty hand, chewing with no identifiable item, holding food, and food merely being present must all return observed=false")
            await self.notifier.send_text_turn(
                f"Silently review the complete observation window and decide {criteria}. "
                f"Call report_habit_observation exactly once with challenge_id='{challenge['id']}', habit_key='{key}', observed, confidence, reason, item_identified, consumption_visible, and classified_unhealthy. "
                "For junk food, observed may be true only when item_identified is specific and both consumption_visible and classified_unhealthy are true. Do not speak a verdict."
            )
            event = await self._next_matching(queue, "habit_observation", 20, challenge["id"])
            await self._apply_observation(key, challenge, event.data)
        except asyncio.CancelledError: raise
        except Exception as exc:
            logger.warning("%s observation inconclusive: %s", key, exc)
            state = self.states[key]; state.pop("challenge", None); state.update(state="inconclusive", retry_after=time.time()+300); self._save(key)
        finally:
            if collector:
                collector.cancel()
                with suppress(asyncio.CancelledError): await collector

    async def _apply_observation(self, key: str, challenge: dict[str, Any], verdict: dict[str, Any]) -> None:
        state = self.states[key]; state.pop("challenge", None)
        if verdict.get("habit_key") != key or verdict.get("challenge_id") != challenge["id"]:
            state.update(state="inconclusive", retry_after=time.time()+300); self._save(key); return
        confidence = max(0, min(1, float(verdict.get("confidence", 0))))
        minimum_confidence = .75 if key == "junk_food" else .65
        if confidence < minimum_confidence:
            state.update(state="inconclusive", retry_after=time.time()+300); self._save(key); return
        observed, now = bool(verdict.get("observed")), time.time()
        details = {"gemini_confidence": confidence, "gemini_reason": str(verdict.get("reason", ""))[:300], "challenge_id": challenge["id"]}
        if key == "not_drinking_enough_water":
            state["accumulated_seconds"] = 0; state["retry_after"] = 0
            if observed:
                state.update(state="completed", latched=False); self._save(key)
            else:
                await self._record(key, now, details, latch=False)
        elif (observed and verdict.get("consumption_visible") is True
              and verdict.get("classified_unhealthy") is True
              and bool(str(verdict.get("item_identified", "")).strip())):
            state["last_evidence"] = now
            details["item_identified"] = str(verdict["item_identified"])[:100]
            await self._record(key, now, details)
        else:
            state.update(state="not_confirmed", retry_after=now+300, latched=False); self._save(key)

    async def _process_clutter(self, now: float, frame: bytes, present: bool) -> None:
        key, state, cfg = "desk_clutter", self.states["desk_clutter"], self.settings["desk_clutter"]
        state.setdefault("last_person", now)
        if present: state["last_person"] = now
        if self._calibrating:
            if present or now-float(state.get("last_person", now)) < 30: return
            descriptor = desk_descriptor(frame)
            if self._calibration_samples and descriptor_change(descriptor, self._calibration_samples[-1]) > .16:
                self._calibrating, self._calibration_error = False, "Scene moved during calibration. Leave the clean desk still and try again."
                return
            self._calibration_samples.append(descriptor)
            if len(self._calibration_samples) >= 5:
                baseline = np.median(np.asarray(self._calibration_samples), axis=0).round(5).tolist()
                self.store.save_monitor_state("desk_clutter_calibration", {"descriptor": baseline, "created_at": _stamp(now)})
                self._calibrating, self._calibration_samples = False, []
                state.update(latched=False, state="idle", candidate_since=None, recovery_since=None); self._save(key)
            return
        baseline = self.store.monitor_state("desk_clutter_calibration").get("descriptor")
        if not cfg["enabled"] or not baseline or present or now-float(state.get("last_person", now)) < 30 or now-self._last_clutter_check < cfg["check_interval_seconds"]: return
        self._last_clutter_check = now
        score = descriptor_change(desk_descriptor(frame), baseline); state["change_score"] = round(score, 4)
        changed = score >= .18
        if state.get("latched"):
            if not changed:
                state["recovery_since"] = state.get("recovery_since") or now
                if now-state["recovery_since"] >= cfg["reset_minutes"]*60: state.update(latched=False,state="idle",candidate_since=None,recovery_since=None)
            else: state["recovery_since"] = None
        elif changed:
            state["candidate_since"] = state.get("candidate_since") or now; state["state"] = "changed"
            if now-state["candidate_since"] >= cfg["sustained_change_minutes"]*60 and self.clutter_verifier:
                try:
                    verdict = await self.clutter_verifier.verify_clutter(frame)
                    if verdict.get("cluttered") and float(verdict.get("confidence",0)) >= .65: await self._record(key, now, {"change_score":score,"gemini_confidence":verdict["confidence"],"gemini_reason":verdict.get("reason","")})
                    else: state.update(candidate_since=None,state="idle")
                except Exception as exc:
                    state["state"] = "verification_unavailable"; state["verification_error"] = str(exc)[:300]
        else: state.update(candidate_since=None,state="idle")
        self._save(key)

    async def detections(self):
        await self.start(); generation = self._detection_generation
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: generation != self._detection_generation)
                generation = self._detection_generation
            if self.latest_detection and self.latest_detection_frame: yield self.latest_detection, self.latest_detection_frame

"""Durable visual habit monitors sharing the camera and pose pipelines."""

from __future__ import annotations

import asyncio
import io
import logging
import statistics
import time
from collections import deque
from contextlib import suppress
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("voice.visual_habits")

DEFAULT_SETTINGS = {
    "sitting_too_long": {"enabled": True, "maximum_sitting_minutes": 60, "break_reset_minutes": 5},
    "phone_distraction": {"enabled": True, "confirmation_minutes": 2, "reset_minutes": 2},
    "desk_clutter": {"enabled": True, "check_interval_seconds": 60, "sustained_change_minutes": 10, "reset_minutes": 5},
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


class VisualHabitService:
    """Latest-frame-only coordinator for sitting, phone, clutter and alerts."""

    def __init__(self, camera: Any, pose_service: Any, detector: Any, store: Any,
                 notifier: Any = None, clutter_verifier: Any = None) -> None:
        self.camera, self.pose_service, self.detector, self.store = camera, pose_service, detector, store
        self.notifier, self.clutter_verifier = notifier, clutter_verifier
        self.settings = {key: {**defaults, **store.monitor_state(f"{key}_settings")} for key, defaults in DEFAULT_SETTINGS.items()}
        self.states = {key: store.monitor_state(key) for key in DEFAULT_SETTINGS}
        self.latest_detection: dict[str, Any] | None = None
        self.latest_detection_frame: bytes | None = None
        self._detection_generation = 0
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._calibrating = False
        self._calibration_samples: list[list[float]] = []
        self._calibration_error: str | None = None
        self._last_clutter_check = 0.0

    async def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError): await self._task
        self._task = None

    def update_settings(self, key: str, values: dict[str, Any]) -> dict[str, Any]:
        if key not in DEFAULT_SETTINGS or not values or set(values) - set(DEFAULT_SETTINGS[key]):
            raise ValueError("Unsupported monitor or setting")
        merged = {**self.settings[key], **values}
        if not isinstance(merged["enabled"], bool): raise ValueError("enabled must be boolean")
        bounds = {"maximum_sitting_minutes": (1, 720), "break_reset_minutes": (1, 120),
                  "confirmation_minutes": (1, 60), "reset_minutes": (1, 120),
                  "check_interval_seconds": (10, 3600), "sustained_change_minutes": (1, 120)}
        for name, value in merged.items():
            if name == "enabled": continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not bounds[name][0] <= value <= bounds[name][1]:
                raise ValueError(f"{name} must be between {bounds[name][0]} and {bounds[name][1]}")
        self.settings[key] = merged
        self.store.save_monitor_state(f"{key}_settings", merged)
        return merged

    def begin_calibration(self) -> dict[str, Any]:
        self._calibrating, self._calibration_samples, self._calibration_error = True, [], None
        return self.snapshot()["desk_clutter"]

    def clear_history(self) -> None:
        self.states = {key: {} for key in DEFAULT_SETTINGS}

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        result = {}
        for key in DEFAULT_SETTINGS:
            state = self.states[key]
            candidate = float(state.get("candidate_since", 0) or 0)
            target = (self.settings[key].get("maximum_sitting_minutes") or self.settings[key].get("confirmation_minutes") or self.settings[key].get("sustained_change_minutes", 1)) * 60
            result[key] = {"habit_key": key, "settings": dict(self.settings[key]), "state": state.get("state", "idle"),
                           "latched": bool(state.get("latched")), "progress": round(min(1, max(0, now-candidate) / target), 3) if candidate else 0,
                           "calibrated": bool(self.store.monitor_state("desk_clutter_calibration")) if key == "desk_clutter" else True}
        result["desk_clutter"].update(calibrating=self._calibrating, calibration_samples=len(self._calibration_samples), calibration_error=self._calibration_error)
        return result

    def _save(self, key: str) -> None: self.store.save_monitor_state(key, self.states[key])

    async def _record(self, key: str, now: float, details: dict[str, Any]) -> None:
        alert = self.store.record_habit_occurrence(key, _stamp(now), details)
        notification = self.store.create_notification(key, alert["event_id"], alert, voice_pending=True)
        self.states[key].update(latched=True, state="alerted", candidate_since=None, recovery_since=None)
        self._save(key)
        if key != "desk_clutter" or self.pose_service.office_occupied() is True:
            await self._deliver(notification)

    async def _deliver(self, notification: dict[str, Any]) -> None:
        if not self.notifier or notification.get("voice_delivered"): return
        key = notification["habit_key"]
        corrections = {"sitting_too_long": "stand up and take a five-minute movement break",
                       "phone_distraction": "put the phone out of reach and return to the task",
                       "desk_clutter": "clear the changed work area before starting the next task"}
        message = f"ADA habit alert: {key.replace('_', ' ')} was confirmed. Give one concise, restrained sarcastic observation, then suggest they {corrections[key]}."
        try:
            await self.notifier.send_text_turn(message)
            self.store.mark_notification_voice(int(notification["id"]))
        except Exception as exc: logger.warning("visual habit speech failed: %s", exc)

    def _presence(self) -> bool:
        return self.pose_service.office_occupied() is True

    async def _run(self) -> None:
        await self.camera.start()
        while True:
            started, now = time.monotonic(), time.time()
            frame = self.camera.latest_frame
            presence = self._presence()
            await self._process_sitting(now, presence)
            if frame and self.settings["phone_distraction"]["enabled"]:
                result = await asyncio.to_thread(self.detector.infer, frame)
                self.latest_detection, self.latest_detection_frame = result, frame
                async with self._condition:
                    self._detection_generation += 1; self._condition.notify_all()
                await self._process_phone(now, phone_near_user(result.get("detections", []), self.pose_service.latest_result))
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

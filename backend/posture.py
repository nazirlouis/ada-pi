"""Calibrated, temporal slouch detection and local event persistence."""

from __future__ import annotations

import asyncio
import json
import math
import logging
import sqlite3
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("voice.posture")


FEATURES = (
    "head_forward", "neck_angle", "torso_angle", "torso_length", "head_height",
)
POSTURE_THRESHOLDS = (0.40, 0.50)


class PostureStore:
    """Small synchronous SQLite store; operations are short and lock-protected."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS posture_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    sensitivity TEXT NOT NULL DEFAULT 'normal',
                    start_hour INTEGER NOT NULL DEFAULT 8,
                    end_hour INTEGER NOT NULL DEFAULT 23,
                    cooldown_minutes INTEGER NOT NULL DEFAULT 5
                );
                INSERT OR IGNORE INTO posture_settings(id) VALUES (1);
                CREATE TABLE IF NOT EXISTS posture_calibration (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    created_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    head_forward REAL NOT NULL,
                    neck_angle REAL NOT NULL,
                    torso_angle REAL NOT NULL,
                    torso_length REAL NOT NULL,
                    head_height REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS posture_slouch_calibration (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    created_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    head_forward REAL NOT NULL,
                    neck_angle REAL NOT NULL,
                    torso_angle REAL NOT NULL,
                    torso_length REAL NOT NULL,
                    head_height REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ada_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    system_prompt TEXT
                );
                INSERT OR IGNORE INTO ada_settings(id, system_prompt) VALUES (1, NULL);
                CREATE TABLE IF NOT EXISTS habit_profiles (
                    habit_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    rolling_occurrences INTEGER NOT NULL,
                    rolling_days INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS posture_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds REAL,
                    average_score REAL NOT NULL DEFAULT 0,
                    worst_score REAL NOT NULL DEFAULT 0,
                    valid_sample_rate REAL NOT NULL DEFAULT 0,
                    reminded INTEGER NOT NULL DEFAULT 0,
                    correction TEXT,
                    gemini_confidence REAL,
                    gemini_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS habit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    habit_key TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    correction TEXT
                );
                CREATE INDEX IF NOT EXISTS habit_events_key_started
                    ON habit_events(habit_key, started_at);
                CREATE TABLE IF NOT EXISTS monitor_state (
                    monitor_key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS habit_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    habit_key TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    visual_acknowledged INTEGER NOT NULL DEFAULT 0,
                    voice_delivered INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(habit_key, event_id)
                );
            """)
            event_columns = {row[1] for row in db.execute("PRAGMA table_info(posture_events)")}
            if "gemini_confidence" not in event_columns:
                db.execute("ALTER TABLE posture_events ADD COLUMN gemini_confidence REAL")
            if "gemini_reason" not in event_columns:
                db.execute("ALTER TABLE posture_events ADD COLUMN gemini_reason TEXT")
            setting_columns = {row[1] for row in db.execute("PRAGMA table_info(posture_settings)")}
            if "cooldown_minutes" not in setting_columns:
                db.execute("ALTER TABLE posture_settings ADD COLUMN cooldown_minutes INTEGER NOT NULL DEFAULT 5")
            settings = db.execute(
                "SELECT cooldown_minutes FROM posture_settings WHERE id=1"
            ).fetchone()
            calibration = db.execute(
                "SELECT head_forward, neck_angle, torso_angle, torso_length, head_height FROM posture_calibration WHERE id=1"
            ).fetchone()
            slouch_calibration = db.execute(
                "SELECT head_forward, neck_angle, torso_angle, torso_length, head_height FROM posture_slouch_calibration WHERE id=1"
            ).fetchone()
        self._settings = {"cooldown_minutes": settings[0]}
        self._calibration = dict(zip(FEATURES, map(float, calibration))) if calibration else None
        self._slouch_calibration = dict(zip(FEATURES, map(float, slouch_calibration))) if slouch_calibration else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    @contextmanager
    def _db(self):
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def settings(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {"cooldown_minutes"}
        if not values or set(values) - allowed:
            raise ValueError("Only cooldown_minutes may be changed")
        current = self.settings()
        current.update(values)
        cooldown = current["cooldown_minutes"]
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or not 0 <= cooldown <= 120:
            raise ValueError("cooldown_minutes must be an integer from 0 to 120")
        with self._lock, self._db() as db:
            db.execute(
                "UPDATE posture_settings SET cooldown_minutes=? WHERE id=1",
                (current["cooldown_minutes"],),
            )
            self._settings = dict(current)
        return current

    def calibration(self) -> dict[str, float] | None:
        with self._lock:
            return dict(self._calibration) if self._calibration else None

    def slouch_calibration(self) -> dict[str, float] | None:
        with self._lock:
            return dict(self._slouch_calibration) if self._slouch_calibration else None

    def save_calibration(self, values: dict[str, float], sample_count: int, kind: str = "good") -> None:
        if kind not in {"good", "slouch"}:
            raise ValueError("Calibration kind must be good or slouch")
        table = "posture_calibration" if kind == "good" else "posture_slouch_calibration"
        with self._lock, self._db() as db:
            db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (1,?,?,?,?,?,?,?)",
                (datetime.now().astimezone().isoformat(), sample_count, *(values[name] for name in FEATURES)),
            )
            if kind == "good":
                self._calibration = dict(values)
                self._slouch_calibration = None
                db.execute("DELETE FROM posture_slouch_calibration")
            else:
                self._slouch_calibration = dict(values)

    def start_event(self, started_at: str, confidence: float, reason: str) -> int:
        with self._lock, self._db() as db:
            cursor = db.execute(
                "INSERT INTO posture_events(started_at, reminded, gemini_confidence, gemini_reason) VALUES (?,1,?,?)",
                (started_at, confidence, reason),
            )
            return int(cursor.lastrowid)

    def finish_event(self, event_id: int, ended_at: str, duration: float, scores: list[float], valid_rate: float) -> None:
        with self._lock, self._db() as db:
            db.execute(
                "UPDATE posture_events SET ended_at=?,duration_seconds=?,average_score=?,worst_score=?,valid_sample_rate=? WHERE id=?",
                (ended_at, round(duration, 1), round(statistics.fmean(scores), 4) if scores else 0,
                 round(max(scores), 4) if scores else 0, round(valid_rate, 4), event_id),
            )

    def events(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock, self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM posture_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def correct_event(self, event_id: int, correction: str) -> bool:
        if correction not in {"correct", "false_alarm", "not_bad_habit"}:
            raise ValueError("Unsupported correction")
        with self._lock, self._db() as db:
            cursor = db.execute("UPDATE posture_events SET correction=? WHERE id=?", (correction, event_id))
            changed = cursor.rowcount > 0
        if changed:
            self.refresh_habit_profile("posture")
        return changed

    def register_habit_occurrence(self, habit_key: str, occurred_at: str) -> dict[str, Any]:
        previous = self.habit_profile(habit_key)
        profile = self.refresh_habit_profile(habit_key, now=datetime.fromisoformat(occurred_at))
        if previous is None:
            alert_type = "first_added"
        elif profile["status"] == "established" and previous["status"] != "established":
            alert_type = "established"
        else:
            alert_type = "occurrence"
        return {"habit": habit_key, "alert_type": alert_type, **profile}

    def record_habit_occurrence(
        self, habit_key: str, occurred_at: str, details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a non-posture occurrence and update its rolling profile."""
        if habit_key == "posture":
            raise ValueError("Posture occurrences are recorded with posture_events")
        with self._lock, self._db() as db:
            cursor = db.execute(
                "INSERT INTO habit_events(habit_key,started_at,details_json) VALUES (?,?,?)",
                (habit_key, occurred_at, json.dumps(details or {}, separators=(",", ":"))),
            )
            event_id = int(cursor.lastrowid)
        return {"event_id": event_id, **self.register_habit_occurrence(habit_key, occurred_at)}

    def refresh_habit_profile(self, habit_key: str, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now().astimezone()
        cutoff = (now - timedelta(days=7)).isoformat()
        with self._lock, self._db() as db:
            if habit_key == "posture":
                rows = db.execute(
                    "SELECT started_at FROM posture_events WHERE started_at>=? AND correction IS NOT 'false_alarm' ORDER BY started_at",
                    (cutoff,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT started_at FROM habit_events WHERE habit_key=? AND started_at>=? "
                    "AND correction IS NOT 'false_alarm' ORDER BY started_at",
                    (habit_key, cutoff),
                ).fetchall()
            count = len(rows)
            days = len({row[0][:10] for row in rows})
            existing = db.execute("SELECT first_seen FROM habit_profiles WHERE habit_key=?", (habit_key,)).fetchone()
            first_seen = existing[0] if existing else (rows[0][0] if rows else now.isoformat())
            status = "established" if count >= 10 and days >= 3 else ("emerging" if count >= 3 else "possible")
            last_seen = rows[-1][0] if rows else now.isoformat()
            db.execute(
                "INSERT OR REPLACE INTO habit_profiles VALUES (?,?,?,?,?,?)",
                (habit_key, status, first_seen, last_seen, count, days),
            )
        return {"status": status, "first_seen": first_seen, "last_seen": last_seen,
                "rolling_occurrences": count, "rolling_days": days}

    def habit_profile(self, habit_key: str) -> dict[str, Any] | None:
        with self._lock, self._db() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM habit_profiles WHERE habit_key=?", (habit_key,)).fetchone()
        return dict(row) if row else None

    def habit_profiles(self) -> list[dict[str, Any]]:
        """Return every tracked habit, newest activity first."""
        with self._lock, self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM habit_profiles ORDER BY last_seen DESC, habit_key"
            ).fetchall()
        return [dict(row) for row in rows]

    def monitor_state(self, monitor_key: str) -> dict[str, Any]:
        with self._lock, self._db() as db:
            row = db.execute(
                "SELECT state_json FROM monitor_state WHERE monitor_key=?", (monitor_key,)
            ).fetchone()
        if not row:
            return {}
        try:
            value = json.loads(row[0])
            return value if isinstance(value, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    def save_monitor_state(self, monitor_key: str, state: dict[str, Any]) -> None:
        payload = json.dumps(state, separators=(",", ":"), sort_keys=True)
        with self._lock, self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO monitor_state(monitor_key,state_json) VALUES (?,?)",
                (monitor_key, payload),
            )

    def create_notification(self, habit_key: str, event_id: int, payload: dict[str, Any], voice_pending: bool = True) -> dict[str, Any]:
        created = datetime.now().astimezone().isoformat()
        with self._lock, self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO habit_notifications(habit_key,event_id,created_at,payload_json,voice_delivered) VALUES (?,?,?,?,?)",
                (habit_key, event_id, created, json.dumps(payload, separators=(",", ":")), 0 if voice_pending else 1),
            )
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM habit_notifications WHERE habit_key=? AND event_id=?", (habit_key, event_id)).fetchone()
        return {**dict(row), "payload": json.loads(row["payload_json"])}

    def notifications(self, pending_visual_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM habit_notifications"
        if pending_visual_only: query += " WHERE visual_acknowledged=0"
        query += " ORDER BY id DESC"
        with self._lock, self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(query).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def pending_voice_notifications(self, habit_key: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM habit_notifications WHERE voice_delivered=0", []
        if habit_key: query, args = query + " AND habit_key=?", [habit_key]
        with self._lock, self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(query + " ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def mark_notification_voice(self, notification_id: int) -> bool:
        with self._lock, self._db() as db:
            return db.execute("UPDATE habit_notifications SET voice_delivered=1 WHERE id=?", (notification_id,)).rowcount > 0

    def acknowledge_notification(self, notification_id: int) -> bool:
        with self._lock, self._db() as db:
            return db.execute("UPDATE habit_notifications SET visual_acknowledged=1 WHERE id=?", (notification_id,)).rowcount > 0

    def today_seconds(self, now: datetime | None = None) -> float:
        day = (now or datetime.now().astimezone()).date().isoformat()
        with self._lock, self._db() as db:
            row = db.execute(
                "SELECT COALESCE(SUM(duration_seconds),0) FROM posture_events WHERE started_at LIKE ? AND correction IS NOT 'false_alarm'",
                (f"{day}%",),
            ).fetchone()
        return round(float(row[0]), 1)

    def clear_habit_history(self) -> None:
        with self._lock, self._db() as db:
            db.execute("DELETE FROM posture_events")
            db.execute("DELETE FROM habit_events")
            db.execute("DELETE FROM habit_profiles")
            db.execute("DELETE FROM habit_notifications")
            # Settings and numerical calibrations survive. Episode candidates and
            # latches are history and are intentionally cleared.
            db.execute("DELETE FROM monitor_state WHERE monitor_key NOT LIKE '%_settings' AND monitor_key NOT IN ('desk_clutter_calibration','office_lights_settings')")
            db.execute("DELETE FROM sqlite_sequence WHERE name='posture_events'")
            db.execute("DELETE FROM sqlite_sequence WHERE name='habit_events'")

    def system_prompt(self, default: str) -> str:
        with self._lock, self._db() as db:
            row = db.execute("SELECT system_prompt FROM ada_settings WHERE id=1").fetchone()
        return str(row[0]) if row and row[0] else default

    def update_system_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not 100 <= len(prompt) <= 12000:
            raise ValueError("System prompt must contain 100 to 12000 characters")
        with self._lock, self._db() as db:
            db.execute("UPDATE ada_settings SET system_prompt=? WHERE id=1", (prompt,))
        return prompt


def _center(*points: dict[str, float]) -> tuple[float, float]:
    return (statistics.fmean(p["x"] for p in points), statistics.fmean(p["y"] for p in points))


def _confidence_center(*points: dict[str, float]) -> tuple[float, float]:
    """Favor the stable, near-side ear instead of jumping to a two-ear midpoint."""
    weights = [max(0.01, point["score"]) ** 4 for point in points]
    total = sum(weights)
    return (
        sum(point["x"] * weight for point, weight in zip(points, weights)) / total,
        sum(point["y"] * weight for point, weight in zip(points, weights)) / total,
    )


def posture_features(result: dict[str, Any], confidence: float = 0.35) -> dict[str, float] | None:
    points = {point["name"]: point for point in result.get("keypoints", [])}
    shoulders = [points.get("left_shoulder"), points.get("right_shoulder")]
    hips = [points.get("left_hip"), points.get("right_hip")]
    ears = [points.get("left_ear"), points.get("right_ear")]
    if any(point is None or point["score"] < confidence for point in shoulders):
        return None
    visible_ears = [point for point in ears if point is not None and point["score"] >= confidence]
    if not visible_ears:
        return None
    shoulder = _center(*shoulders)
    ear = _confidence_center(*visible_ears)
    shoulder_width = math.dist((shoulders[0]["x"], shoulders[0]["y"]), (shoulders[1]["x"], shoulders[1]["y"]))
    if shoulder_width < 0.025:
        return None
    neck_dx, neck_dy = ear[0] - shoulder[0], shoulder[1] - ear[1]
    features = {
        "head_forward": abs(neck_dx) / shoulder_width,
        "neck_angle": math.degrees(math.atan2(abs(neck_dx), max(0.001, neck_dy))),
        # Zero marks optional torso features as unavailable. Head/shoulder
        # features remain sufficient for desk-mounted, upper-body framing.
        "torso_angle": 0.0,
        "torso_length": 0.0,
        "head_height": neck_dy / shoulder_width,
    }
    if all(point is not None and point["score"] >= confidence for point in hips):
        hip = _center(*hips)
        torso_dx, torso_dy = shoulder[0] - hip[0], hip[1] - shoulder[1]
        features["torso_angle"] = math.degrees(math.atan2(abs(torso_dx), max(0.001, torso_dy)))
        features["torso_length"] = math.dist(shoulder, hip) / shoulder_width
    return features


def posture_score(features: dict[str, float], baseline: dict[str, float]) -> float:
    components = {
        # Dead zones absorb normal MoveNet landmark jitter around a freshly
        # calibrated pose before scaling the remaining movement to 0..1.
        "head_forward": (0.35, max(0.0, features["head_forward"] - baseline["head_forward"] - 0.12) / 0.40),
        "neck_angle": (0.25, max(0.0, features["neck_angle"] - baseline["neck_angle"] - 4.0) / 30.0),
        "head_height": (0.10, max(0.0, baseline["head_height"] - features["head_height"] - 0.08) / max(0.08, abs(baseline["head_height"]) * 0.35)),
    }
    if baseline["torso_length"] > 0 and features["torso_length"] > 0:
        components["torso_angle"] = (0.20, abs(features["torso_angle"] - baseline["torso_angle"]) / 20.0)
        components["torso_length"] = (0.10, max(0.0, baseline["torso_length"] - features["torso_length"]) / max(0.05, baseline["torso_length"] * 0.25))
    weight_total = sum(weight for weight, _ in components.values())
    return round(sum(weight * min(1.0, value) for weight, value in components.values()) / weight_total, 4)


def benchmark_posture_score(
    features: dict[str, float], good: dict[str, float], slouch: dict[str, float]
) -> float:
    """Measure progress from the user's upright benchmark toward their slouch."""
    weights = {"head_forward": 0.35, "neck_angle": 0.25, "torso_angle": 0.20,
               "torso_length": 0.10, "head_height": 0.10}
    minimum_change = {"head_forward": 0.08, "neck_angle": 3.0, "torso_angle": 3.0,
                      "torso_length": 0.08, "head_height": 0.06}
    components = []
    for name, weight in weights.items():
        if name not in features or name not in good or name not in slouch:
            continue
        if name in {"torso_angle", "torso_length"} and (
                good["torso_length"] <= 0 or slouch["torso_length"] <= 0 or features["torso_length"] <= 0):
            continue
        span = slouch[name] - good[name]
        if abs(span) < minimum_change[name]:
            continue
        progress = (features[name] - good[name]) / span
        components.append((weight, max(0.0, min(1.0, progress))))
    if not components:
        return posture_score(features, good)
    weight_total = sum(weight for weight, _ in components)
    return round(sum(weight * value for weight, value in components) / weight_total, 4)


class PostureMonitor:
    CALIBRATION_SECONDS = 30.0
    MIN_CALIBRATION_SAMPLES = 40

    def __init__(self, store: PostureStore) -> None:
        self.store = store
        self.state = "uncalibrated" if store.calibration() is None else "good"
        self.score: float | None = None
        self._scores: deque[tuple[float, float]] = deque()
        self._candidate_since: float | None = None
        self._candidate_samples = self._candidate_valid = 0
        self._recovery_since: float | None = None
        self._last_valid: float | None = None
        self._event_id: int | None = None
        self._event_started: float | None = None
        self._event_scores: list[float] = []
        self._event_samples = self._event_valid = 0
        self._cooldown_until = 0.0
        self._calibration_started: float | None = None
        self._calibration_kind: str | None = None
        self._calibration_samples: list[dict[str, float]] = []
        self.calibration_error: str | None = None
        self.last_event_id: int | None = None
        self._pending_habit_alert: dict[str, Any] | None = None
        self._verification_requested = False
        self._verification_retry_at = 0.0
        self._gemini_confirmed: bool | None = None
        self.gemini_status = "idle"
        self.gemini_confidence: float | None = None
        self.gemini_reason = ""

    def start_calibration(self, now: float | None = None, kind: str = "good") -> None:
        if kind not in {"good", "slouch"}:
            raise ValueError("Calibration kind must be good or slouch")
        if kind == "slouch" and self.store.calibration() is None:
            raise ValueError("Calibrate good posture first")
        started = now if now is not None else time.monotonic()
        if self._event_id is not None:
            self._close_event(started, None)
        self._candidate_since = None
        self._candidate_samples = self._candidate_valid = 0
        self._scores.clear()
        self._reset_verification()
        self._calibration_started = started
        self._calibration_kind = kind
        self._calibration_samples = []
        self.calibration_error = None
        self.state = "calibrating"

    def clear_habit_history(self) -> None:
        self.store.clear_habit_history()
        self.state = "good" if self.store.calibration() is not None else "uncalibrated"
        self.score = None
        self._scores.clear()
        self._candidate_since = None
        self._candidate_samples = self._candidate_valid = 0
        self._recovery_since = None
        self._last_valid = None
        self._event_id = self._event_started = None
        self._event_scores = []
        self._event_samples = self._event_valid = 0
        self._cooldown_until = 0.0
        self._calibration_started = None
        self._calibration_kind = None
        self._calibration_samples = []
        self.calibration_error = None
        self.last_event_id = None
        self._pending_habit_alert = None
        self._reset_verification()

    def process(self, result: dict[str, Any], now: float | None = None, wall_time: datetime | None = None) -> None:
        now = now if now is not None else time.monotonic()
        features = posture_features(result)
        if self._calibration_started is not None:
            if features is not None:
                self._calibration_samples.append(features)
            if now - self._calibration_started >= self.CALIBRATION_SECONDS:
                self._finish_calibration()
            return
        baseline = self.store.calibration()
        if baseline is None:
            self.state, self.score = "uncalibrated", None
            return
        if features is None:
            if self._candidate_since is not None:
                self._candidate_samples += 1
            self._event_samples += int(self._event_id is not None)
            if self._last_valid is not None and now - self._last_valid >= 30 and self._event_id is not None:
                self._close_event(now, wall_time)
            return
        self._last_valid = now
        slouch_baseline = self.store.slouch_calibration()
        raw_score = (
            benchmark_posture_score(features, baseline, slouch_baseline)
            if slouch_baseline else posture_score(features, baseline)
        )
        self._scores.append((now, raw_score))
        while self._scores and now - self._scores[0][0] > 5:
            self._scores.popleft()
        self.score = round(statistics.fmean(value for _, value in self._scores), 4)
        suspect_threshold, slouch_threshold = POSTURE_THRESHOLDS
        if self._event_id is not None:
            self._event_samples += 1
            self._event_valid += 1
            self._event_scores.append(self.score)
            if self.score < suspect_threshold:
                self._recovery_since = self._recovery_since or now
                if now - self._recovery_since >= 10:
                    self._close_event(now, wall_time)
            else:
                self._recovery_since = None
            return
        if now < self._cooldown_until:
            self.state = "cooldown"
            return
        if self.score >= suspect_threshold:
            if self._candidate_since is None:
                self._candidate_since = now
                self._candidate_samples = self._candidate_valid = 0
                self._reset_verification()
            self._candidate_samples += 1
            self._candidate_valid += 1
            elapsed = now - self._candidate_since
            self.state = "suspected" if elapsed >= 10 else "good"
            if elapsed >= 10 and self._gemini_confirmed is None and now >= self._verification_retry_at:
                self._verification_requested = True
                self._verification_retry_at = now + 30
            valid_rate = self._candidate_valid / max(1, self._candidate_samples)
            if (elapsed >= 20 and valid_rate >= 0.70 and self.score >= slouch_threshold
                    and self._gemini_confirmed is True):
                self._open_event(now, wall_time)
        else:
            self._candidate_since = None
            self._candidate_samples = self._candidate_valid = 0
            self._reset_verification()
            self.state = "good"

    def take_verification_request(self) -> bool:
        if not self._verification_requested:
            return False
        self._verification_requested = False
        self.gemini_status = "verifying"
        return True

    def apply_verification(self, verdict: dict[str, object] | None, error: str | None = None) -> None:
        if error is not None:
            self.gemini_status = "unavailable"
            self.gemini_reason = error[:300]
            return
        assert verdict is not None
        self._gemini_confirmed = bool(verdict["slouching"])
        self.gemini_confidence = float(verdict["confidence"])
        self.gemini_reason = str(verdict["reason"])
        self.gemini_status = "confirmed" if self._gemini_confirmed else "not_confirmed"

    def _reset_verification(self) -> None:
        self._verification_requested = False
        self._verification_retry_at = 0.0
        self._gemini_confirmed = None
        self.gemini_status = "idle"
        self.gemini_confidence = None
        self.gemini_reason = ""

    def _finish_calibration(self) -> None:
        samples = self._calibration_samples
        kind = self._calibration_kind or "good"
        self._calibration_started = None
        self._calibration_kind = None
        if len(samples) < self.MIN_CALIBRATION_SAMPLES:
            self.calibration_error = "Not enough clear samples. Keep both shoulders and at least one ear visible."
            self.state = "uncalibrated" if self.store.calibration() is None else "good"
            return
        baseline = {name: statistics.median(sample[name] for sample in samples) for name in FEATURES}
        # Large torso-angle spread indicates that the user or camera moved.
        torso_angles = [sample["torso_angle"] for sample in samples if sample["torso_length"] > 0]
        if ((len(torso_angles) >= self.MIN_CALIBRATION_SAMPLES * 0.7 and statistics.pstdev(torso_angles) > 8)
                or statistics.pstdev(sample["head_forward"] for sample in samples) > 0.35):
            self.calibration_error = "Too much movement during calibration. Sit still and try again."
            self.state = "uncalibrated" if self.store.calibration() is None else "good"
            return
        self.store.save_calibration(baseline, len(samples), kind=kind)
        self.state = "good"
        self.calibration_error = None

    def _open_event(self, now: float, wall_time: datetime | None) -> None:
        stamp = (wall_time or datetime.now().astimezone()).isoformat()
        self._event_id = self.store.start_event(
            stamp, self.gemini_confidence or 0.0, self.gemini_reason,
        )
        self.last_event_id = self._event_id
        self._pending_habit_alert = self.store.register_habit_occurrence("posture", stamp)
        self._event_started = now
        self._event_scores = [self.score or 0]
        self._event_samples = self._event_valid = 1
        self._candidate_since = None
        self._candidate_samples = self._candidate_valid = 0
        self.state = "slouching"

    def take_habit_alert(self) -> dict[str, Any] | None:
        alert = self._pending_habit_alert
        self._pending_habit_alert = None
        return alert

    def _close_event(self, now: float, wall_time: datetime | None) -> None:
        assert self._event_id is not None and self._event_started is not None
        valid_rate = self._event_valid / max(1, self._event_samples)
        self.store.finish_event(
            self._event_id, (wall_time or datetime.now().astimezone()).isoformat(),
            now - self._event_started, self._event_scores, valid_rate,
        )
        self._event_id = self._event_started = None
        self._event_scores = []
        self._cooldown_until = now + self.store.settings()["cooldown_minutes"] * 60
        self._recovery_since = None
        self.state = "cooldown"

    def status(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.monotonic()
        progress = 0.0
        if self._calibration_started is not None:
            progress = min(1.0, (now - self._calibration_started) / self.CALIBRATION_SECONDS)
        return {
            **self.store.settings(), "state": self.state, "score": self.score,
            "calibrated": self.store.calibration() is not None,
            "slouch_calibrated": self.store.slouch_calibration() is not None,
            "calibration_kind": self._calibration_kind,
            "today_seconds": self.store.today_seconds(),
            "calibration_progress": round(progress, 3), "calibration_error": self.calibration_error,
            "last_event_id": self.last_event_id,
            "gemini_status": self.gemini_status,
            "gemini_confidence": self.gemini_confidence,
            "gemini_reason": self.gemini_reason,
        }


class PoseService:
    """Run one pose inference loop and share its latest result with consumers."""

    def __init__(self, camera: Any, estimator: Any, monitor: PostureMonitor, verifier: Any = None, notifier: Any = None, fps: float = 2) -> None:
        self.camera, self.estimator, self.monitor = camera, estimator, monitor
        self.verifier, self.notifier, self.fps = verifier, notifier, fps
        self.latest_result: dict[str, Any] | None = None
        self.latest_frame: bytes | None = None
        self.latest_result_at: float | None = None
        self._generation = 0
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._verification_task: asyncio.Task[None] | None = None
        self._notification_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            async for frame in self.camera.frames():
                started = time.monotonic()
                result = await asyncio.to_thread(self.estimator.infer, frame)
                self.monitor.process(result)
                if (self.verifier is not None and self.monitor.take_verification_request()
                        and (self._verification_task is None or self._verification_task.done())):
                    self._verification_task = asyncio.create_task(self._verify(frame))
                habit_alert = self.monitor.take_habit_alert()
                if habit_alert and self.notifier is not None:
                    task = asyncio.create_task(self._notify_habit(frame, habit_alert))
                    self._notification_tasks.add(task)
                    task.add_done_callback(self._notification_tasks.discard)
                self.latest_result, self.latest_frame = result, frame
                self.latest_result_at = time.monotonic()
                async with self._condition:
                    self._generation += 1
                    self._condition.notify_all()
                await asyncio.sleep(max(0, 1 / self.fps - (time.monotonic() - started)))
        finally:
            async with self._condition:
                self._condition.notify_all()

    async def _verify(self, frame: bytes) -> None:
        try:
            self.monitor.apply_verification(await self.verifier.verify(frame))
        except Exception as exc:
            logger.warning("Gemini posture verification failed: %s", exc)
            self.monitor.apply_verification(None, error=str(exc))

    async def _notify_habit(self, frame: bytes, alert: dict[str, Any]) -> None:
        kind = alert["alert_type"]
        if kind == "first_added":
            message = "ADA habit alert: confirmed posture slouching was observed for the first time. Add posture as a possible bad habit. Give one concise, dry observation and one practical posture correction."
        elif kind == "established":
            message = f"ADA habit alert: posture is now an established bad habit after {alert['rolling_occurrences']} confirmed occurrences across {alert['rolling_days']} days in the last week. Deliver a composed, mildly sarcastic observation that the pattern is now established, then suggest one small correction."
        else:
            message = f"ADA habit alert: another confirmed posture-slouch occurrence was observed. This is occurrence {alert['rolling_occurrences']} across {alert['rolling_days']} days in the rolling week. Briefly acknowledge the repeat with understated dry wit and give one useful correction."
        try:
            await self.notifier.send_habit_alert(frame, message)
        except Exception as exc:
            logger.warning("Could not send habit alert to Gemini Live: %s", exc)

    async def results(self):
        await self.start()
        generation = self._generation
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._generation != generation or (self._task and self._task.done()))
                if self._task and self._task.done():
                    return
                generation = self._generation
            if self.latest_result is not None and self.latest_frame is not None:
                yield self.latest_result, self.latest_frame

    def office_occupied(self) -> bool | None:
        """Return recent local person presence, or None when vision is stale."""
        if (self.latest_result is None or self.latest_result_at is None
                or time.monotonic() - self.latest_result_at > 10):
            return None
        core = {"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder"}
        visible = [
            point for point in self.latest_result.get("keypoints", [])
            if point.get("name") in core and float(point.get("score", 0)) >= 0.25
        ]
        shoulders = {point.get("name") for point in visible} & {"left_shoulder", "right_shoulder"}
        return len(visible) >= 3 and bool(shoulders)

    async def stop(self) -> None:
        for task in tuple(self._notification_tasks):
            task.cancel()
        for task in tuple(self._notification_tasks):
            with suppress(asyncio.CancelledError):
                await task
        if self._verification_task:
            self._verification_task.cancel()
            try:
                await self._verification_task
            except asyncio.CancelledError:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

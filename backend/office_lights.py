"""Office-light waste monitor combining Home Assistant and local presence."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from datetime import datetime, time as wall_time, timedelta
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from .home_assistant import HomeAssistantClient, HomeAssistantSnapshot

logger = logging.getLogger("voice.office_lights")

HABIT_KEY = "office_lights_left_on"
MONITOR_KEY = "office_lights"
SETTINGS_KEY = "office_lights_settings"
INVALID_PERSON_STATES = {"", "unknown", "unavailable"}


class OfficeLightMonitor:
    def __init__(
        self,
        home_assistant: HomeAssistantClient,
        store: Any,
        presence_getter: Callable[[], bool | None],
        notifier: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        grace_seconds: int | None = None,
        poll_seconds: int | None = None,
    ) -> None:
        self.home_assistant = home_assistant
        self.store = store
        self.presence_getter = presence_getter
        self.notifier = notifier
        self.grace_seconds = grace_seconds if grace_seconds is not None else int(
            os.environ.get("OFFICE_EMPTY_GRACE_SECONDS", "300")
        )
        self.poll_seconds = poll_seconds if poll_seconds is not None else int(
            os.environ.get("HOME_ASSISTANT_POLL_SECONDS", "15")
        )
        self.reset_hour = int(os.environ.get("OFFICE_LIGHTS_RESET_HOUR", "18"))
        saved_settings = store.monitor_state(SETTINGS_KEY)
        self.grace_seconds = int(saved_settings.get("grace_seconds", self.grace_seconds))
        self.poll_seconds = int(saved_settings.get("poll_seconds", self.poll_seconds))
        self.reset_hour = int(saved_settings.get("reset_hour", self.reset_hour))
        self.timezone = ZoneInfo(os.environ.get("ADA_TIMEZONE", "America/New_York"))
        self._task: asyncio.Task[None] | None = None
        self.status = "disabled" if not home_assistant.configured else "starting"
        self.last_error = "" if home_assistant.configured else "HOME_ASSISTANT_TOKEN is not set"
        self.latest: dict[str, Any] = {}

    @staticmethod
    def _parse_stamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _next_day_reset(self, now: datetime) -> datetime:
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(tomorrow, wall_time(self.reset_hour, 0), tzinfo=now.tzinfo)

    def settings(self) -> dict[str, int]:
        return {
            "grace_minutes": self.grace_seconds // 60,
            "poll_seconds": self.poll_seconds,
            "reset_hour": self.reset_hour,
        }

    def update_settings(self, values: dict[str, Any]) -> dict[str, int]:
        allowed = {"grace_minutes", "poll_seconds", "reset_hour"}
        if not values or set(values) - allowed:
            raise ValueError("Only grace_minutes, poll_seconds, and reset_hour may be changed")
        current = self.settings()
        current.update(values)
        for key in allowed:
            value = current[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if not 1 <= current["grace_minutes"] <= 120:
            raise ValueError("grace_minutes must be from 1 to 120")
        if not 5 <= current["poll_seconds"] <= 300:
            raise ValueError("poll_seconds must be from 5 to 300")
        if not 0 <= current["reset_hour"] <= 23:
            raise ValueError("reset_hour must be from 0 to 23")
        self.grace_seconds = current["grace_minutes"] * 60
        self.poll_seconds = current["poll_seconds"]
        self.reset_hour = current["reset_hour"]
        self.store.save_monitor_state(SETTINGS_KEY, {
            "grace_seconds": self.grace_seconds,
            "poll_seconds": self.poll_seconds,
            "reset_hour": self.reset_hour,
        })
        return self.settings()

    def evaluate(
        self,
        snapshot: HomeAssistantSnapshot,
        office_occupied: bool | None,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Advance the persisted state machine and return a newly logged alert."""
        now = now or datetime.now(self.timezone)
        state = self.store.monitor_state(MONITOR_KEY)
        previous_state = dict(state)
        candidate_since = self._parse_stamp(state.get("candidate_since"))
        reset_at = self._parse_stamp(state.get("reset_at"))
        latched = bool(state.get("latched", False))
        lights_on = list(snapshot.lights_on)

        if not lights_on:
            state = {"latched": False, "candidate_since": None, "reset_at": None}
            condition = False
            absence_reason = "lights_off"
        else:
            if latched and reset_at is not None and now >= reset_at:
                latched = False
                candidate_since = None
                reset_at = None

            person_state = snapshot.person_state.strip().lower()
            if person_state in INVALID_PERSON_STATES:
                condition = False
                absence_reason = "person_state_unavailable"
            elif person_state != "home":
                condition = True
                absence_reason = "away_from_home"
            elif office_occupied is None:
                condition = False
                absence_reason = "office_presence_unavailable"
            else:
                condition = not office_occupied
                absence_reason = "office_empty" if condition else "office_occupied"

            state = {
                "latched": latched,
                "candidate_since": candidate_since.isoformat() if candidate_since else None,
                "reset_at": reset_at.isoformat() if reset_at else None,
            }

        alert = None
        if lights_on and not state["latched"]:
            if condition:
                if candidate_since is None:
                    candidate_since = now
                    state["candidate_since"] = now.isoformat()
                if (now - candidate_since).total_seconds() >= self.grace_seconds:
                    details = {
                        "lights_on": lights_on,
                        "person_state": snapshot.person_state,
                        "absence_reason": absence_reason,
                        "confirmed_after_seconds": self.grace_seconds,
                    }
                    alert = self.store.record_habit_occurrence(HABIT_KEY, now.isoformat(), details)
                    reset_at = self._next_day_reset(now)
                    state = {
                        "latched": True,
                        "candidate_since": None,
                        "reset_at": reset_at.isoformat(),
                        "last_event_id": alert["event_id"],
                    }
                    alert.update(details)
            else:
                state["candidate_since"] = None

        if state != previous_state:
            self.store.save_monitor_state(MONITOR_KEY, state)
        self.latest = {
            "status": self.status,
            "person_state": snapshot.person_state,
            "office_occupied": office_occupied,
            "lights_on": lights_on,
            "absence_reason": absence_reason,
            **state,
        }
        return alert

    async def start(self) -> None:
        if not self.home_assistant.configured:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                snapshot = await self.home_assistant.snapshot()
                self.status = "monitoring"
                self.last_error = ""
                alert = self.evaluate(snapshot, self.presence_getter())
                if alert is not None:
                    logger.info(
                        "office lights habit recorded reason=%s lights=%s reset_at=%s",
                        alert["absence_reason"], alert["lights_on"], self.latest.get("reset_at"),
                    )
                    if self.notifier is not None:
                        try:
                            await self.notifier(alert)
                            logger.info("office lights spoken habit alert sent event_id=%s", alert["event_id"])
                        except Exception as exc:
                            logger.warning("could not send office lights spoken alert: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status = "unavailable"
                self.last_error = str(exc)
                logger.warning("office light monitor check failed: %s", exc)
            await asyncio.sleep(max(5, self.poll_seconds))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        await self.home_assistant.close()

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status, "error": self.last_error, "settings": self.settings(), **self.latest}

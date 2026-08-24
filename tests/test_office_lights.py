import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.home_assistant import HomeAssistantSnapshot
from backend.office_lights import HABIT_KEY, OfficeLightMonitor
from backend.posture import PostureStore


class FakeHomeAssistant:
    configured = True

    async def close(self):
        pass


def snapshot(*, person="not_home", lights=("light.left_office_light",)):
    return HomeAssistantSnapshot(
        person_state=person,
        lights_on=tuple(lights),
        light_states={entity: "on" for entity in lights},
    )


class OfficeLightMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PostureStore(Path(self.temp.name) / "habits.db")
        self.monitor = OfficeLightMonitor(
            FakeHomeAssistant(), self.store, lambda: False,
            grace_seconds=300, poll_seconds=5,
        )
        self.start = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_away_and_lights_on_requires_five_minute_recheck(self):
        self.assertIsNone(self.monitor.evaluate(snapshot(), False, self.start))
        self.assertIsNone(self.monitor.evaluate(snapshot(), False, self.start + timedelta(seconds=299)))
        alert = self.monitor.evaluate(snapshot(), False, self.start + timedelta(seconds=300))
        self.assertEqual(alert["habit"], HABIT_KEY)
        self.assertEqual(alert["absence_reason"], "away_from_home")
        self.assertEqual(alert["lights_on"], ["light.left_office_light"])
        self.assertEqual(self.store.habit_profile(HABIT_KEY)["rolling_occurrences"], 1)

    def test_latch_prevents_duplicates_until_next_day_six_pm(self):
        self.monitor.evaluate(snapshot(), False, self.start)
        first = self.monitor.evaluate(snapshot(), False, self.start + timedelta(minutes=5))
        self.assertIsNotNone(first)
        next_day = self.start + timedelta(days=1)
        self.assertIsNone(self.monitor.evaluate(snapshot(), False, next_day.replace(hour=17, minute=59)))
        self.assertIsNone(self.monitor.evaluate(snapshot(), False, next_day.replace(hour=18, minute=0)))
        second = self.monitor.evaluate(snapshot(), False, next_day.replace(hour=18, minute=5))
        self.assertIsNotNone(second)
        self.assertEqual(self.store.habit_profile(HABIT_KEY)["rolling_occurrences"], 2)

    def test_latch_survives_monitor_restart(self):
        self.monitor.evaluate(snapshot(), False, self.start)
        self.monitor.evaluate(snapshot(), False, self.start + timedelta(minutes=5))
        restarted = OfficeLightMonitor(
            FakeHomeAssistant(), self.store, lambda: False,
            grace_seconds=300, poll_seconds=5,
        )
        self.assertIsNone(restarted.evaluate(snapshot(), False, self.start + timedelta(hours=3)))
        self.assertEqual(self.store.habit_profile(HABIT_KEY)["rolling_occurrences"], 1)

    def test_lights_off_resets_episode(self):
        self.monitor.evaluate(snapshot(), False, self.start)
        self.monitor.evaluate(snapshot(), False, self.start + timedelta(minutes=5))
        self.monitor.evaluate(snapshot(lights=()), False, self.start + timedelta(minutes=6))
        self.monitor.evaluate(snapshot(), False, self.start + timedelta(minutes=7))
        second = self.monitor.evaluate(snapshot(), False, self.start + timedelta(minutes=12))
        self.assertIsNotNone(second)
        self.assertEqual(self.store.habit_profile(HABIT_KEY)["rolling_occurrences"], 2)

    def test_home_uses_office_presence_and_unknown_fails_safe(self):
        self.monitor.evaluate(snapshot(person="home"), True, self.start)
        self.assertIsNone(self.monitor.evaluate(snapshot(person="home"), True, self.start + timedelta(minutes=10)))
        self.monitor.evaluate(snapshot(person="home"), False, self.start + timedelta(minutes=11))
        alert = self.monitor.evaluate(snapshot(person="home"), False, self.start + timedelta(minutes=16))
        self.assertEqual(alert["absence_reason"], "office_empty")

        self.monitor.evaluate(snapshot(lights=()), False, self.start + timedelta(minutes=17))
        self.monitor.evaluate(snapshot(person="unknown"), False, self.start + timedelta(minutes=18))
        self.assertIsNone(self.monitor.evaluate(snapshot(person="unknown"), False, self.start + timedelta(minutes=30)))

    def test_timing_settings_validate_and_survive_restart(self):
        settings = self.monitor.update_settings({
            "grace_minutes": 12,
            "poll_seconds": 30,
            "reset_hour": 20,
        })
        self.assertEqual(settings, {"grace_minutes": 12, "poll_seconds": 30, "reset_hour": 20})
        restarted = OfficeLightMonitor(FakeHomeAssistant(), self.store, lambda: False)
        self.assertEqual(restarted.settings(), settings)
        self.store.clear_habit_history()
        restarted_again = OfficeLightMonitor(FakeHomeAssistant(), self.store, lambda: False)
        self.assertEqual(restarted_again.settings(), settings)

    def test_custom_reset_hour_applies_to_new_latch(self):
        self.monitor.update_settings({"reset_hour": 21})
        self.monitor.evaluate(snapshot(), False, self.start)
        self.monitor.evaluate(snapshot(), False, self.start + timedelta(minutes=5))
        state = self.store.monitor_state("office_lights")
        self.assertEqual(datetime.fromisoformat(state["reset_at"]).hour, 21)


if __name__ == "__main__":
    unittest.main()

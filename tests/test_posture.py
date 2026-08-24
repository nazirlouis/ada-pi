import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.posture import (
    POSTURE_THRESHOLDS, PostureMonitor, PostureStore, benchmark_posture_score,
    posture_features, posture_score,
)


def pose(*, slouch: bool = False):
    coordinates = {
        "left_ear": (0.25 if slouch else 0.43, 0.45 if slouch else 0.30),
        "right_ear": (0.35 if slouch else 0.53, 0.45 if slouch else 0.30),
        "left_shoulder": (0.45, 0.50), "right_shoulder": (0.55, 0.50),
        "left_hip": (0.46, 0.80), "right_hip": (0.56, 0.80),
    }
    names = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    )
    return {"keypoints": [
        {"name": name, "x": coordinates.get(name, (0.5, 0.5))[0],
         "y": coordinates.get(name, (0.5, 0.5))[1],
         "score": 0.99 if name in coordinates else 0.1}
        for name in names
    ]}


class PostureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PostureStore(Path(self.temp.name) / "habits.db")

    def tearDown(self):
        self.temp.cleanup()

    def test_features_are_scale_normalized_and_slouch_scores_higher(self):
        baseline = posture_features(pose())
        slouch = posture_features(pose(slouch=True))
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(slouch)
        self.assertLess(posture_score(baseline, baseline), 0.01)
        self.assertGreater(posture_score(slouch, baseline), 0.60)

    def test_user_tuned_score_thresholds(self):
        self.assertEqual(POSTURE_THRESHOLDS, (0.40, 0.50))

    def test_monitoring_is_always_on_and_cooldown_is_configurable(self):
        settings = self.store.update_settings({"cooldown_minutes": 12})
        self.assertEqual(settings["cooldown_minutes"], 12)
        with self.assertRaises(ValueError):
            self.store.update_settings({"enabled": False})
        with self.assertRaises(ValueError):
            self.store.update_settings({"start_hour": 9})
        with self.assertRaises(ValueError):
            self.store.update_settings({"sensitivity": "high"})

    def test_upper_body_only_pose_is_valid_without_hips(self):
        upright = pose()
        slouched = pose(slouch=True)
        for result in (upright, slouched):
            for point in result["keypoints"]:
                if point["name"] in {"left_hip", "right_hip"}:
                    point["score"] = 0.1
        baseline = posture_features(upright)
        current = posture_features(slouched)
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline["torso_length"], 0)
        self.assertGreater(posture_score(current, baseline), 0.60)

    def test_small_upper_body_landmark_jitter_stays_below_suspected(self):
        baseline = posture_features(pose())
        jittered = pose()
        for point in jittered["keypoints"]:
            if point["name"] in {"left_ear", "right_ear"}:
                point["x"] -= 0.008
                point["y"] += 0.006
        self.assertLess(posture_score(posture_features(jittered), baseline), 0.20)

    def test_personal_slouch_benchmark_maps_good_to_zero_and_slouch_to_one(self):
        good = posture_features(pose())
        slouch = posture_features(pose(slouch=True))
        self.assertLess(benchmark_posture_score(good, good, slouch), 0.01)
        self.assertGreater(benchmark_posture_score(slouch, good, slouch), 0.99)

    def test_recalibrating_good_posture_invalidates_old_slouch_benchmark(self):
        good = posture_features(pose())
        slouch = posture_features(pose(slouch=True))
        self.store.save_calibration(good, 60)
        self.store.save_calibration(slouch, 60, kind="slouch")
        self.assertIsNotNone(self.store.slouch_calibration())
        self.store.save_calibration(good, 60, kind="good")
        self.assertIsNone(self.store.slouch_calibration())

    def test_clear_history_preserves_calibrations_and_settings(self):
        good = posture_features(pose())
        self.store.save_calibration(good, 60)
        self.store.save_calibration(posture_features(pose(slouch=True)), 60, kind="slouch")
        self.store.update_settings({"cooldown_minutes": 22})
        event_id = self.store.start_event("2026-08-23T12:00:00-04:00", 0.9, "confirmed")
        self.store.finish_event(event_id, "2026-08-23T12:01:00-04:00", 60, [0.8], 1)
        monitor = PostureMonitor(self.store)
        monitor.clear_habit_history()
        self.assertEqual(monitor.state, "good")
        self.assertEqual(self.store.calibration(), good)
        self.assertIsNotNone(self.store.slouch_calibration())
        self.assertEqual(self.store.events(), [])
        self.assertIsNone(self.store.habit_profile("posture"))
        self.assertEqual(self.store.settings()["cooldown_minutes"], 22)

    def test_habit_becomes_established_after_ten_occurrences_on_three_days(self):
        start = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)
        alerts = []
        for index in range(10):
            stamp = (start + timedelta(days=index % 3, minutes=index)).isoformat()
            self.store.start_event(stamp, 0.9, "Gemini confirmed slouching")
            alerts.append(self.store.register_habit_occurrence("posture", stamp))
        self.assertEqual(alerts[0]["alert_type"], "first_added")
        self.assertEqual(alerts[1]["alert_type"], "occurrence")
        self.assertEqual(alerts[-1]["alert_type"], "established")
        self.assertEqual(alerts[-1]["status"], "established")
        self.assertEqual(alerts[-1]["rolling_occurrences"], 10)
        self.assertEqual(alerts[-1]["rolling_days"], 3)

    def test_habit_catalog_returns_profiles(self):
        stamp = "2026-08-22T12:00:00+00:00"
        self.store.start_event(stamp, 0.9, "confirmed")
        self.store.register_habit_occurrence("posture", stamp)
        profiles = self.store.habit_profiles()
        self.assertEqual([item["habit_key"] for item in profiles], ["posture"])
        self.assertEqual(profiles[0]["rolling_occurrences"], 1)

    def test_system_prompt_is_persisted_and_validated(self):
        prompt = "Ada is a supportive desk companion. " * 4
        self.assertEqual(self.store.update_system_prompt(prompt), prompt.strip())
        self.assertEqual(self.store.system_prompt("default"), prompt.strip())
        with self.assertRaises(ValueError):
            self.store.update_system_prompt("too short")

    def test_calibration_persists_numeric_baseline(self):
        monitor = PostureMonitor(self.store)
        monitor.CALIBRATION_SECONDS = 2
        monitor.MIN_CALIBRATION_SAMPLES = 3
        monitor.start_calibration(now=0)
        for timestamp in (0, 0.5, 1, 1.5, 2):
            monitor.process(pose(), now=timestamp)
        self.assertTrue(monitor.status(now=2)["calibrated"])
        self.assertEqual(monitor.state, "good")
        self.assertEqual(set(self.store.calibration()), {"head_forward", "neck_angle", "torso_angle", "torso_length", "head_height"})

    def test_sustained_slouch_creates_one_episode_and_recovery_closes_it(self):
        baseline = posture_features(pose())
        self.store.save_calibration(baseline, 60)
        monitor = PostureMonitor(self.store)
        wall = datetime(2026, 8, 23, 12, 0).astimezone()
        for timestamp in range(22):
            monitor.process(pose(slouch=True), now=float(timestamp), wall_time=wall)
            if monitor.take_verification_request():
                monitor.apply_verification({"slouching": True, "confidence": 0.91, "reason": "Visible forward head posture"})
        self.assertEqual(monitor.state, "slouching")
        self.assertIsNotNone(monitor.last_event_id)
        self.assertEqual(len(self.store.events()), 1)
        for timestamp in range(22, 42):
            monitor.process(pose(), now=float(timestamp), wall_time=wall)
        event = self.store.events()[0]
        self.assertEqual(monitor.state, "cooldown")
        self.assertIsNotNone(event["ended_at"])
        self.assertEqual(event["reminded"], 1)
        self.assertAlmostEqual(event["gemini_confidence"], 0.91)

    def test_gemini_rejection_prevents_event_logging(self):
        good = posture_features(pose())
        self.store.save_calibration(good, 60)
        self.store.save_calibration(posture_features(pose(slouch=True)), 60, kind="slouch")
        monitor = PostureMonitor(self.store)
        wall = datetime(2026, 8, 23, 12, 0).astimezone()
        for timestamp in range(40):
            monitor.process(pose(slouch=True), now=float(timestamp), wall_time=wall)
            if monitor.take_verification_request():
                monitor.apply_verification({"slouching": False, "confidence": 0.88, "reason": "Appears upright"})
        self.assertEqual(monitor.gemini_status, "not_confirmed")
        self.assertEqual(self.store.events(), [])

    def test_invalid_landmarks_do_not_trigger(self):
        self.store.save_calibration(posture_features(pose()), 60)
        monitor = PostureMonitor(self.store)
        invalid = pose(slouch=True)
        for point in invalid["keypoints"]:
            point["score"] = 0.1
        wall = datetime(2026, 8, 23, 12, 0).astimezone()
        for timestamp in range(60):
            monitor.process(invalid, now=float(timestamp), wall_time=wall)
        self.assertEqual(self.store.events(), [])


if __name__ == "__main__":
    unittest.main()

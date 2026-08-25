import io
import unittest

from PIL import Image

from backend.hailo_vision import (
    POSE_HEF,
    VisionBackendManager,
    letterbox,
    repair_hailo_keypoint_scores,
    restore_display_box,
    restore_display_point,
    select_primary_person,
    unletterbox_box,
    unletterbox_point,
)


class FakeBBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.values = xmin, ymin, xmax, ymax

    def xmin(self): return self.values[0]
    def ymin(self): return self.values[1]
    def xmax(self): return self.values[2]
    def ymax(self): return self.values[3]


class HailoVisionTests(unittest.TestCase):
    def test_default_pose_model_matches_raspberry_pi_hailo_package(self):
        system_model = "/usr/share/hailo-models/yolov8s_pose_h8.hef"
        if __import__("pathlib").Path(system_model).exists():
            self.assertEqual(str(POSE_HEF), system_model)

    def test_repairs_hailo_scores_when_ui_would_hide_entire_pose(self):
        points = [{"score": .12}, {"score": .03}]
        self.assertTrue(repair_hailo_keypoint_scores(points, .81))
        self.assertEqual([point["score"] for point in points], [.81, .81])

        usable = [{"score": .12}, {"score": .42}]
        self.assertFalse(repair_hailo_keypoint_scores(usable, .81))
        self.assertEqual([point["score"] for point in usable], [.12, .42])

    def test_letterbox_and_reverse_mapping_preserve_four_three_frame(self):
        image = Image.new("RGB", (640, 480), "white")
        encoded = io.BytesIO(); image.save(encoded, "JPEG")
        rgb, original, mapping = letterbox(encoded.getvalue())
        self.assertEqual(rgb.shape, (640, 640, 3))
        self.assertEqual(original, (640, 480))
        self.assertEqual(mapping, (1.0, 0.0, 80.0))
        self.assertEqual(unletterbox_point(.5, .5, original, mapping), (.5, .5))
        box = unletterbox_box(FakeBBox(.25, .3125, .75, .6875), original, mapping)
        self.assertEqual(box, {"x": .25, "y": .25, "width": .5, "height": .5})

    def test_primary_person_prefers_size_then_continuity(self):
        left = {"x": .1, "y": .1, "width": .2, "height": .4, "score": .9}
        right = {"x": .55, "y": .1, "width": .35, "height": .7, "score": .8}
        self.assertIs(select_primary_person([left, right], None), right)
        self.assertIs(select_primary_person([left, right], (.2, .3)), left)

    def test_model_only_rotation_restores_display_coordinates(self):
        self.assertEqual(restore_display_point(.2, .3, True), (.8, .7))
        box = restore_display_box({"x": .1, "y": .2, "width": .3, "height": .4}, True)
        self.assertAlmostEqual(box["x"], .6)
        self.assertAlmostEqual(box["y"], .4)
        self.assertEqual((box["width"], box["height"]), (.3, .4))

    def test_cpu_mode_keeps_contract_and_reports_backend(self):
        image = Image.new("RGB", (320, 240), "black")
        encoded = io.BytesIO(); image.save(encoded, "JPEG")
        manager = VisionBackendManager("cpu")
        pose = manager.pose.infer(encoded.getvalue())
        detection = manager.detection.infer(encoded.getvalue())
        self.assertEqual(pose["backend"], "cpu")
        self.assertEqual(pose["model"], "movenet_lightning")
        self.assertEqual(len(pose["keypoints"]), 17)
        self.assertEqual(detection["backend"], "cpu")
        self.assertEqual(detection["model"], "efficientdet_lite0")
        self.assertIn("detections", detection)

    def test_auto_runtime_failure_is_one_way_cpu_fallback(self):
        manager = VisionBackendManager("auto")
        manager.mode = "hailo"
        manager._hailo_pose.infer = lambda _: (_ for _ in ()).throw(RuntimeError("device lost"))
        manager._hailo_pose.close = lambda: None
        manager._hailo_detection.close = lambda: None
        manager._cpu_pose.infer = lambda _: {"keypoints": []}
        first = manager.pose.infer(b"frame")
        second = manager.pose.infer(b"frame")
        self.assertEqual(first["backend"], "cpu_transient")
        self.assertEqual(second["backend"], "cpu_transient")
        result = manager.pose.infer(b"frame")
        self.assertEqual(result["backend"], "cpu")
        self.assertTrue(manager.status()["fallback_active"])
        self.assertEqual(manager.status()["last_error"], "device lost")


if __name__ == "__main__":
    unittest.main()

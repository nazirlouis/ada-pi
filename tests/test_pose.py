import io
import unittest

from PIL import Image

from backend.pose import KEYPOINT_NAMES, MODEL_PATH, PoseEstimator


class PoseEstimatorTests(unittest.TestCase):
    def test_model_is_bundled(self) -> None:
        self.assertTrue(MODEL_PATH.exists())
        self.assertGreater(MODEL_PATH.stat().st_size, 1_000_000)

    def test_inference_returns_seventeen_normalized_keypoints(self) -> None:
        image = Image.new("RGB", (320, 240), "black")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        result = PoseEstimator(threads=1).infer(buffer.getvalue())
        self.assertEqual((result["width"], result["height"]), (320, 240))
        self.assertEqual(len(result["keypoints"]), 17)
        self.assertEqual(tuple(point["name"] for point in result["keypoints"]), KEYPOINT_NAMES)
        for point in result["keypoints"]:
            self.assertGreaterEqual(point["x"], 0)
            self.assertLessEqual(point["x"], 1)
            self.assertGreaterEqual(point["y"], 0)
            self.assertLessEqual(point["y"], 1)
            self.assertGreaterEqual(point["score"], 0)
            self.assertLessEqual(point["score"], 1)


if __name__ == "__main__":
    unittest.main()

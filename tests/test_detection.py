import io
import unittest

from PIL import Image

from backend.detection import COCO_LABELS, MODEL_PATH, ObjectDetector


class ObjectDetectorTests(unittest.TestCase):
    def test_model_is_bundled(self) -> None:
        self.assertTrue(MODEL_PATH.exists())
        self.assertGreater(MODEL_PATH.stat().st_size, 4_000_000)

    def test_label_map_preserves_coco_indices(self) -> None:
        self.assertEqual(len(COCO_LABELS), 90)
        self.assertEqual(COCO_LABELS[0], "person")
        self.assertEqual(COCO_LABELS[2], "car")
        self.assertEqual(COCO_LABELS[89], "toothbrush")

    def test_inference_returns_normalized_detections(self) -> None:
        image = Image.new("RGB", (320, 240), "black")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        result = ObjectDetector(threads=1).infer(buffer.getvalue())
        self.assertEqual((result["width"], result["height"]), (320, 240))
        self.assertIsInstance(result["detections"], list)
        self.assertGreater(result["inference_ms"], 0)
        for detection in result["detections"]:
            self.assertIn(detection["label"], COCO_LABELS)
            self.assertGreaterEqual(detection["score"], 0.45)
            for key in ("x", "y", "width", "height"):
                self.assertGreaterEqual(detection[key], 0)
                self.assertLessEqual(detection[key], 1)


if __name__ == "__main__":
    unittest.main()

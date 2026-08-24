"""Lazy CPU EfficientDet-Lite0 object detector."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).with_name("models") / "efficientdet_lite0_int8.tflite"

# EfficientDet's embedded COCO label map retains gaps from the original COCO
# category IDs. Keeping those placeholders is necessary to align class indices.
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "???", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella",
    "???", "???", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "???", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "???", "dining table", "???",
    "???", "toilet", "???", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "???", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer",
    "toothbrush",
)


class ObjectDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threads: int = 3,
        threshold: float = 0.45,
        max_detections: int = 10,
    ) -> None:
        self.model_path = model_path
        self.threads = threads
        self.threshold = threshold
        self.max_detections = max_detections
        self._interpreter = None
        self._input_index = 0
        self._outputs: dict[str, int] = {}
        self._input_size = (320, 320)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._interpreter is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not self.model_path.exists():
            raise RuntimeError(f"Detection model missing: {self.model_path}")
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        self._input_index = input_detail["index"]
        self._input_size = (int(input_detail["shape"][2]), int(input_detail["shape"][1]))
        outputs = interpreter.get_output_details()
        # The bundled model exposes boxes, classes, scores, and count in this
        # shape pattern; using shapes avoids relying on generated tensor names.
        for detail in outputs:
            shape = tuple(int(value) for value in detail["shape"])
            if len(shape) == 3 and shape[-1] == 4:
                self._outputs["boxes"] = detail["index"]
            elif shape == (1,):
                self._outputs["count"] = detail["index"]
            elif "classes" not in self._outputs:
                self._outputs["classes"] = detail["index"]
            else:
                self._outputs["scores"] = detail["index"]
        # For this EfficientDet export, the two [1,N] tensors are ordered
        # classes then scores by index. Normalize that explicitly.
        vector_outputs = sorted(
            detail["index"] for detail in outputs
            if tuple(int(value) for value in detail["shape"]) != (1,)
            and not (len(detail["shape"]) == 3 and int(detail["shape"][-1]) == 4)
        )
        self._outputs["classes"], self._outputs["scores"] = vector_outputs
        self._interpreter = interpreter

    def infer(self, jpeg: bytes) -> dict[str, object]:
        started = time.perf_counter()
        with self._lock:
            self.load()
            with Image.open(io.BytesIO(jpeg)) as image:
                width, height = image.size
                rgb = image.convert("RGB").resize(self._input_size, Image.Resampling.BILINEAR)
                tensor = np.asarray(rgb, dtype=np.uint8)[None, ...]
            self._interpreter.set_tensor(self._input_index, tensor)
            self._interpreter.invoke()
            boxes = self._interpreter.get_tensor(self._outputs["boxes"])[0]
            classes = self._interpreter.get_tensor(self._outputs["classes"])[0]
            scores = self._interpreter.get_tensor(self._outputs["scores"])[0]
            count = int(self._interpreter.get_tensor(self._outputs["count"])[0])

        detections = []
        for box, class_value, score_value in zip(boxes[:count], classes[:count], scores[:count]):
            score = float(score_value)
            class_id = int(class_value)
            if score < self.threshold or not 0 <= class_id < len(COCO_LABELS):
                continue
            label = COCO_LABELS[class_id]
            if label == "???":
                continue
            ymin, xmin, ymax, xmax = (float(value) for value in box)
            detections.append({
                "label": label,
                "score": round(score, 4),
                "x": round(max(0.0, min(1.0, xmin)), 5),
                "y": round(max(0.0, min(1.0, ymin)), 5),
                "width": round(max(0.0, min(1.0, xmax) - max(0.0, min(1.0, xmin))), 5),
                "height": round(max(0.0, min(1.0, ymax) - max(0.0, min(1.0, ymin))), 5),
            })
            if len(detections) >= self.max_detections:
                break

        return {
            "detections": detections,
            "width": width,
            "height": height,
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
        }

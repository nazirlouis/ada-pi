"""Lazy CPU MoveNet pose estimator."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).with_name("models") / "movenet_singlepose_lightning_int8.tflite"
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


class PoseEstimator:
    def __init__(self, model_path: Path = MODEL_PATH, threads: int = 3) -> None:
        self.model_path = model_path
        self.threads = threads
        self._interpreter = None
        self._input_index = 0
        self._output_index = 0
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._interpreter is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not self.model_path.exists():
            raise RuntimeError(f"Pose model missing: {self.model_path}")
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        interpreter.allocate_tensors()
        self._input_index = interpreter.get_input_details()[0]["index"]
        self._output_index = interpreter.get_output_details()[0]["index"]
        self._interpreter = interpreter

    def infer(self, jpeg: bytes) -> dict[str, object]:
        started = time.perf_counter()
        with self._lock:
            self.load()
            with Image.open(io.BytesIO(jpeg)) as image:
                original_size = image.size
                rgb = image.convert("RGB").resize((192, 192), Image.Resampling.BILINEAR)
                tensor = np.asarray(rgb, dtype=np.uint8)[None, ...]
            self._interpreter.set_tensor(self._input_index, tensor)
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._output_index)[0, 0]
        keypoints = [
            {"name": name, "x": round(float(point[1]), 5),
             "y": round(float(point[0]), 5), "score": round(float(point[2]), 5)}
            for name, point in zip(KEYPOINT_NAMES, raw)
        ]
        return {
            "keypoints": keypoints,
            "width": original_size[0],
            "height": original_size[1],
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
        }

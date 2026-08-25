"""Hailo-8 accelerated detection and pose with coordinated CPU fallback."""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .detection import COCO_LABELS, ObjectDetector
from .pose import KEYPOINT_NAMES, PoseEstimator

logger = logging.getLogger("voice.hailo_vision")

# HailoRT 4.23/TAPPAS 5.1 can crash when two Python-owned GStreamer pipelines
# configure or dispatch against one VDevice concurrently. ADA only needs ten
# total inferences per second, so serialize device access while retaining both
# persistent configured networks.
_HAILO_EXECUTION_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parents[1]
HAILO_MODEL_DIR = Path(os.environ.get("ADA_HAILO_MODEL_DIR", ROOT / "data" / "models" / "hailo8"))
DETECTION_HEF = Path(os.environ.get("ADA_HAILO_DETECTION_HEF", HAILO_MODEL_DIR / "yolov8m.hef"))
SYSTEM_POSE_HEF = Path("/usr/share/hailo-models/yolov8s_pose_h8.hef")
# Use the Raspberry Pi package's Hailo-8 pose HEF by default. This is the
# exact model selected by hailo_yolov8_pose.json and is therefore guaranteed
# to match the installed camera/Hailo stack. Keep the Model Zoo HEF as a
# fallback for systems without the Raspberry Pi model package.
DEFAULT_POSE_HEF = SYSTEM_POSE_HEF if SYSTEM_POSE_HEF.exists() else HAILO_MODEL_DIR / "yolov8m_pose.hef"
POSE_HEF = Path(os.environ.get("ADA_HAILO_POSE_HEF", DEFAULT_POSE_HEF))
POSTPROCESS_DIR = Path("/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes")
DETECTION_POSTPROCESS = POSTPROCESS_DIR / "libyolo_hailortpp_post.so"
POSE_POSTPROCESS = POSTPROCESS_DIR / "libyolov8pose_post.so"
HAILO_ROTATE_180 = os.environ.get("ADA_HAILO_ROTATE_180", "false").lower() not in {"0", "false", "no"}


def letterbox(jpeg: bytes, size: int = 640, rotate_180: bool = False) -> tuple[np.ndarray, tuple[int, int], tuple[float, float, float]]:
    """Return square RGB input plus (scale, x padding, y padding) mapping."""
    with Image.open(io.BytesIO(jpeg)) as source:
        image = source.convert("RGB")
        if rotate_180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        original = image.size
        scale = min(size / original[0], size / original[1])
        resized = (max(1, round(original[0] * scale)), max(1, round(original[1] * scale)))
        image = image.resize(resized, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), "black")
        pad_x, pad_y = (size - resized[0]) / 2, (size - resized[1]) / 2
        canvas.paste(image, (round(pad_x), round(pad_y)))
        return np.asarray(canvas, dtype=np.uint8), original, (scale, pad_x, pad_y)


def unletterbox_point(x: float, y: float, original: tuple[int, int], mapping: tuple[float, float, float], size: int = 640) -> tuple[float, float]:
    scale, pad_x, pad_y = mapping
    px = (x * size - pad_x) / scale / original[0]
    py = (y * size - pad_y) / scale / original[1]
    return max(0.0, min(1.0, px)), max(0.0, min(1.0, py))


def unletterbox_box(bbox: Any, original: tuple[int, int], mapping: tuple[float, float, float]) -> dict[str, float]:
    x1, y1 = unletterbox_point(float(bbox.xmin()), float(bbox.ymin()), original, mapping)
    x2, y2 = unletterbox_point(float(bbox.xmax()), float(bbox.ymax()), original, mapping)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}


def restore_display_point(x: float, y: float, rotate_180: bool) -> tuple[float, float]:
    return (1.0 - x, 1.0 - y) if rotate_180 else (x, y)


def restore_display_box(box: dict[str, float], rotate_180: bool) -> dict[str, float]:
    if not rotate_180:
        return box
    return {**box, "x": max(0.0, 1.0 - box["x"] - box["width"]),
            "y": max(0.0, 1.0 - box["y"] - box["height"])}


def select_primary_person(people: list[dict[str, Any]], previous_center: tuple[float, float] | None) -> dict[str, Any] | None:
    """Prefer continuity, otherwise the largest confident person."""
    if not people:
        return None
    if previous_center is None:
        return max(people, key=lambda person: float(person["score"]) * float(person["width"]) * float(person["height"]))
    return min(
        people,
        key=lambda person: (float(person["x"]) + float(person["width"]) / 2 - previous_center[0]) ** 2
        + (float(person["y"]) + float(person["height"]) / 2 - previous_center[1]) ** 2,
    )


def repair_hailo_keypoint_scores(points: list[dict[str, Any]], person_score: float,
                                 display_threshold: float = .25) -> bool:
    """Repair TAPPAS landmark scores when their scale hides every valid joint."""
    scores = [float(point.get("score", 0)) for point in points]
    should_repair = bool(scores and max(scores) < display_threshold and person_score >= .5)
    if should_repair:
        for point in points:
            point["score"] = round(person_score, 5)
    return should_repair


def _hailo_imports() -> tuple[Any, Any]:
    # Raspberry Pi packages the native bindings in dist-packages, outside a
    # normal venv. Keep the bridge narrow instead of exposing all system Python
    # packages through ADA's environment.
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages not in sys.path:
        sys.path.append(dist_packages)
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    import hailo
    return Gst, hailo


def _hailo_device_class() -> Any:
    """Load the system HailoRT control binding without exposing the whole venv."""
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages not in sys.path:
        sys.path.append(dist_packages)
    from hailo_platform import Device
    return Device


class _HailoPipeline:
    def __init__(self, hef: Path, postprocess: Path, kind: str, timeout: float = 3.0) -> None:
        self.hef, self.postprocess, self.kind, self.timeout = hef, postprocess, kind, timeout
        self.pipeline = self.source = self.sink = self.Gst = self.hailo = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.pipeline is not None:
            return
        if not Path("/dev/hailo0").exists():
            raise RuntimeError("Hailo device /dev/hailo0 is unavailable")
        if not self.hef.exists():
            raise RuntimeError(f"Hailo {self.kind} model missing: {self.hef}")
        if not self.postprocess.exists():
            raise RuntimeError(f"Hailo {self.kind} postprocessor missing: {self.postprocess}")
        Gst, hailo = _hailo_imports()
        Gst.init(None)
        net_options = (
            "nms-score-threshold=0.45 nms-iou-threshold=0.45 "
            "output-format-type=HAILO_FORMAT_TYPE_FLOAT32 " if self.kind == "detection" else ""
        )
        # The installed TAPPAS 5.1 detection library exports
        # filter_letterbox, while its pose library exports only filter/yolov8.
        postprocess_function = "filter_letterbox" if self.kind == "detection" else "filter"
        pipeline = Gst.parse_launch(
            "appsrc name=source is-live=true block=true format=time "
            "caps=video/x-raw,format=RGB,width=640,height=640,framerate=5/1 ! "
            "queue leaky=downstream max-size-buffers=2 ! "
            f"hailonet hef-path=\"{self.hef}\" batch-size=1 {net_options}"
            "vdevice-group-id=ada-vision scheduling-algorithm=1 force-writable=true ! "
            "queue leaky=downstream max-size-buffers=2 ! "
            f"hailofilter function-name={postprocess_function} so-path=\"{self.postprocess}\" "
            "remove-tensors=false qos=false ! "
            "queue leaky=downstream max-size-buffers=2 ! "
            "appsink name=sink sync=false max-buffers=1 drop=true"
        )
        source, sink = pipeline.get_by_name("source"), pipeline.get_by_name("sink")
        if source is None or sink is None:
            raise RuntimeError("Could not construct Hailo GStreamer pipeline")
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Hailo {self.kind} pipeline failed to start")
        self.Gst, self.hailo, self.pipeline, self.source, self.sink = Gst, hailo, pipeline, source, sink
        logger.info("Hailo %s active: %s", self.kind, self.hef.name)

    def run(self, rgb: np.ndarray) -> Any:
        with _HAILO_EXECUTION_LOCK, self._lock:
            self.load()
            assert self.Gst is not None and self.source is not None and self.sink is not None
            payload = rgb.tobytes()
            buffer = self.Gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = time.monotonic_ns()
            flow = self.source.emit("push-buffer", buffer)
            if flow != self.Gst.FlowReturn.OK:
                raise RuntimeError(f"Hailo {self.kind} rejected an input frame: {flow}")
            sample = self.sink.emit("try-pull-sample", int(self.timeout * 1_000_000_000))
            if sample is None:
                bus = self.pipeline.get_bus()
                message = bus.pop_filtered(self.Gst.MessageType.ERROR)
                if message is not None:
                    error, debug = message.parse_error()
                    raise RuntimeError(f"Hailo {self.kind} pipeline error: {error}; {debug}")
                raise RuntimeError(f"Hailo {self.kind} inference timed out")
            return self.hailo.get_roi_from_buffer(sample.get_buffer())

    def close(self) -> None:
        with _HAILO_EXECUTION_LOCK, self._lock:
            if self.pipeline is not None and self.Gst is not None:
                self.pipeline.set_state(self.Gst.State.NULL)
            self.pipeline = self.source = self.sink = None


class HailoObjectDetector:
    target_fps = 5.0

    def __init__(self, model_path: Path = DETECTION_HEF, threshold: float = 0.45, max_detections: int = 10) -> None:
        self.model_path, self.threshold, self.max_detections = model_path, threshold, max_detections
        self._pipeline = _HailoPipeline(model_path, DETECTION_POSTPROCESS, "detection")

    def infer(self, jpeg: bytes) -> dict[str, object]:
        started = time.perf_counter()
        rgb, original, mapping = letterbox(jpeg, rotate_180=HAILO_ROTATE_180)
        roi = self._pipeline.run(rgb)
        tensor_names = [str(tensor.name()) for tensor in roi.get_tensors()]
        detections = []
        raw_items = roi.get_objects_typed(self._pipeline.hailo.HAILO_DETECTION)
        for item in raw_items:
            score = float(item.get_confidence())
            class_id = int(item.get_class_id())
            label = str(item.get_label() or "")
            if not label and 0 <= class_id < len(COCO_LABELS):
                label = COCO_LABELS[class_id]
            if score < self.threshold or not label or label == "???":
                continue
            box = restore_display_box(unletterbox_box(item.get_bbox(), original, mapping), HAILO_ROTATE_180)
            detections.append({"label": label, "score": round(score, 4), **{key: round(value, 5) for key, value in box.items()}})
            if len(detections) >= self.max_detections:
                break
        return {"detections": detections, "width": original[0], "height": original[1],
                "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                "backend": "hailo", "model": "yolov8m", "device": "hailo8",
                "raw_detections": len(raw_items), "tensor_names": tensor_names}

    def close(self) -> None:
        self._pipeline.close()


class HailoPoseEstimator:
    target_fps = 5.0

    def __init__(self, model_path: Path = POSE_HEF, confidence: float = 0.0) -> None:
        self.model_path, self.confidence = model_path, confidence
        self._pipeline = _HailoPipeline(model_path, POSE_POSTPROCESS, "pose")
        self._previous_center: tuple[float, float] | None = None

    def infer(self, jpeg: bytes) -> dict[str, object]:
        started = time.perf_counter()
        rgb, original, mapping = letterbox(jpeg, rotate_180=HAILO_ROTATE_180)
        roi = self._pipeline.run(rgb)
        tensor_names = [str(tensor.name()) for tensor in roi.get_tensors()]
        people = []
        hailo = self._pipeline.hailo
        raw_items = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for item in raw_items:
            label = str(item.get_label() or "").lower()
            if label != "person" and int(item.get_class_id()) != 0:
                continue
            box = restore_display_box(unletterbox_box(item.get_bbox(), original, mapping), HAILO_ROTATE_180)
            landmarks = item.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks:
                continue
            bbox = item.get_bbox()
            points = []
            for name, point in zip(KEYPOINT_NAMES, landmarks[0].get_points()):
                input_x = float(bbox.xmin()) + float(point.x()) * float(bbox.width())
                input_y = float(bbox.ymin()) + float(point.y()) * float(bbox.height())
                x, y = unletterbox_point(input_x, input_y, original, mapping)
                x, y = restore_display_point(x, y, HAILO_ROTATE_180)
                points.append({"name": name, "x": round(x, 5), "y": round(y, 5),
                               "score": round(float(point.confidence()), 5)})
            # TAPPAS 5.1 can expose H8 pose point confidences on the wrong
            # scale. Hailo's reference Pi callback draws the returned landmarks
            # without a point-confidence gate; otherwise Ada's .25 UI threshold
            # can hide the entire valid skeleton.
            repaired_scores = repair_hailo_keypoint_scores(points, float(item.get_confidence()))
            people.append({**box, "score": float(item.get_confidence()), "keypoints": points,
                           "scores_repaired": repaired_scores})
        primary = select_primary_person(people, self._previous_center)
        if primary is None:
            keypoints = [{"name": name, "x": 0.0, "y": 0.0, "score": 0.0} for name in KEYPOINT_NAMES]
        else:
            self._previous_center = (float(primary["x"]) + float(primary["width"]) / 2,
                                     float(primary["y"]) + float(primary["height"]) / 2)
            keypoints = primary["keypoints"]
        scores = [float(point["score"]) for point in keypoints]
        return {"keypoints": keypoints, "width": original[0], "height": original[1],
                "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                "backend": "hailo", "model": self.model_path.stem, "device": "hailo8",
                "raw_detections": len(raw_items), "people": len(people), "tensor_names": tensor_names,
                "keypoint_score_min": round(min(scores), 5), "keypoint_score_max": round(max(scores), 5),
                "person_score": round(float(primary["score"]), 5) if primary else None,
                "keypoint_scores_repaired": bool(primary and primary.get("scores_repaired"))}

    def close(self) -> None:
        self._pipeline.close()


class VisionBackendManager:
    """Select Hailo when available and make runtime fallback one-way."""

    def __init__(self, requested: str | None = None) -> None:
        requested = (requested or os.environ.get("ADA_VISION_BACKEND", "auto")).lower()
        if requested not in {"auto", "hailo", "cpu"}:
            raise ValueError("ADA_VISION_BACKEND must be auto, hailo, or cpu")
        self.requested = requested
        self.mode = "cpu" if requested == "cpu" else "hailo"
        self.fallback_active = False
        self.last_error: str | None = None
        self.last_success_at: float | None = None
        self.latest_results: dict[str, dict[str, object]] = {}
        self._last_good: dict[str, dict[str, object]] = {}
        self._hailo_failures = {"pose": 0, "detection": 0}
        self._lock = threading.Lock()
        self._temperature_device = None
        self._temperature_reading: dict[str, object] = {}
        self._temperature_read_at = 0.0
        self._cpu_pose, self._cpu_detection = PoseEstimator(), ObjectDetector()
        self._hailo_pose, self._hailo_detection = HailoPoseEstimator(), HailoObjectDetector()
        self.pose = _ManagedEstimator(self, "pose")
        self.detection = _ManagedEstimator(self, "detection")
        if requested == "auto":
            problem = self._preflight_problem()
            if problem:
                self.mode, self.fallback_active, self.last_error = "cpu", True, problem
                logger.warning("Hailo preflight failed; starting CPU fallback: %s", problem)

    def _preflight_problem(self) -> str | None:
        required = (Path("/dev/hailo0"), DETECTION_HEF, POSE_HEF, DETECTION_POSTPROCESS, POSE_POSTPROCESS)
        missing = [str(path) for path in required if not path.exists()]
        return f"Missing Hailo runtime files: {', '.join(missing)}" if missing else None

    def infer(self, kind: str, jpeg: bytes) -> dict[str, object]:
        with self._lock:
            mode = self.mode
        estimator = (self._hailo_pose if kind == "pose" else self._hailo_detection) if mode == "hailo" else (self._cpu_pose if kind == "pose" else self._cpu_detection)
        try:
            result = estimator.infer(jpeg)
            result.setdefault("backend", mode)
            result.setdefault("model", "movenet_lightning" if kind == "pose" else "efficientdet_lite0")
            result.setdefault("device", "cpu" if mode == "cpu" else "hailo8")
            if mode == "hailo":
                self._hailo_failures[kind] = 0
                self._last_good[kind] = dict(result)
                if not any(self._hailo_failures.values()):
                    self.last_error = None
            self.last_success_at = time.time()
            self.latest_results[kind] = {
                "inference_ms": result.get("inference_ms"),
                "raw_detections": result.get("raw_detections"),
                "result_count": len(result.get("detections", [])) if kind == "detection"
                else sum(float(point.get("score", 0)) >= .25 for point in result.get("keypoints", [])),
                "people": result.get("people"),
                "tensor_names": result.get("tensor_names"),
                "keypoint_score_min": result.get("keypoint_score_min"),
                "keypoint_score_max": result.get("keypoint_score_max"),
                "person_score": result.get("person_score"),
                "keypoint_scores_repaired": result.get("keypoint_scores_repaired"),
            }
            return result
        except Exception as exc:
            if mode != "hailo" or self.requested == "hailo":
                self.last_error = str(exc)
                raise
            with self._lock:
                self._hailo_failures[kind] += 1
                failures = self._hailo_failures[kind]
                self.last_error = str(exc)
            if failures < 3:
                logger.warning("Transient Hailo %s failure %d/3: %s", kind, failures, exc)
                cached = self._last_good.get(kind)
                if cached is not None:
                    return {**cached, "stale": True, "transient_error": str(exc)}
                cpu = self._cpu_pose if kind == "pose" else self._cpu_detection
                result = cpu.infer(jpeg)
                return {**result, "backend": "cpu_transient", "device": "cpu",
                        "transient_error": str(exc)}
            with self._lock:
                if self.mode == "hailo":
                    self.mode, self.fallback_active, self.last_error = "cpu", True, str(exc)
                    self._hailo_pose.close(); self._hailo_detection.close()
                    logger.warning("Hailo vision unavailable; switching to CPU fallback: %s", exc)
            return self.infer(kind, jpeg)

    def _chip_temperature(self) -> dict[str, object]:
        """Return a short-lived cached reading from both Hailo-8 sensors."""
        now = time.monotonic()
        if now - self._temperature_read_at < 2.5:
            return dict(self._temperature_reading)
        try:
            with _HAILO_EXECUTION_LOCK:
                if self._temperature_device is None:
                    self._temperature_device = _hailo_device_class()()
                reading = self._temperature_device.control.get_chip_temperature()
            sensors = [round(float(reading.ts0_temperature), 1),
                       round(float(reading.ts1_temperature), 1)]
            self._temperature_reading = {
                "hailo_8_temperature": round(max(sensors), 1),
                "hailo_8_sensor_0_temperature": sensors[0],
                "hailo_8_sensor_1_temperature": sensors[1],
            }
        except Exception as exc:
            self._temperature_reading = {"hailo_8_temperature_error": str(exc)}
        self._temperature_read_at = now
        return dict(self._temperature_reading)

    def status(self, include_temperature: bool = False) -> dict[str, object]:
        target = 5.0 if self.mode == "hailo" else None
        result = {"requested_backend": self.requested, "backend": self.mode,
                "device": "hailo8" if self.mode == "hailo" else "cpu",
                "models": {"detection": "yolov8m" if self.mode == "hailo" else "efficientdet_lite0",
                           "pose": POSE_HEF.stem if self.mode == "hailo" else "movenet_lightning"},
                "model_input_rotation": 180 if self.mode == "hailo" and HAILO_ROTATE_180 else 0,
                "target_fps": target, "fallback_active": self.fallback_active,
                "last_success_at": self.last_success_at, "last_error": self.last_error,
                "consecutive_errors": dict(self._hailo_failures),
                "latest_results": self.latest_results}
        if include_temperature and Path("/dev/hailo0").exists():
            result.update(self._chip_temperature())
        return result

    def close(self) -> None:
        self._hailo_pose.close()
        self._hailo_detection.close()
        if getattr(self, "_temperature_device", None) is not None:
            self._temperature_device.release()
            self._temperature_device = None


class _ManagedEstimator:
    def __init__(self, manager: VisionBackendManager, kind: str) -> None:
        self.manager, self.kind = manager, kind

    @property
    def target_fps(self) -> float:
        return 5.0 if self.manager.mode == "hailo" else (2.0 if self.kind == "pose" else 0.5)

    def infer(self, jpeg: bytes) -> dict[str, object]:
        return self.manager.infer(self.kind, jpeg)

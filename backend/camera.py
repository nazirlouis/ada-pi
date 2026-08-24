"""Shared native Pi camera stream for Gemini vision and local inference."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from contextlib import suppress

logger = logging.getLogger("voice.camera")


class CameraHub:
    """Capture one MJPEG stream and distribute latest frames to many consumers."""

    def __init__(self, fps: int = 5) -> None:
        self.fps = fps
        self.latest_frame: bytes | None = None
        self.latest_frame_at: float | None = None
        self._generation = 0
        self._condition = asyncio.Condition()
        self._process: asyncio.subprocess.Process | None = None
        self._capture_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._capture_task and not self._capture_task.done():
                return
            camera_bin = shutil.which("rpicam-vid")
            if camera_bin is None:
                raise RuntimeError("rpicam-vid not found")
            self._process = await asyncio.create_subprocess_exec(
                camera_bin,
                "--nopreview", "--verbose", "0", "--timeout", "0",
                "--codec", "mjpeg", "--width", "640", "--height", "480",
                "--rotation", "180",
                "--framerate", str(self.fps), "--quality", "80", "--flush",
                "--output", "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._capture_task = asyncio.create_task(self._capture())

    async def _capture(self) -> None:
        buffer = bytearray()
        captured = 0
        assert self._process and self._process.stdout
        try:
            while True:
                chunk = await self._process.stdout.read(65536)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 1:
                            del buffer[:-1]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start:
                            del buffer[:start]
                        break
                    frame = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    self.latest_frame = frame
                    self.latest_frame_at = time.monotonic()
                    captured += 1
                    if captured == 1:
                        logger.info("Pi camera streaming (640x480 MJPEG at %d FPS)", self.fps)
                    async with self._condition:
                        self._generation += 1
                        self._condition.notify_all()
        finally:
            async with self._condition:
                self._condition.notify_all()

    async def frames(self):
        await self.start()
        generation = self._generation
        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._generation != generation
                    or (self._capture_task is not None and self._capture_task.done())
                )
                if self._capture_task is not None and self._capture_task.done():
                    return
                generation = self._generation
                frame = self.latest_frame
            if frame is not None:
                yield frame

    @property
    def generation(self) -> int:
        """Monotonic frame sequence for latest-frame-only consumers."""
        return self._generation

    async def stop(self) -> None:
        if self._capture_task:
            self._capture_task.cancel()
        if self._process and self._process.returncode is None:
            self._process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=2)
        if self._capture_task:
            with suppress(asyncio.CancelledError):
                await self._capture_task
        self._capture_task = None
        self._process = None

"""FastAPI browser-to-realtime-provider audio bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import shutil
import time
from contextlib import suppress
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .realtime_provider import create_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("voice.backend")

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
app = FastAPI(title="Pi Full-Duplex Voice Prototype")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    """Keep long-running kiosk Chromium sessions on the current UI code."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/shutdown", status_code=202)
async def shutdown(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Allow only the locally displayed kiosk to stop this backend process."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=403, detail="Shutdown is available only from this device")
    background_tasks.add_task(os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}


async def send_json(socket: WebSocket, event_type: str, **data: object) -> None:
    await socket.send_text(json.dumps({"type": event_type, **data}))


@app.websocket("/ws")
async def voice_socket(browser: WebSocket) -> None:
    await browser.accept()
    logger.info("browser connected (%s)", browser.client)
    provider = create_provider()
    closed = asyncio.Event()
    speech_active = asyncio.Event()
    video_mode = os.environ.get("ADA_VIDEO_MODE", "activity").lower()
    latest_camera_frame: bytes | None = None
    last_video_sent = 0.0
    forwarded_video_count = 0
    video_send_lock = asyncio.Lock()
    speech_clear_task: asyncio.Task[None] | None = None

    try:
        await provider.connect()
    except Exception as exc:
        logger.exception("could not open AI session")
        with suppress(Exception):
            await send_json(browser, "error", message=str(exc))
            await browser.close(code=1011)
        return

    try:
        await send_json(browser, "ready")
    except (WebSocketDisconnect, RuntimeError):
        # The kiosk may close while Gemini is still negotiating its session.
        # This is a normal shutdown race, not an AI connection failure.
        await provider.close()
        return

    async def browser_to_provider() -> None:
        nonlocal speech_clear_task
        packet_count = 0
        last_audio_log = time.monotonic()
        try:
            while True:
                message = await browser.receive()
                if message["type"] == "websocket.disconnect":
                    break
                pcm16 = message.get("bytes")
                if pcm16:
                    await provider.send_audio(pcm16)
                    packet_count += 1
                    now = time.monotonic()
                    if now - last_audio_log >= 5:
                        logger.info("audio chunks received (%d in last interval)", packet_count)
                        packet_count = 0
                        last_audio_log = now
                    continue
                text = message.get("text")
                if text:
                    with suppress(ValueError, TypeError):
                        control = json.loads(text)
                        if control.get("type") == "local_speech_started":
                            if speech_clear_task is not None:
                                speech_clear_task.cancel()
                                speech_clear_task = None
                            speech_active.set()
                            if video_mode == "activity":
                                await forward_video_frame(latest_camera_frame)
                        elif control.get("type") == "local_speech_stopped":
                            if speech_clear_task is not None:
                                speech_clear_task.cancel()
                            speech_clear_task = asyncio.create_task(
                                clear_speech_activity_after_grace()
                            )
        except WebSocketDisconnect:
            pass
        finally:
            closed.set()

    async def forward_video_frame(frame: bytes | None) -> None:
        """Forward at most one frame/sec, including activity-triggered frames."""
        nonlocal last_video_sent, forwarded_video_count
        if frame is None:
            return
        async with video_send_lock:
            now = time.monotonic()
            if now - last_video_sent < 0.95:
                return
            await provider.send_video(frame)
            last_video_sent = time.monotonic()
            forwarded_video_count += 1
            if forwarded_video_count == 1 or forwarded_video_count % 10 == 0:
                logger.info(
                    "video frames forwarded (%d total, latest=%d bytes)",
                    forwarded_video_count,
                    len(frame),
                )

    async def clear_speech_activity_after_grace() -> None:
        # A final camera frame after the utterance helps Gemini associate an
        # object being shown with the question that was just asked.
        await asyncio.sleep(1.5)
        speech_active.clear()

    async def provider_to_browser() -> None:
        nonlocal speech_clear_task
        forward_audio = True
        try:
            async for event in provider.events():
                if event.type == "audio":
                    # A few in-flight provider chunks can arrive after barge-in.
                    # Never let those refill the queue we just cleared.
                    if forward_audio:
                        await browser.send_bytes(event.data["pcm16"])
                elif event.type == "speech_started":
                    if speech_clear_task is not None:
                        speech_clear_task.cancel()
                        speech_clear_task = None
                    speech_active.set()
                    forward_audio = False
                    # Clear audio already queued in Chromium immediately.
                    await send_json(browser, "speech_started")
                    await send_json(browser, "clear_audio")
                    if video_mode == "activity":
                        await forward_video_frame(latest_camera_frame)
                elif event.type == "response_interrupted":
                    forward_audio = False
                    await send_json(browser, "clear_audio")
                    await send_json(browser, event.type)
                elif event.type == "speech_stopped":
                    if speech_clear_task is not None:
                        speech_clear_task.cancel()
                    speech_clear_task = asyncio.create_task(
                        clear_speech_activity_after_grace()
                    )
                    await send_json(browser, "speech_stopped")
                elif event.type == "assistant_transcript_delta":
                    if forward_audio:
                        await send_json(browser, event.type, text=event.data["text"])
                elif event.type == "user_transcript":
                    await send_json(browser, event.type, text=event.data["text"])
                elif event.type == "response_started":
                    forward_audio = True
                    await send_json(browser, event.type)
                else:
                    await send_json(browser, event.type, **event.data)
        finally:
            closed.set()

    async def camera_to_provider() -> None:
        nonlocal latest_camera_frame
        camera_bin = shutil.which("rpicam-vid")
        if camera_bin is None:
            logger.warning("rpicam-vid not found; continuing without video")
            return
        command = [
            camera_bin,
            "--nopreview",
            "--verbose", "0",
            "--timeout", "0",
            "--codec", "mjpeg",
            "--width", "640",
            "--height", "480",
            "--framerate", "1",
            "--quality", "80",
            "--flush",
            "--output", "-",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        buffer = bytearray()
        captured_count = 0
        try:
            while not closed.is_set() and process.stdout is not None:
                chunk = await process.stdout.read(65536)
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
                    if captured_count == 0:
                        logger.info(
                            "Pi camera streaming (mode=%s, 640x480 MJPEG at 1 FPS)",
                            video_mode,
                        )
                    captured_count += 1
                    latest_camera_frame = frame
                    if video_mode == "activity" and not speech_active.is_set():
                        continue
                    await forward_video_frame(frame)
        finally:
            if process.returncode is None:
                process.terminate()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=2)
            if process.returncode is None:
                process.kill()
                await process.wait()
            if captured_count == 0 and process.stderr is not None:
                camera_error = (await process.stderr.read()).decode(errors="replace").strip()
                logger.error(
                    "Pi camera produced no frames%s",
                    f": {camera_error}" if camera_error else "",
                )
            if not closed.is_set():
                logger.warning("Pi camera stream ended unexpectedly")

    tasks = {
        asyncio.create_task(browser_to_provider()),
        asyncio.create_task(provider_to_browser()),
        asyncio.create_task(camera_to_provider()),
    }
    try:
        await closed.wait()
    finally:
        if speech_clear_task is not None:
            speech_clear_task.cancel()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        await provider.close()
        with suppress(Exception):
            await browser.close()
        logger.info("session closed")

"""FastAPI browser-to-realtime-provider audio bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


async def send_json(socket: WebSocket, event_type: str, **data: object) -> None:
    await socket.send_text(json.dumps({"type": event_type, **data}))


@app.websocket("/ws")
async def voice_socket(browser: WebSocket) -> None:
    await browser.accept()
    logger.info("browser connected (%s)", browser.client)
    provider = create_provider()
    closed = asyncio.Event()

    try:
        await provider.connect()
        await send_json(browser, "ready")
    except Exception as exc:
        logger.exception("could not open AI session")
        await send_json(browser, "error", message=str(exc))
        await browser.close(code=1011)
        return

    async def browser_to_provider() -> None:
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
        except WebSocketDisconnect:
            pass
        finally:
            closed.set()

    async def provider_to_browser() -> None:
        forward_audio = True
        try:
            async for event in provider.events():
                if event.type == "audio":
                    # A few in-flight provider chunks can arrive after barge-in.
                    # Never let those refill the queue we just cleared.
                    if forward_audio:
                        await browser.send_bytes(event.data["pcm16"])
                elif event.type == "speech_started":
                    forward_audio = False
                    # Clear audio already queued in Chromium immediately.
                    await send_json(browser, "speech_started")
                    await send_json(browser, "clear_audio")
                elif event.type == "response_interrupted":
                    forward_audio = False
                    await send_json(browser, "clear_audio")
                    await send_json(browser, event.type)
                elif event.type == "speech_stopped":
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

    tasks = {
        asyncio.create_task(browser_to_provider()),
        asyncio.create_task(provider_to_browser()),
    }
    try:
        await closed.wait()
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        await provider.close()
        with suppress(Exception):
            await browser.close()
        logger.info("session closed")

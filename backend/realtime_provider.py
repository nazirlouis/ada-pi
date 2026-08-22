"""Gemini Live provider for the browser audio bridge."""

from __future__ import annotations

import abc
import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("voice.provider")


@dataclass(slots=True)
class ProviderEvent:
    type: str
    data: dict[str, Any]


class RealtimeProvider(abc.ABC):
    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def send_audio(self, pcm16: bytes) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[ProviderEvent]: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class GeminiLiveProvider(RealtimeProvider):
    """Gemini 3.1 Flash Live over Google's asynchronous Live API SDK."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
        self.voice = os.environ.get("GEMINI_LIVE_VOICE", "Kore")
        self.instructions = os.environ.get(
            "GEMINI_LIVE_INSTRUCTIONS",
            "You are a concise, friendly voice assistant. Reply naturally and briefly.",
        )
        self._client: Any = None
        self._session_context: Any = None
        self._session: Any = None
        self._closed = False

    async def connect(self) -> None:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self._client = genai.Client(api_key=self.api_key)
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self.instructions,
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": self.voice},
                }
            },
            "realtime_input_config": {
                # Be explicit about barge-in and favor detecting near-end speech
                # over the assistant audio playing through the Pi's speakers.
                "activity_handling": types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                "automatic_activity_detection": {
                    "disabled": False,
                    "start_of_speech_sensitivity": (
                        types.StartSensitivity.START_SENSITIVITY_HIGH
                    ),
                    "end_of_speech_sensitivity": (
                        types.EndSensitivity.END_SENSITIVITY_HIGH
                    ),
                    "prefix_padding_ms": 100,
                    "silence_duration_ms": 300,
                },
            },
        }
        self._session_context = self._client.aio.live.connect(
            model=self.model,
            config=config,
        )
        self._session = await self._session_context.__aenter__()
        logger.info("Gemini Live session connected (model=%s, voice=%s)", self.model, self.voice)

    async def send_audio(self, pcm16: bytes) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm16, mime_type="audio/pcm;rate=16000")
        )

    async def events(self) -> AsyncIterator[ProviderEvent]:
        if self._session is None:
            raise RuntimeError("provider is not connected")

        response_active = False
        input_transcript = ""

        while not self._closed:
            async for message in self._session.receive():
                content = message.server_content
                if content is None:
                    continue

                if content.interrupted:
                    logger.info("assistant interrupted")
                    response_active = False
                    yield ProviderEvent("response_interrupted", {})
                    # Gemini 3.1 can include several content parts in one event.
                    # Any audio/transcript accompanying an interruption belongs
                    # to the cancelled response and must not restart playback.
                    continue

                transcription = content.input_transcription
                if transcription and transcription.text:
                    input_transcript += transcription.text

                output_transcription = content.output_transcription
                if output_transcription and output_transcription.text:
                    if not response_active:
                        response_active = True
                        yield ProviderEvent("response_started", {})
                    yield ProviderEvent(
                        "assistant_transcript_delta",
                        {"text": output_transcription.text},
                    )

                model_turn = content.model_turn
                if model_turn:
                    for part in model_turn.parts or []:
                        inline_data = part.inline_data
                        if inline_data and inline_data.data:
                            if not response_active:
                                response_active = True
                                yield ProviderEvent("response_started", {})
                            yield ProviderEvent("audio", {"pcm16": inline_data.data})

                if content.turn_complete:
                    transcript = input_transcript.strip()
                    if transcript:
                        yield ProviderEvent("user_transcript", {"text": transcript})
                    input_transcript = ""
                    if response_active:
                        logger.info("assistant response completed")
                        yield ProviderEvent("response_completed", {})
                    response_active = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session_context is not None:
            try:
                await asyncio.wait_for(
                    self._session_context.__aexit__(None, None, None), timeout=3
                )
            except (TimeoutError, Exception):
                logger.debug("Gemini session close did not complete cleanly", exc_info=True)
        if self._client is not None:
            try:
                await self._client.aio.aclose()
            except Exception:
                logger.debug("Gemini client close did not complete cleanly", exc_info=True)


def create_provider() -> RealtimeProvider:
    return GeminiLiveProvider()

"""Gemini Flash-Lite confirmation for locally suspected slouching."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from google import genai
from google.genai import types


class GeminiPostureVerifier:
    def __init__(self, client: Any = None) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = os.environ.get("GEMINI_POSTURE_MODEL", "gemini-3.5-flash-lite")
        self._client = client

    async def verify(self, jpeg: bytes) -> dict[str, object]:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._client = genai.Client(api_key=self.api_key)
        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_text(text=(
                        "Inspect this current camera frame of a person seated at a desk. "
                        "Decide whether the person is visibly slouching: rounded or collapsed "
                        "upper back, clearly forward head/neck, dropped shoulders, or a hunched "
                        "seated posture. Do not call ordinary upright sitting, a slight camera "
                        "angle, or briefly looking to the side slouching. If the upper body is "
                        "too unclear, return slouching=false with low confidence."
                    )),
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "slouching": {"type": "boolean"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["slouching", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            ),
            timeout=15,
        )
        data = json.loads(response.text)
        if not isinstance(data.get("slouching"), bool):
            raise ValueError("Gemini posture verdict is missing slouching")
        confidence = float(data.get("confidence", 0))
        return {
            "slouching": data["slouching"] and confidence >= 0.60,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason", ""))[:300],
        }


class GeminiClutterVerifier(GeminiPostureVerifier):
    """Confirm that a sustained local scene change is actually desk clutter."""

    async def verify_clutter(self, jpeg: bytes) -> dict[str, object]:
        if self._client is None:
            if not self.api_key: raise RuntimeError("GEMINI_API_KEY is not set")
            self._client = genai.Client(api_key=self.api_key)
        response = await asyncio.wait_for(self._client.aio.models.generate_content(
            model=self.model,
            contents=[types.Part.from_text(text=("Inspect this desk camera frame. Decide whether the work surface is materially cluttered with misplaced objects, dishes, packaging, loose papers, or accumulated items. Do not count normal work equipment or a person. Return a conservative structured verdict.")), types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema={
                "type":"object","properties":{"cluttered":{"type":"boolean"},"confidence":{"type":"number","minimum":0,"maximum":1},"reason":{"type":"string"}},
                "required":["cluttered","confidence","reason"],"additionalProperties":False},
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))), timeout=15)
        data=json.loads(response.text); confidence=max(0.0,min(1.0,float(data.get("confidence",0))))
        return {"cluttered": bool(data.get("cluttered")), "confidence":confidence, "reason":str(data.get("reason",""))[:300]}

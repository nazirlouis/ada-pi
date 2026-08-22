import unittest
from pathlib import Path

from google.genai import types

from backend.realtime_provider import GeminiLiveProvider


class InterruptedSession:
    def __init__(self, provider: GeminiLiveProvider) -> None:
        self.provider = provider

    async def receive(self):
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                interrupted=True,
                model_turn=types.Content(
                    parts=[types.Part(inline_data=types.Blob(data=b"stale", mime_type="audio/pcm;rate=24000"))]
                ),
                output_transcription=types.Transcription(text="stale transcript"),
            )
        )
        self.provider._closed = True


class ExpressionToolSession:
    def __init__(self, provider: GeminiLiveProvider) -> None:
        self.provider = provider
        self.responses = []

    async def receive(self):
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(
                    id="expression-call-1",
                    name="set_facial_expression",
                    args={"expression": "sassy"},
                )
            ])
        )
        self.provider._closed = True

    async def send_tool_response(self, *, function_responses):
        self.responses.extend(function_responses)


class ProviderEventTests(unittest.IsolatedAsyncioTestCase):
    def test_live_config_guards_long_full_duplex_sessions(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "backend/realtime_provider.py").read_text()
        self.assertIn("START_SENSITIVITY_LOW", source)
        self.assertIn('"prefix_padding_ms": 200', source)
        self.assertIn('"silence_duration_ms": 500', source)
        self.assertIn("ContextWindowCompressionConfig", source)
        self.assertIn("SlidingWindow", source)

    async def test_interruption_discards_coalesced_stale_audio(self) -> None:
        provider = GeminiLiveProvider()
        provider._session = InterruptedSession(provider)
        events = [event async for event in provider.events()]
        self.assertEqual([event.type for event in events], ["response_interrupted"])

    async def test_expression_tool_is_forwarded_and_acknowledged_silently(self) -> None:
        provider = GeminiLiveProvider()
        session = ExpressionToolSession(provider)
        provider._session = session

        events = [event async for event in provider.events()]

        self.assertEqual([(event.type, event.data) for event in events], [
            ("expression", {"name": "sassy"})
        ])
        self.assertEqual(len(session.responses), 1)
        response = session.responses[0]
        self.assertEqual(response.id, "expression-call-1")
        self.assertEqual(response.scheduling, types.FunctionResponseScheduling.SILENT)


if __name__ == "__main__":
    unittest.main()

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


class OfficeStateToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(id="office-state-1", name="get_office_state", args={})
            ])
        )
        self.provider._closed = True


class ProviderEventTests(unittest.IsolatedAsyncioTestCase):
    def test_live_config_guards_long_full_duplex_sessions(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "backend/realtime_provider.py").read_text()
        self.assertIn("START_SENSITIVITY_LOW", source)
        self.assertIn('"prefix_padding_ms": 200', source)
        self.assertIn("ActivityHandling.START_OF_ACTIVITY_INTERRUPTS", source)
        self.assertIn('"silence_duration_ms": 500', source)
        self.assertIn("ContextWindowCompressionConfig", source)
        self.assertIn("SlidingWindow", source)
        self.assertIn("SessionResumptionConfig", source)
        self.assertIn("session_resumption_update", source)
        self.assertIn("message.go_away", source)

    async def test_interruption_discards_coalesced_stale_audio(self) -> None:
        provider = GeminiLiveProvider()
        provider._session = InterruptedSession(provider)
        events = [event async for event in provider.events()]
        self.assertEqual([event.type for event in events], ["response_interrupted"])

    async def test_expression_tool_is_forwarded_and_spoken_reply_can_continue(self) -> None:
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
        self.assertEqual(response.scheduling, types.FunctionResponseScheduling.WHEN_IDLE)

    async def test_office_state_tool_returns_monitor_snapshot(self) -> None:
        state = {"status": "monitoring", "person_state": "not_home", "lights_on": ["light.left_office_light"]}
        provider = GeminiLiveProvider(office_state_getter=lambda: state)
        session = OfficeStateToolSession(provider)
        provider._session = session

        events = [event async for event in provider.events()]

        self.assertEqual(events, [])
        self.assertEqual(session.responses[0].response, {"output": state})
        self.assertEqual(session.responses[0].name, "get_office_state")

    async def test_video_frame_uses_live_video_input(self) -> None:
        class VideoSession:
            def __init__(self) -> None:
                self.video = None

            async def send_realtime_input(self, *, video):
                self.video = video

        provider = GeminiLiveProvider()
        session = VideoSession()
        provider._session = session

        await provider.send_video(b"jpeg-frame")

        self.assertEqual(session.video.data, b"jpeg-frame")
        self.assertEqual(session.video.mime_type, "image/jpeg")

    async def test_habit_alert_sends_text_and_current_image_as_a_turn(self) -> None:
        class AlertSession:
            async def send_client_content(self, **kwargs):
                self.payload = kwargs

        provider = GeminiLiveProvider()
        session = AlertSession()
        provider._session = session
        await provider.send_habit_alert(b"current-frame", "first possible habit")
        self.assertTrue(session.payload["turn_complete"])
        parts = session.payload["turns"].parts
        self.assertEqual(parts[0].text, "first possible habit")
        self.assertEqual(parts[1].inline_data.data, b"current-frame")
        self.assertEqual(parts[1].inline_data.mime_type, "image/jpeg")

    async def test_text_turn_can_trigger_a_boot_greeting(self) -> None:
        class TextSession:
            async def send_client_content(self, **kwargs):
                self.payload = kwargs

        provider = GeminiLiveProvider()
        session = TextSession()
        provider._session = session
        await provider.send_text_turn("Give a short greeting")
        self.assertTrue(session.payload["turn_complete"])
        self.assertEqual(session.payload["turns"].parts[0].text, "Give a short greeting")


if __name__ == "__main__":
    unittest.main()

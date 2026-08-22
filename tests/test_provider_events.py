import unittest

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


class ProviderEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_interruption_discards_coalesced_stale_audio(self) -> None:
        provider = GeminiLiveProvider()
        provider._session = InterruptedSession(provider)
        events = [event async for event in provider.events()]
        self.assertEqual([event.type for event in events], ["response_interrupted"])


if __name__ == "__main__":
    unittest.main()

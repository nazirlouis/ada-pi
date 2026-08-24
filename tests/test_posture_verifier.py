import json
import unittest

from backend.posture_verifier import GeminiPostureVerifier


class FakeResponse:
    text = json.dumps({"slouching": True, "confidence": 0.84, "reason": "Head and shoulders are hunched"})


class FakeModels:
    def __init__(self):
        self.call = None

    async def generate_content(self, **kwargs):
        self.call = kwargs
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.aio = type("Aio", (), {})()
        self.aio.models = FakeModels()


class PostureVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_flash_lite_image_and_structured_verdict(self):
        client = FakeClient()
        verifier = GeminiPostureVerifier(client=client)
        result = await verifier.verify(b"jpeg")
        self.assertTrue(result["slouching"])
        self.assertEqual(result["confidence"], 0.84)
        self.assertEqual(client.aio.models.call["model"], "gemini-3.5-flash-lite")
        config = client.aio.models.call["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIsNotNone(config.response_json_schema)
        self.assertTrue(config.automatic_function_calling.disable)

    async def test_low_confidence_does_not_confirm(self):
        class LowConfidenceModels(FakeModels):
            async def generate_content(self, **kwargs):
                return type("Response", (), {"text": '{"slouching":true,"confidence":0.4,"reason":"unclear"}'})()
        client = FakeClient()
        client.aio.models = LowConfidenceModels()
        result = await GeminiPostureVerifier(client=client).verify(b"jpeg")
        self.assertFalse(result["slouching"])


if __name__ == "__main__":
    unittest.main()

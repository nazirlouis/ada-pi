import os
import unittest
from unittest.mock import AsyncMock

from backend.home_assistant import HomeAssistantClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class HomeAssistantClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_token = os.environ.get("HOME_ASSISTANT_TOKEN")
        os.environ["HOME_ASSISTANT_TOKEN"] = "test-token"

    async def asyncTearDown(self):
        if self.previous_token is None:
            os.environ.pop("HOME_ASSISTANT_TOKEN", None)
        else:
            os.environ["HOME_ASSISTANT_TOKEN"] = self.previous_token

    async def test_entities_exposes_only_simple_power_domains(self):
        client = AsyncMock()
        client.get.return_value = FakeResponse([
            {"entity_id": "light.office", "state": "on", "attributes": {"friendly_name": "Office"}},
            {"entity_id": "switch.monitor", "state": "unavailable", "attributes": {}},
            {"entity_id": "sensor.temperature", "state": "72", "attributes": {}},
        ])
        home = HomeAssistantClient(client=client)
        entities = await home.entities()
        self.assertEqual([item["entity_id"] for item in entities], ["light.office", "switch.monitor"])
        self.assertFalse(entities[1]["available"])

    async def test_entities_deduplicates_same_object_across_domains(self):
        client = AsyncMock()
        client.get.return_value = FakeResponse([
            {"entity_id": "switch.office_lamp", "state": "on", "attributes": {"friendly_name": "Office lamp"}},
            {"entity_id": "light.office_lamp", "state": "on", "attributes": {"friendly_name": "Office lamp"}},
            {"entity_id": "switch.office_lamp_usb", "state": "off", "attributes": {"friendly_name": "Office lamp USB"}},
        ])
        home = HomeAssistantClient(client=client)
        entities = await home.entities()
        self.assertEqual(
            [item["entity_id"] for item in entities],
            ["light.office_lamp", "switch.office_lamp_usb"],
        )

    async def test_power_calls_allow_listed_home_assistant_service(self):
        client = AsyncMock()
        client.post.return_value = FakeResponse([])
        home = HomeAssistantClient(client=client)
        result = await home.set_power("light.office", True)
        client.post.assert_awaited_once_with("/api/services/light/turn_on", json={"entity_id": "light.office"})
        self.assertEqual(result["state"], "on")

    async def test_power_rejects_unsupported_domain(self):
        home = HomeAssistantClient(client=AsyncMock())
        with self.assertRaises(ValueError):
            await home.set_power("lock.front_door", True)


if __name__ == "__main__":
    unittest.main()

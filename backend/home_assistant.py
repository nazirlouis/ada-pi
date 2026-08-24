"""Narrow, read-only Home Assistant state client for ADA monitors."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_OFFICE_LIGHTS = (
    "light.left_office_light",
    "light.right_office_light",
    "light.office_chest_overhead_light",
    "light.office_desk_overhead_light",
    "light.office_desk_rbg_light",
)
CONTROLLABLE_DOMAINS = {"light", "switch", "fan", "input_boolean"}
DOMAIN_PRIORITY = {"light": 0, "fan": 1, "switch": 2, "input_boolean": 3}


@dataclass(slots=True)
class HomeAssistantSnapshot:
    person_state: str
    lights_on: tuple[str, ...]
    light_states: dict[str, str]


class HomeAssistantClient:
    def __init__(self, client: Any = None) -> None:
        self.base_url = os.environ.get("HOME_ASSISTANT_URL", "http://127.0.0.1:8123").rstrip("/")
        self.token = os.environ.get("HOME_ASSISTANT_TOKEN", "").strip()
        self.person_entity = os.environ.get("HOME_ASSISTANT_PERSON", "person.naz").strip()
        configured = os.environ.get("HOME_ASSISTANT_OFFICE_LIGHTS", "")
        self.light_entities = tuple(
            item.strip() for item in configured.split(",") if item.strip()
        ) or DEFAULT_OFFICE_LIGHTS
        self._client = client
        self._owns_client = client is None

    @property
    def configured(self) -> bool:
        return bool(self.token)

    async def snapshot(self) -> HomeAssistantSnapshot:
        if not self.token:
            raise RuntimeError("HOME_ASSISTANT_TOKEN is not set")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=5.0,
            )
        response = await self._client.get("/api/states")
        response.raise_for_status()
        states = {
            str(item.get("entity_id")): str(item.get("state", "unknown"))
            for item in response.json()
            if isinstance(item, dict) and item.get("entity_id")
        }
        light_states = {entity: states.get(entity, "unavailable") for entity in self.light_entities}
        return HomeAssistantSnapshot(
            person_state=states.get(self.person_entity, "unavailable"),
            lights_on=tuple(entity for entity, state in light_states.items() if state == "on"),
            light_states=light_states,
        )

    async def entities(self) -> list[dict[str, Any]]:
        """Return the safe subset of entities that support simple on/off control."""
        states = await self._states()
        entities_by_object_id: dict[str, dict[str, Any]] = {}
        for item in states:
            entity_id = str(item.get("entity_id", ""))
            domain, separator, object_id = entity_id.partition(".")
            if domain not in CONTROLLABLE_DOMAINS:
                continue
            attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
            entity = {
                "entity_id": entity_id,
                "domain": domain,
                "name": str(attributes.get("friendly_name") or entity_id),
                "state": str(item.get("state", "unknown")),
                "available": str(item.get("state", "unknown")) not in {"unknown", "unavailable"},
            }
            # Some integrations expose one physical load as both light.foo and
            # switch.foo. Keep the richer domain while preserving similarly named
            # devices whose actual Home Assistant object IDs differ.
            existing = entities_by_object_id.get(object_id)
            if existing is None or DOMAIN_PRIORITY[domain] < DOMAIN_PRIORITY[existing["domain"]]:
                entities_by_object_id[object_id] = entity
        entities = list(entities_by_object_id.values())
        return sorted(entities, key=lambda entity: (entity["domain"], entity["name"].lower()))

    async def set_power(self, entity_id: str, turn_on: bool) -> dict[str, Any]:
        domain, separator, _ = entity_id.partition(".")
        if not separator or domain not in CONTROLLABLE_DOMAINS:
            raise ValueError("Only lights, switches, fans, and input booleans can be controlled")
        client = await self._http_client()
        service = "turn_on" if turn_on else "turn_off"
        response = await client.post(f"/api/services/{domain}/{service}", json={"entity_id": entity_id})
        response.raise_for_status()
        return {"entity_id": entity_id, "state": "on" if turn_on else "off"}

    async def _http_client(self) -> Any:
        if not self.token:
            raise RuntimeError("HOME_ASSISTANT_TOKEN is not set")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=5.0,
            )
        return self._client

    async def _states(self) -> list[dict[str, Any]]:
        client = await self._http_client()
        response = await client.get("/api/states")
        response.raise_for_status()
        payload = response.json()
        return [item for item in payload if isinstance(item, dict)]

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

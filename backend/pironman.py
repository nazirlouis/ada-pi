"""Small, validated client for the locally running Pironman dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any


SAFE_CONFIG_KEYS = {
    "temperature_unit", "rgb_enable", "rgb_color", "rgb_brightness",
    "rgb_style", "rgb_speed", "oled_enable", "oled_rotation",
    "oled_sleep_timeout", "oled_pages", "gpio_fan_mode", "gpio_fan_led",
}

EXPRESSION_COLORS = {
    "neutral": "#17dfff",
    "sassy": "#ff3dbe",
    "amused": "#35ff9a",
    "skeptical": "#7d8cff",
    "annoyed": "#ff6b35",
    "mad": "#ff2400",
    "concerned": "#4a8fff",
    "surprised": "#ffd43b",
    "mischievous": "#b45cff",
    "serious": "#d9f7ff",
    "alert": "#ff8618",
}


class PironmanError(RuntimeError):
    """Raised when the local Pironman dashboard cannot satisfy a request."""


class PironmanClient:
    def __init__(self, base_url: str | None = None, timeout: float = 2.0) -> None:
        configured = base_url or os.environ.get("PIRONMAN_URL", "http://127.0.0.1:34001")
        self.base_url = configured.rstrip("/")
        self.timeout = timeout

    async def request(
        self, path: str, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        return await asyncio.to_thread(self._request, path, method, payload)

    def _request(
        self, path: str, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise PironmanError(
                f"Pironman API returned HTTP {exc.code}{f': {detail}' if detail else ''}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise PironmanError(f"Pironman dashboard unavailable: {reason}") from exc
        try:
            result = json.loads(content.decode("utf-8")) if content else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PironmanError("Pironman dashboard returned invalid data") from exc
        if isinstance(result, dict) and result.get("status") is False:
            raise PironmanError(str(result.get("error", "Pironman request failed")))
        return result.get("data", result) if isinstance(result, dict) else result

    async def snapshot(self) -> dict[str, Any]:
        data, config = await asyncio.gather(
            self.request("/api/v1.0/get-data"),
            self.request("/api/v1.0/get-config"),
        )
        if isinstance(data, list):
            data = data[-1] if data else {}
        system_config = config.get("system", config) if isinstance(config, dict) else {}
        safe_config = {
            key: value for key, value in system_config.items() if key in SAFE_CONFIG_KEYS
        }
        return {
            "online": True,
            "dashboard_url": f"{self.base_url}/small",
            "data": data if isinstance(data, dict) else {},
            "config": safe_config,
        }

    async def ensure_oled_on(self) -> bool:
        """Keep the OLED enabled with its sleep timer disabled."""
        config = await self.request("/api/v1.0/get-config")
        system = config.get("system", config) if isinstance(config, dict) else {}
        updates: dict[str, Any] = {}
        if system.get("oled_enable") is not True:
            updates["oled_enable"] = True
        if system.get("oled_sleep_timeout") != 0:
            updates["oled_sleep_timeout"] = 0
        if updates:
            await self.update_controls(updates)
            return True
        return False

    async def update_controls(self, controls: dict[str, Any]) -> dict[str, Any]:
        validated = validate_controls(controls)
        routes = {
            "rgb_enable": ("set-rgb-enable", "enable"),
            "rgb_color": ("set-rgb-color", "color"),
            "rgb_brightness": ("set-rgb-brightness", "brightness"),
            "rgb_style": ("set-rgb-style", "style"),
            "rgb_speed": ("set-rgb-speed", "speed"),
            "oled_enable": ("set-oled-enable", "enable"),
            "oled_rotation": ("set-oled-rotation", "rotation"),
            "oled_sleep_timeout": ("set-oled-sleep-timeout", "timeout"),
            "temperature_unit": ("set-temperature-unit", "unit"),
            "gpio_fan_mode": ("set-fan-mode", "fan_mode"),
            "gpio_fan_led": ("set-fan-led", "led"),
        }
        results = {}
        for key, value in validated.items():
            route, parameter = routes[key]
            results[key] = await self.request(
                f"/api/v1.0/{route}", "POST", {parameter: value}
            )
        return {"updated": validated, "result": results}

    async def set_expression_lighting(self, expression: str) -> None:
        """Match the case LEDs to Ada's active expression when it has a palette entry."""
        color = EXPRESSION_COLORS.get(expression)
        if color is not None:
            # Expression changes are latency-sensitive. Lighting is enabled by
            # the user's normal case configuration, so one color request is
            # sufficient and avoids a second config write on every expression.
            await self.update_controls({"rgb_color": color})


def validate_controls(controls: dict[str, Any]) -> dict[str, Any]:
    """Accept only reversible display/lighting controls with strict value bounds."""
    if not isinstance(controls, dict) or not controls:
        raise ValueError("At least one control is required")
    allowed = {
        "rgb_enable", "rgb_color", "rgb_brightness", "rgb_style", "rgb_speed",
        "oled_enable", "oled_rotation", "oled_sleep_timeout", "temperature_unit",
        "gpio_fan_mode", "gpio_fan_led",
    }
    unknown = set(controls) - allowed
    if unknown:
        raise ValueError(f"Unsupported control: {sorted(unknown)[0]}")

    result: dict[str, Any] = {}
    for key, value in controls.items():
        if key in {"rgb_enable", "oled_enable"}:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif key in {"rgb_brightness", "rgb_speed"}:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError(f"{key} must be an integer from 0 to 100")
        elif key == "rgb_color":
            if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
                raise ValueError("rgb_color must use #RRGGBB format")
            try:
                int(value[1:], 16)
            except ValueError as exc:
                raise ValueError("rgb_color must use #RRGGBB format") from exc
            value = value.lower()
        elif key == "rgb_style":
            if not isinstance(value, str) or not value or len(value) > 40:
                raise ValueError("rgb_style is invalid")
            if not all(char.isalnum() or char in "_- " for char in value):
                raise ValueError("rgb_style is invalid")
        elif key == "oled_rotation":
            if isinstance(value, bool) or value not in {0, 180}:
                raise ValueError("oled_rotation must be 0 or 180")
        elif key == "oled_sleep_timeout":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 3600:
                raise ValueError("oled_sleep_timeout must be from 0 to 3600 seconds")
        elif key == "temperature_unit":
            if value not in {"C", "F"}:
                raise ValueError("temperature_unit must be C or F")
        elif key == "gpio_fan_mode":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError("gpio_fan_mode must be from 0 to 4")
        elif key == "gpio_fan_led":
            if value not in {"on", "off", "follow"}:
                raise ValueError("gpio_fan_led must be on, off, or follow")
        result[key] = value
    return result

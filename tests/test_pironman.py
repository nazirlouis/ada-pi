import unittest
from unittest.mock import AsyncMock

from backend.pironman import EXPRESSION_COLORS, PironmanClient, validate_controls


class PironmanControlTests(unittest.TestCase):
    def test_accepts_safe_controls(self) -> None:
        self.assertEqual(validate_controls({
            "rgb_enable": True,
            "rgb_color": "#17DFFF",
            "rgb_brightness": 55,
            "rgb_style": "breathing",
            "rgb_speed": 20,
            "oled_enable": False,
        }), {
            "rgb_enable": True,
            "rgb_color": "#17dfff",
            "rgb_brightness": 55,
            "rgb_style": "breathing",
            "rgb_speed": 20,
            "oled_enable": False,
        })

    def test_rejects_unknown_or_invalid_controls(self) -> None:
        invalid = [
            {}, {"shutdown": True}, {"rgb_brightness": 101},
            {"rgb_brightness": True}, {"rgb_color": "red"},
            {"oled_enable": 1}, {"rgb_style": "x<script>"},
            {"reboot": True}, {"oled_rotation": 90},
            {"gpio_fan_mode": 5}, {"gpio_fan_led": "blink"},
        ]
        for controls in invalid:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                validate_controls(controls)


class PironmanClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_uses_versioned_dashboard_api(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(side_effect=[
            [{"cpu_temperature": 42}],
            {"system": {"rgb_enable": True, "smtp_password": "secret"}},
        ])
        snapshot = await client.snapshot()
        self.assertEqual(snapshot["data"], {"cpu_temperature": 42})
        self.assertEqual(snapshot["config"], {"rgb_enable": True})
        self.assertEqual(client.request.await_args_list[0].args, ("/api/v1.0/get-data",))
        self.assertEqual(client.request.await_args_list[1].args, ("/api/v1.0/get-config",))

    async def test_controls_use_individual_dashboard_routes(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value="OK")
        result = await client.update_controls({"rgb_brightness": 60, "oled_enable": True})
        self.assertEqual(result["updated"], {"rgb_brightness": 60, "oled_enable": True})
        self.assertEqual(client.request.await_args_list[0].args, (
            "/api/v1.0/set-rgb-brightness", "POST", {"brightness": 60}
        ))
        self.assertEqual(client.request.await_args_list[1].args, (
            "/api/v1.0/set-oled-enable", "POST", {"enable": True}
        ))

    async def test_safe_fan_control_uses_versioned_route(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value="OK")
        await client.update_controls({"gpio_fan_mode": 4})
        self.assertEqual(client.request.await_args.args, (
            "/api/v1.0/set-fan-mode", "POST", {"fan_mode": 4}
        ))

    async def test_mad_expression_sets_case_rgb_red(self) -> None:
        client = PironmanClient()
        client.update_controls = AsyncMock(return_value={})
        await client.set_expression_lighting("mad")
        client.update_controls.assert_awaited_once_with({
            "rgb_color": "#ff2400"
        })

    async def test_unknown_expression_does_not_change_lighting(self) -> None:
        client = PironmanClient()
        client.update_controls = AsyncMock(return_value={})
        await client.set_expression_lighting("not-an-expression")
        client.update_controls.assert_not_awaited()

    async def test_oled_guard_corrects_disabled_and_sleeping_config(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value={
            "system": {"oled_enable": False, "oled_sleep_timeout": 10}
        })
        client.update_controls = AsyncMock(return_value={})
        changed = await client.ensure_oled_on()
        self.assertTrue(changed)
        client.update_controls.assert_awaited_once_with({
            "oled_enable": True, "oled_sleep_timeout": 0
        })

    async def test_oled_guard_does_not_rewrite_correct_config(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value={
            "system": {"oled_enable": True, "oled_sleep_timeout": 0}
        })
        client.update_controls = AsyncMock(return_value={})
        self.assertFalse(await client.ensure_oled_on())
        client.update_controls.assert_not_awaited()

    async def test_fan_guard_sets_always_on_mode(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value={
            "system": {"gpio_fan_mode": 3}
        })
        client.update_controls = AsyncMock(return_value={})
        self.assertTrue(await client.ensure_fans_max())
        client.update_controls.assert_awaited_once_with({"gpio_fan_mode": 0})

    async def test_fan_guard_does_not_rewrite_always_on_mode(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value={
            "system": {"gpio_fan_mode": 0}
        })
        client.update_controls = AsyncMock(return_value={})
        self.assertFalse(await client.ensure_fans_max())
        client.update_controls.assert_not_awaited()

    async def test_fan_guard_ignores_variants_without_gpio_fans(self) -> None:
        client = PironmanClient()
        client.request = AsyncMock(return_value={"system": {}})
        client.update_controls = AsyncMock(return_value={})
        self.assertFalse(await client.ensure_fans_max())
        client.update_controls.assert_not_awaited()

    def test_every_ada_expression_has_a_case_color(self) -> None:
        self.assertEqual(set(EXPRESSION_COLORS), {
            "neutral", "sassy", "amused", "skeptical", "annoyed", "mad",
            "concerned", "surprised", "mischievous", "serious", "alert",
        })


if __name__ == "__main__":
    unittest.main()

import unittest
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "frontend" / "assets" / "ada"


class FrontendContractTests(unittest.TestCase):
    def test_layered_visual_controller_and_hidden_kiosk_controls_exist(self) -> None:
        html = (ROOT / "frontend/index.html").read_text()
        self.assertIn('id="ada-stage"', html)
        self.assertIn('id="ada-rig"', html)
        self.assertIn('class="eye-window left"', html)
        self.assertIn('class="blink-layer"', html)
        self.assertIn('src="/static/ada-visual.js"', html)
        self.assertIn('id="dev-panel" hidden', html)
        self.assertNotIn('id="particle-field"', html)

    def test_rig_assets_are_display_sized_and_layers_have_alpha(self) -> None:
        expected = {
            "base.png",
            "blink.png",
            "mouth-small.png",
            "mouth-medium.png",
            "mouth-wide.png",
            "mouth-round.png",
        }
        self.assertEqual({path.name for path in ASSETS.glob("*.png")}, expected)
        for name in expected:
            payload = (ASSETS / name).read_bytes()[:26]
            self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", payload[16:24]), (800, 480))
            if name != "base.png":
                self.assertIn(payload[25], (4, 6), "overlay PNG must include alpha")

    def test_audio_pipeline_keeps_one_microphone_and_adds_output_analyser(self) -> None:
        app = (ROOT / "frontend/app.js").read_text()
        self.assertEqual(app.count("getUserMedia("), 1)
        self.assertIn("playbackContext.createAnalyser()", app)
        self.assertIn("playbackNode.connect(playbackAnalyser)", app)
        self.assertIn("playbackAnalyser.connect(playbackContext.destination)", app)
        self.assertIn("playbackAnalyser.getByteFrequencyData(spectrum)", app)
        self.assertIn("setSpeechFeatures(level, brightness)", app)

    def test_public_visual_control_surface(self) -> None:
        visual = (ROOT / "frontend/ada-visual.js").read_text()
        for api in (
            "setAdaState",
            "setGaze",
            "setExpression",
            "setSpeechLevel",
            "setSpeechFeatures",
        ):
            self.assertIn(f"window.{api}", visual)


if __name__ == "__main__":
    unittest.main()

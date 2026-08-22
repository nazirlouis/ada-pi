import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_visual_controller_and_hidden_kiosk_controls_exist(self) -> None:
        html = (ROOT / "frontend/index.html").read_text()
        self.assertIn('id="ada-stage"', html)
        self.assertIn('id="particle-field"', html)
        self.assertIn('src="/static/ada-visual.js"', html)
        self.assertIn('id="dev-panel" hidden', html)

    def test_audio_pipeline_keeps_one_microphone_and_adds_output_analyser(self) -> None:
        app = (ROOT / "frontend/app.js").read_text()
        self.assertEqual(app.count("getUserMedia("), 1)
        self.assertIn("playbackContext.createAnalyser()", app)
        self.assertIn("playbackNode.connect(playbackAnalyser)", app)
        self.assertIn("playbackAnalyser.connect(playbackContext.destination)", app)

    def test_public_visual_control_surface(self) -> None:
        visual = (ROOT / "frontend/ada-visual.js").read_text()
        for api in ("setAdaState", "setGaze", "setExpression", "setSpeechLevel"):
            self.assertIn(f"window.{api}", visual)


if __name__ == "__main__":
    unittest.main()

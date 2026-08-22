import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_plain_chat_controls_and_transcript_exist(self) -> None:
        html = (ROOT / "frontend/index.html").read_text()
        self.assertIn('id="connect"', html)
        self.assertIn('id="disconnect"', html)
        self.assertIn('id="connection-status"', html)
        self.assertIn('id="microphone-status"', html)
        self.assertIn('id="log"', html)
        self.assertNotIn("<img", html)
        self.assertNotIn("ada-visual.js", html)

    def test_audio_pipeline_keeps_one_microphone_and_direct_playback(self) -> None:
        app = (ROOT / "frontend/app.js").read_text()
        self.assertEqual(app.count("getUserMedia("), 1)
        self.assertIn("playbackNode.connect(playbackContext.destination)", app)
        self.assertNotIn("createAnalyser", app)
        self.assertNotIn("setAdaState", app)

    def test_no_frontend_image_assets_remain(self) -> None:
        assets = ROOT / "frontend/assets"
        self.assertFalse(assets.exists() and any(assets.rglob("*.*")))


if __name__ == "__main__":
    unittest.main()

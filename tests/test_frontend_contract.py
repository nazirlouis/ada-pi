import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_idle_face_and_hidden_chat_contract_exist(self) -> None:
        html = (ROOT / "frontend/index.html").read_text()
        self.assertIn('id="face-stage"', html)
        self.assertIn('id="idle-face"', html)
        self.assertIn('viewBox="140 84 1320 792"', html)
        self.assertEqual(html.count('class="eye-panel"'), 2)
        self.assertEqual(html.count('expression-brow '), 2)
        self.assertIn('class="mouth"', html)
        self.assertNotIn('class="connection-scan"', html)
        self.assertNotIn('class="mouth-opening"', html)
        self.assertNotIn('class="upper-lid"', html)
        self.assertNotIn('id="eye-fill"', html)
        self.assertNotIn('id="eye-fill-mirror"', html)
        self.assertIn('src="/static/idle-face.js?', html)
        self.assertIn('id="connect"', html)
        self.assertIn('id="disconnect"', html)
        self.assertIn('id="exit"', html)
        self.assertIn('id="expression-controls"', html)
        self.assertIn('id="connection-status"', html)
        self.assertIn('id="microphone-status"', html)
        self.assertIn('id="log"', html)
        self.assertNotIn("<img", html)
        self.assertNotIn("ada-visual.js", html)

    def test_idle_animation_is_visual_only_and_clamped(self) -> None:
        animation = (ROOT / "frontend/idle-face.js").read_text()
        styles = (ROOT / "frontend/style.css").read_text()
        self.assertIn("--gaze-x", animation)
        self.assertIn("--gaze-y", animation)
        self.assertIn("--blink", animation)
        self.assertIn("clamp(x, 18)", animation)
        self.assertIn("scheduleSaccade", animation)
        self.assertIn('"--blink-left"', animation)
        self.assertIn('"--blink-right"', animation)
        self.assertIn('"--brow-left-lift"', animation)
        self.assertIn('"--brow-right-lift"', animation)
        self.assertIn("blinkOnce(scheduleBlink, 1.3)", animation)
        self.assertIn("randomBetween(4500, 7000)", animation)
        self.assertIn(".mouth-track", styles)
        self.assertIn("var(--gaze-x) * .55", styles)
        self.assertIn(".eye-panel,.mouth", styles)
        self.assertIn("@keyframes mouth-aura", styles)
        self.assertIn("@keyframes face-breathing", styles)
        self.assertIn("@keyframes edge-shimmer", styles)
        self.assertNotIn("@keyframes connect-scan", styles)
        self.assertIn("setSpeechLevel", animation)
        self.assertIn("setConnecting", animation)
        self.assertIn("setExpression", animation)
        self.assertIn("expressionBrowPaths", animation)
        for expression in (
            "neutral", "sassy", "amused", "skeptical", "annoyed", "mad",
            "concerned", "surprised", "mischievous", "serious", "alert",
        ):
            self.assertIn(f'"{expression}"', animation)
            self.assertIn(f".expression-{expression}", styles)
        self.assertIn('round:', animation)
        self.assertIn('wide:', animation)
        self.assertIn('path.setAttribute("d"', animation)
        self.assertNotIn("getUserMedia", animation)
        self.assertNotIn("AudioContext", animation)

    def test_audio_pipeline_keeps_one_microphone_and_passive_output_meter(self) -> None:
        app = (ROOT / "frontend/app.js").read_text()
        self.assertEqual(app.count("getUserMedia("), 1)
        self.assertIn("playbackContext.createAnalyser()", app)
        self.assertIn("playbackNode.connect(playbackAnalyser)", app)
        self.assertIn("playbackAnalyser.connect(playbackContext.destination)", app)
        self.assertIn("getFloatTimeDomainData", app)
        self.assertIn("setSpeechLevel(smoothed)", app)
        self.assertIn("setSpeechLevel(0, true)", app)
        self.assertIn("this.startThreshold = 1440", app)
        self.assertIn("bufferedSamples", app)
        self.assertIn('type: "flush"', app)
        self.assertIn('fetch("/shutdown"', app)
        self.assertIn('case "expression"', app)
        self.assertNotIn("setAdaState", app)

    def test_no_frontend_image_assets_remain(self) -> None:
        assets = ROOT / "frontend/assets"
        self.assertFalse(assets.exists() and any(assets.rglob("*.*")))


if __name__ == "__main__":
    unittest.main()

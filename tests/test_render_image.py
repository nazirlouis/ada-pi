"""Unit tests for backend/tools.d/ada_render_image.py.

gemini_pool.generate and traffic_camera.publish_relay are faked — no
network, no real image model call. Covers prompt validation, aspect
mapping, ref_url shapes, the render->save->publish flow, and manifest
wiring.
"""

from __future__ import annotations

import asyncio
import base64
import os
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import google.genai  # noqa: F401
except ImportError:
    import sys as _sys
    import types as _types
    _genai = _types.ModuleType("google.genai")
    _genai.types = _types.ModuleType("google.genai.types")
    _sys.modules.setdefault("google.genai", _genai)
    _sys.modules.setdefault("google.genai.types", _genai.types)

TOOL_PATH = (Path(__file__).resolve().parent.parent
             / "backend" / "tools.d" / "ada_render_image.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "tools_d.ada_render_image_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


render = _load_module()
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 600
JPEG_BYTES = b"\xff\xd8jpeg-fake\xff\xd9"


def _resp_with_image(data: bytes = PNG_BYTES, text: str = ""):
    """Fake generate_content response: one candidate, inline image."""
    part = MagicMock()
    part.inline_data = MagicMock()
    part.inline_data.data = data
    part.inline_data.mime_type = "image/png"
    part.text = None
    cand = MagicMock()
    cand.content.parts = [part]
    resp = MagicMock()
    resp.candidates = [cand]
    return resp


class TestAdaRenderImage(unittest.TestCase):

    def _run(self, **kw):
        return asyncio.run(render.run(MagicMock(), **kw))

    def test_no_prompt_is_honest(self):
        out = self._run()
        self.assertFalse(out["ok"])
        self.assertIn("prompt", out["error"])

    def test_aspect_aliases(self):
        for raw, want in [("", "1:1"), ("square", "1:1"),
                          ("landscape", "16:9"), ("portrait", "9:16"),
                          ("16x9", "16:9"), ("3:4", "3:4")]:
            got, err = render._aspect(raw)
            self.assertIsNone(err, raw)
            self.assertEqual(got, want)
        bad, err = render._aspect("cinematic")
        self.assertIsNone(bad)
        self.assertIn("unknown aspect", err)

    def test_happy_path_publishes_image_url(self):
        import backend.gemini_pool as gp
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", {"ADA_RENDER_DIR": td}), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()), \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:abc"):
            out = self._run(prompt="a cat in a hat", aspect="landscape")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["image_url"], "https://x/frame?token=render:abc")
        self.assertEqual(out["aspect"], "16:9")
        self.assertTrue(Path(out["file"]).name.startswith("render-img-"))

    def test_ref_url_refuses_bad_shape(self):
        out = self._run(prompt="edit it", ref_url="ftp://nope")
        self.assertFalse(out["ok"])
        self.assertIn("ref_url", out["error"])

    def test_ref_url_http_fetch_and_edit(self):
        import httpx
        fake = httpx.Response(200, content=JPEG_BYTES,
                              headers={"content-type": "image/jpeg"})
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", {"ADA_RENDER_DIR": td,
                                      "ADA_RENDER_IMAGE_SELFCHECK": "0"}), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=fake), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:def"):
            import backend.gemini_pool as gp
            with patch.object(gp, "generate", new_callable=AsyncMock,
                              return_value=_resp_with_image()) as gen, \
                 patch.object(gp, "configured_keys",
                              return_value=["k1"]):
                out = self._run(prompt="make it red",
                                ref_url="https://x/in.jpg")
        self.assertTrue(out["ok"], out)
        # the ref went into the request as inline_data — call_args_list[0]
        # is the render call (selfcheck disabled via env)
        contents = gen.call_args_list[0][0][1]
        parts = contents[0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1]["inline_data"]["mime_type"],
                         "image/jpeg")

    def test_no_image_in_response_is_honest(self):
        resp = MagicMock()
        resp.candidates = []
        resp.prompt_feedback = None
        import backend.gemini_pool as gp
        with patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=resp), \
             patch.object(gp, "configured_keys", return_value=["k1"]):
            out = self._run(prompt="nothing")
        self.assertFalse(out["ok"])
        self.assertIn("no image", out["error"])

    def test_no_image_response_tries_explicit_image_fallbacks(self):
        import backend.gemini_pool as gp
        resp = MagicMock()
        resp.candidates = []
        resp.prompt_feedback = None
        with patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=resp) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]):
            out = self._run(prompt="fallback smoke")
        self.assertFalse(out["ok"])
        self.assertEqual(
            [call.args[0] for call in gen.await_args_list],
            [
                "gemini-3.1-flash-image",
                "gemini-3.1-flash-lite-image",
                "or-gemini-3.1-flash-image",
            ],
        )

    def test_size_param_reaches_image_config(self):
        with tempfile.TemporaryDirectory() as td, \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:ghi"), \
             patch.dict("os.environ", {"ADA_RENDER_IMAGE_SELFCHECK": "0",
                                      "LINE_MEDIA_ARCHIVE_DIR": td,
                                      "ADA_RENDER_DIR": td}):
            import backend.gemini_pool as gp
            with patch.object(gp, "generate", new_callable=AsyncMock,
                              return_value=_resp_with_image()) as gen,                  patch.object(gp, "configured_keys",
                              return_value=["k1"]):
                out = self._run(prompt="big", size="4K")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["size"], "4K")
        # call_args_list[0] is the render call — selfcheck adds a second
        # generate() call when enabled
        cfg = gen.call_args_list[0][1].get("config") \
            or gen.call_args_list[0][0][2]
        icfg = getattr(cfg, "image_config", None) or cfg["image_config"]
        val = getattr(icfg, "image_size", None) or icfg["image_size"]
        self.assertEqual(str(val).upper(), "4K")

    def test_gate_file_refuses(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ada_render_image.closed").touch()
            with patch.dict(os.environ, {"ADA_GATES_DIR": d}):
                out = self._run(prompt="anything")
        self.assertFalse(out["ok"])
        self.assertIn("switched off", out["error"])

    def test_lane_gate_only_blocks_that_lane(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ada_render_image.kk.closed").touch()
            env = {"ADA_GATES_DIR": d, "ADA_INSTANCE_ID": "kk"}
            with patch.dict(os.environ, env):
                out = self._run(prompt="anything")
            self.assertFalse(out["ok"])
            env2 = {"ADA_GATES_DIR": d, "ADA_INSTANCE_ID": "chaba"}
            with patch.dict(os.environ, env2), \
                 patch("backend.traffic_camera.publish_relay",
                       return_value="https://x/frame?token=render:jkl"):
                import backend.gemini_pool as gp
                with patch.object(gp, "generate", new_callable=AsyncMock,
                                  return_value=_resp_with_image()),                      patch.object(gp, "configured_keys",
                                  return_value=["k1"]):
                    out2 = self._run(prompt="anything")
            self.assertTrue(out2["ok"], out2)

    def test_manifest_entry(self):
        import yaml
        m = yaml.safe_load(open(
            Path(__file__).resolve().parent.parent
            / "backend" / "tools.d" / "manifest.yml"))
        ent = m["tools"]["ada_render_image"]
        self.assertEqual(ent["module"], "ada_render_image")
        self.assertEqual(ent["policy"], "read")


def _sd_job_payload(status="completed", job="job_test_1"):
    """sd-server /sdcpp/v1/jobs/<id> payload."""
    out = {"id": job, "kind": "img_gen", "status": status,
           "queue_position": 0, "result": None, "error": None}
    if status == "completed":
        out["result"] = {
            "images": [{"b64_json": base64.b64encode(PNG_BYTES).decode(),
                        "index": 0}],
            "output_format": "png"}
    if status == "failed":
        out["error"] = {"message": "out of memory"}
    return out


class TestAdaRenderImageLocal(unittest.TestCase):
    """provider='local' — sd-server sdcpp async job path."""

    def _run(self, **kw):
        env = {"ADA_SD_SERVER_URL": "http://sd.test:8399"}
        with patch.dict(os.environ, env):
            return asyncio.run(render.run(MagicMock(), **kw))

    def _sd_env(self, td, wait="0.01"):
        return {"ADA_SD_SERVER_URL": "http://sd.test:8399",
                "ADA_RENDER_DIR": td,
                "ADA_RENDER_LOCAL_WAIT_S": wait,
                "ADA_RENDER_LOCAL_POLL_S": "0.01"}

    def test_unknown_provider_is_honest(self):
        out = self._run(prompt="a cat", provider="dalle")
        self.assertFalse(out["ok"])
        self.assertIn("unknown provider", out["error"])

    def test_provider_aliases(self):
        self.assertEqual(render._provider("omen"), ("local", None))
        self.assertEqual(render._provider(""), ("gemini", None))

    def test_local_ref_edit_refused(self):
        out = self._run(prompt="make it red", provider="local",
                        ref_url="https://x/in.jpg")
        self.assertFalse(out["ok"])
        self.assertIn("text-to-image", out["error"])

    def test_job_param_needs_local(self):
        out = self._run(prompt="a cat", job="job_abc")
        self.assertFalse(out["ok"])
        self.assertIn("provider='local'", out["error"])
        out = self._run(provider="local", job="not a job")
        self.assertFalse(out["ok"])
        self.assertIn("job_...", out["error"])

    def test_local_submit_completed_fast_path(self):
        import httpx
        post = httpx.Response(200, json={"id": "job_test_1",
                                         "status": "queued",
                                         "queue_position": 0})
        get = httpx.Response(200, json=_sd_job_payload())
        with tempfile.TemporaryDirectory() as td, \
             patch.dict(os.environ, self._sd_env(td)), \
             patch.object(httpx.AsyncClient, "post",
                          new_callable=AsyncMock, return_value=post) as p, \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=get), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:loc"):
            out = self._run(prompt="a paper crane", provider="local",
                            aspect="landscape")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["provider"], "local")
        self.assertEqual(out["job"], "job_test_1")
        self.assertEqual(out["image_url"],
                         "https://x/frame?token=render:loc")
        body = p.call_args[1]["json"]
        self.assertEqual((body["width"], body["height"]), (1344, 768))
        self.assertEqual(body["sample_params"]["sample_steps"], 4)

    def test_local_pending_arms_delivery(self):
        import httpx
        post = httpx.Response(200, json={"id": "job_pend_1",
                                         "status": "queued",
                                         "queue_position": 2})
        get = httpx.Response(200, json={**_sd_job_payload(
            status="generating", job="job_pend_1"), "queue_position": 2})
        with tempfile.TemporaryDirectory() as td, \
             patch.dict(os.environ, self._sd_env(td)), \
             patch.object(httpx.AsyncClient, "post",
                          new_callable=AsyncMock, return_value=post), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=get), \
             patch.object(render, "_arm_local_delivery") as arm:
            out = self._run(prompt="slow render", provider="local")
        self.assertFalse(out["ok"])
        self.assertTrue(out["pending"])
        self.assertEqual(out["job"], "job_pend_1")
        self.assertEqual(out["queue_position"], 2)
        self.assertIn("chat_send", out["error"])
        arm.assert_called_once()

    def test_local_resume_completed(self):
        import httpx
        get = httpx.Response(200, json=_sd_job_payload(job="job_re_9"))
        with tempfile.TemporaryDirectory() as td, \
             patch.dict(os.environ, self._sd_env(td)), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=get), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:re"):
            out = self._run(provider="local", job="job_re_9")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["job"], "job_re_9")

    def test_local_failed_job_is_honest(self):
        import httpx
        post = httpx.Response(200, json={"id": "job_fail_1",
                                         "status": "queued"})
        get = httpx.Response(200, json=_sd_job_payload(
            status="failed", job="job_fail_1"))
        with tempfile.TemporaryDirectory() as td, \
             patch.dict(os.environ, self._sd_env(td)), \
             patch.object(httpx.AsyncClient, "post",
                          new_callable=AsyncMock, return_value=post), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=get):
            out = self._run(prompt="boom", provider="local")
        self.assertFalse(out["ok"])
        self.assertIn("failed", out["error"])
        self.assertIn("out of memory", out["error"])

    def test_local_server_down_is_honest(self):
        import httpx
        with tempfile.TemporaryDirectory() as td, \
             patch.dict(os.environ, self._sd_env(td)), \
             patch.object(httpx.AsyncClient, "post",
                          new_callable=AsyncMock,
                          side_effect=httpx.ConnectError("refused")):
            out = self._run(prompt="a cat", provider="local")
        self.assertFalse(out["ok"])
        self.assertIn("didn't answer", out["error"])

    def test_sd_result_bytes_shapes(self):
        data, err = render._sd_result_bytes(_sd_job_payload())
        self.assertEqual(data, PNG_BYTES)
        self.assertIsNone(err)
        data, err = render._sd_result_bytes({"status": "completed",
                                             "result": {}})
        self.assertIsNone(data)
        self.assertIn("no image", err)

    def test_deliver_local_pushes_photo(self):
        import httpx
        runner = MagicMock()
        runner.chat_send = AsyncMock(return_value={"status": "queued"})
        runner.session_id = "sess-x"
        get = httpx.Response(200, json=_sd_job_payload(job="job_dl_1"))
        env = {"ADA_SD_SERVER_URL": "http://sd.test:8399",
               "ADA_RENDER_LOCAL_POLL_S": "0.01"}
        with tempfile.TemporaryDirectory() as td, \
             patch.dict(os.environ, {**env, "ADA_RENDER_DIR": td}), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=get), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:dl"):
            asyncio.run(render._deliver_local(runner, "job_dl_1",
                                              "a crane"))
        runner.chat_send.assert_awaited_once()
        kw = runner.chat_send.await_args[1]
        self.assertEqual(kw["image_url"],
                         "https://x/frame?token=render:dl")
        self.assertIn("done", kw["text"])

    def test_deliver_local_failure_pushes_text(self):
        import httpx
        runner = MagicMock()
        runner.chat_send = AsyncMock(return_value={"status": "queued"})
        runner.session_id = "sess-x"
        get = httpx.Response(200, json=_sd_job_payload(
            status="failed", job="job_dl_2"))
        env = {"ADA_SD_SERVER_URL": "http://sd.test:8399",
               "ADA_RENDER_LOCAL_POLL_S": "0.01"}
        with patch.dict(os.environ, env), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=get):
            asyncio.run(render._deliver_local(runner, "job_dl_2", "x"))
        kw = runner.chat_send.await_args[1]
        self.assertEqual(kw["image_url"], "")
        self.assertIn("failed", kw["text"])


def _resp_with_text(text: str):
    """Fake generate_content response: one candidate, text only."""
    part = MagicMock()
    part.inline_data = None
    part.text = text
    cand = MagicMock()
    cand.content.parts = [part]
    resp = MagicMock()
    resp.candidates = [cand]
    return resp


def _runner(sid: str = "sess-t1"):
    """A runner whose context/session resolve a real session id."""
    r = MagicMock()
    r.context = None
    r.session_id = sid
    return r


class TestRenderFlowQuality(unittest.TestCase):
    """render-flow-quality-pass: iterate-on-last, multi-ref, vision
    self-check, transient retry, aspect notices, arch full_url."""

    def setUp(self):
        from backend import render_state
        render_state.clear()

    def tearDown(self):
        from backend import render_state
        render_state.clear()

    def _gemini_env(self, td):
        return {"ADA_RENDER_DIR": td, "LINE_MEDIA_ARCHIVE_DIR": td,
                "ADA_RENDER_IMAGE_SELFCHECK": "0"}

    def _patch_stack(self, td, gen):
        import backend.gemini_pool as gp
        return patch.dict("os.environ", self._gemini_env(td)), \
            patch.object(gp, "generate", gen), \
            patch.object(gp, "configured_keys", return_value=["k1"]), \
            patch("backend.traffic_camera.publish_relay",
                  return_value="https://x/frame?token=render:q")

    def test_last_ref_feeds_previous_render(self):
        import backend.gemini_pool as gp
        runner = _runner()
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td)), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:a"):
            out1 = asyncio.run(render.run(runner, prompt="a red barn"))
            self.assertTrue(out1["ok"], out1)
            out2 = asyncio.run(render.run(
                runner, prompt="warmer golden light", ref_url="last"))
        self.assertTrue(out2["ok"], out2)
        # the second call carried the remembered artifact as inline_data
        parts = gen.call_args_list[-1][0][1][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")

    def test_last_ref_without_prior_render_is_honest(self):
        out = asyncio.run(render.run(
            _runner(), prompt="make it warmer", ref_url="last"))
        self.assertFalse(out["ok"])
        self.assertIn("nothing rendered", out["error"])

    def test_auto_ref_on_edit_cue(self):
        import backend.gemini_pool as gp
        runner = _runner("sess-auto")
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td)), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:b"):
            asyncio.run(render.run(runner, prompt="a red barn"))
            out = asyncio.run(render.run(runner, prompt="make it warmer"))
        self.assertTrue(out["ok"], out)
        self.assertEqual(out.get("auto_ref"), "last")
        parts = gen.call_args_list[-1][0][1][0]["parts"]
        self.assertEqual(len(parts), 2)

    def test_fresh_cue_suppresses_auto_ref(self):
        import backend.gemini_pool as gp
        runner = _runner("sess-fresh")
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td)), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:c"):
            asyncio.run(render.run(runner, prompt="a red barn"))
            out = asyncio.run(render.run(
                runner, prompt="a new mountain scene instead"))
        self.assertTrue(out["ok"], out)
        self.assertNotIn("auto_ref", out)
        parts = gen.call_args_list[-1][0][1][0]["parts"]
        self.assertEqual(len(parts), 1)

    def test_phrase_cue_change_the_text(self):
        """Card's canonical ask: 'change the text' auto-feeds the last
        render — but 'a poster with the text SALE' stays a fresh gen."""
        import backend.gemini_pool as gp
        runner = _runner("sess-text")
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td)), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:t"):
            asyncio.run(render.run(runner, prompt="a barn poster"))
            out = asyncio.run(render.run(
                runner, prompt="change the text to say HELLO"))
            parts = gen.call_args_list[-1][0][1][0]["parts"]
            self.assertEqual(out.get("auto_ref"), "last")
            self.assertEqual(len(parts), 2)
            out2 = asyncio.run(render.run(
                runner, prompt="a poster with the text SALE"))
            parts2 = gen.call_args_list[-1][0][1][0]["parts"]
            self.assertNotIn("auto_ref", out2)
            self.assertEqual(len(parts2), 1)

    def test_ref_urls_multi_ref(self):
        import httpx
        import backend.gemini_pool as gp
        fake = httpx.Response(200, content=JPEG_BYTES,
                              headers={"content-type": "image/jpeg"})
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td)), \
             patch.object(httpx.AsyncClient, "get",
                          new_callable=AsyncMock, return_value=fake), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:d"):
            out = asyncio.run(render.run(
                _runner(), prompt="combine these two photos",
                ref_urls=["https://x/a.jpg", "https://x/b.jpg"]))
        self.assertTrue(out["ok"], out)
        parts = gen.call_args_list[-1][0][1][0]["parts"]
        self.assertEqual(len(parts), 3)

    def test_ref_urls_over_cap_is_honest(self):
        out = asyncio.run(render.run(
            _runner(), prompt="combine", ref_url="https://x/a.jpg",
            ref_urls=["https://x/b.jpg", "https://x/c.jpg",
                      "https://x/d.jpg"]))
        self.assertFalse(out["ok"])
        self.assertIn("3", out["error"])

    def test_selfcheck_miss_retries_once(self):
        import backend.gemini_pool as gp
        calls = []

        async def _gen(model, contents, **kw):
            calls.append(kw.get("tool"))
            if kw.get("tool") == "render_check":
                return _resp_with_text("MISS: wrong subject")
            return _resp_with_image()

        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td) |
                        {"ADA_RENDER_IMAGE_SELFCHECK": "1"}), \
             patch.object(gp, "generate", side_effect=_gen), \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:e"):
            out = asyncio.run(render.run(_runner(), prompt="a cat"))
        self.assertTrue(out["ok"], out)
        self.assertEqual(calls, ["render_image", "render_check",
                                 "render_image"])
        self.assertTrue(out["selfcheck"]["retried"])
        self.assertIn("wrong subject", out["selfcheck"]["detail"])

    def test_selfcheck_pass_no_retry(self):
        import backend.gemini_pool as gp
        calls = []

        async def _gen(model, contents, **kw):
            calls.append(kw.get("tool"))
            return (_resp_with_text("PASS") if kw.get("tool") == "render_check"
                    else _resp_with_image())

        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td) |
                        {"ADA_RENDER_IMAGE_SELFCHECK": "1"}), \
             patch.object(gp, "generate", side_effect=_gen), \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:f"):
            out = asyncio.run(render.run(_runner(), prompt="a cat"))
        self.assertTrue(out["ok"], out)
        self.assertEqual(calls, ["render_image", "render_check"])
        self.assertEqual(out["selfcheck"]["verdict"], "pass")

    def test_transient_error_retries_once(self):
        import backend.gemini_pool as gp
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td) |
                        {"ADA_RENDER_RETRY_S": "0"}), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          side_effect=[ConnectionError("conn reset"),
                                       _resp_with_image()]), \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:g"):
            out = asyncio.run(render.run(_runner(), prompt="a cat"))
        self.assertTrue(out["ok"], out)

    def test_quota_error_does_not_retry(self):
        import backend.gemini_pool as gp
        exc = gp.QuotaExhaustedError("429 quota exhausted")
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td) |
                        {"ADA_RENDER_RETRY_S": "0"}), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          side_effect=exc) as gen, \
             patch.object(gp, "configured_keys", return_value=["k1"]):
            out = asyncio.run(render.run(_runner(), prompt="a cat"))
        self.assertFalse(out["ok"])
        self.assertIn("failed", out["error"])
        self.assertEqual(gen.await_count, 1)

    def test_big_render_attaches_full_url(self):
        import backend.gemini_pool as gp
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td) |
                        {"ADA_RELAY_PUBLIC_BASE": "https://r.test/img"}), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()), \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:h"):
            out = asyncio.run(render.run(_runner(), prompt="big",
                                         size="2K"))
            arch = Path(td) / "renders" / Path(out["file"]).name
            self.assertTrue(arch.is_file())
        self.assertTrue(out["ok"], out)
        self.assertTrue(str(out["full_url"]).startswith(
            "https://r.test/img/arch/renders/"))

    def test_landing_page_notice(self):
        import backend.gemini_pool as gp
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("os.environ", self._gemini_env(td)), \
             patch.object(gp, "generate", new_callable=AsyncMock,
                          return_value=_resp_with_image()), \
             patch.object(gp, "configured_keys", return_value=["k1"]), \
             patch("backend.traffic_camera.publish_relay",
                   return_value="https://x/frame?token=render:i"):
            out = asyncio.run(render.run(
                _runner(), prompt="a 1920px landing page for a cafe"))
        self.assertTrue(out["ok"], out)
        self.assertIn("picture", out["notice"])
        self.assertIn("pixel", out["notice"])


if __name__ == "__main__":
    unittest.main()

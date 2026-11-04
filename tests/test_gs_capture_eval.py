from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "backend" / "tools.d" / "gs_capture_eval.py"
SPEC = importlib.util.spec_from_file_location("gs_capture_eval", MODULE_PATH)
assert SPEC and SPEC.loader
gs_capture_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gs_capture_eval)


class FakeProcess:
    returncode = 0

    async def communicate(self):
        return (
            json.dumps({
                "dataset": "/work/gs-fixtures/hibiscus",
                "images": 14,
                "gates": {
                    "non_empty": True,
                    "same_dimensions": True,
                    "duplicate_groups": 0,
                    "min_sharpness": 17.328,
                    "sharpness_median": 270.383,
                },
                "recommendation": "train_baseline_then_review_masks",
            }).encode(),
            b"",
        )


class GsCaptureEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_directory_outside_fixture_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"ADA_GS_IMAGE_ROOTS": "/work/gs-fixtures"}):
                result = await gs_capture_eval.run(None, image_dir=tmp)
        self.assertFalse(result["ok"])
        self.assertIn("outside", result["error"])

    async def test_returns_bounded_evidence_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "hibiscus"
            fixture.mkdir()
            env = {
                "ADA_GS_IMAGE_ROOTS": tmp,
                "ADA_GS_EVAL_SCRIPT": __file__,
            }
            with patch.dict(os.environ, env):
                with patch.object(
                    asyncio, "create_subprocess_exec",
                    return_value=FakeProcess(),
                ):
                    result = await gs_capture_eval.run(
                        None, image_dir=str(fixture))
        self.assertTrue(result["ok"])
        self.assertEqual(result["images"], 14)
        self.assertEqual(result["gates"]["duplicate_groups"], 0)
        self.assertEqual(
            result["recommendation"],
            "train_baseline_then_review_masks",
        )
        self.assertTrue(result["read_only"])

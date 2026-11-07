"""Read-only Gaussian Splat capture evaluation for Ada.

The evaluator lives in the Chaba checkout so it can also be called through
the Kornia MCP server. This Ada tool is a small, fixed-command bridge for the
first supervised test; it never shells out through a user-provided command.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


DECLARATION = {
    "name": "gs_capture_eval",
    "description": (
        "Evaluate a local image set for Gaussian Splat capture readiness "
        "using the deterministic read-only evaluator."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_dir": {
                "type": "string",
                "description": (
                    "Read-only local image directory under the configured "
                    "Gaussian fixture roots."
                ),
            },
        },
        "required": ["image_dir"],
        "additionalProperties": False,
    },
}


def _configured_roots() -> list[Path]:
    raw = os.environ.get("ADA_GS_IMAGE_ROOTS", "/work/gs-fixtures")
    return [Path(item).expanduser().resolve()
            for item in raw.split(os.pathsep) if item.strip()]


def _allowed_directory(value: str) -> tuple[Path | None, str | None]:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        return None, f"image directory does not exist: {root}"
    if not any(root == allowed or allowed in root.parents
               for allowed in _configured_roots()):
        return None, "image directory is outside the configured fixture roots"
    return root, None


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    gates = payload.get("gates") or {}
    return {
        "ok": True,
        "dataset": payload.get("dataset"),
        "images": payload.get("images", 0),
        "gates": {
            "non_empty": gates.get("non_empty", False),
            "same_dimensions": gates.get("same_dimensions", False),
            "duplicate_groups": gates.get("duplicate_groups", 0),
            "min_sharpness": gates.get("min_sharpness", 0.0),
            "sharpness_median": gates.get("sharpness_median", 0.0),
        },
        "recommendation": payload.get("recommendation", "reject"),
        "read_only": True,
    }


async def run(runner, **args) -> dict[str, Any]:
    del runner
    image_dir, error = _allowed_directory(str(args.get("image_dir") or ""))
    if error:
        return {"ok": False, "error": error}

    script = Path(os.environ.get(
        "ADA_GS_EVAL_SCRIPT",
        "/home/tony/CascadeProjects/chaba/scripts/gs/image_set_eval.py",
    )).expanduser()
    if not script.is_file():
        return {"ok": False, "error": f"evaluator is unavailable: {script}"}

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), str(image_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=45)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "image evaluation timed out after 45 seconds"}
    except Exception as exc:
        return {"ok": False, "error": f"image evaluation could not start: {exc}"}

    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()[-240:]
        return {"ok": False, "error": f"image evaluation failed: {detail}"}
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"evaluator returned invalid JSON: {exc}"}
    return _summary(payload)

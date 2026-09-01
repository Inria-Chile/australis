from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def capture_provenance(root: Path, resolved_config: str) -> dict[str, Any]:
    status = _git_value(root, "status", "--porcelain")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "environment": {
            key: os.environ[key]
            for key in (
                "POLARFUNC_DATA_ROOT",
                "POLARFUNC_ARTIFACT_ROOT",
                "POLARFUNC_OUTPUT_ROOT",
                "CUDA_VISIBLE_DEVICES",
            )
            if key in os.environ
        },
        "resolved_config": resolved_config,
    }


def write_provenance(record: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

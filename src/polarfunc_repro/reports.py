from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_json_reports(root: Path) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    invalid: list[str] = []
    files = sorted(root.rglob("*.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid.append(str(path.relative_to(root)))
            continue
        if isinstance(payload, dict) and "status" in payload:
            statuses[str(payload["status"])] += 1
    return {
        "report_root": str(root),
        "json_files": len(files),
        "statuses": dict(sorted(statuses.items())),
        "invalid_json": invalid,
        "status": "PASS" if not invalid else "FAIL",
    }

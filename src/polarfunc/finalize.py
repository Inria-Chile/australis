from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_lines(root: Path, paths: list[Path]) -> list[str]:
    resolved = sorted({Path(path).resolve() for path in paths})
    lines = []
    for path in resolved:
        relative = path.relative_to(root.resolve())
        lines.append(f"{sha256(path)}  {relative}")
    return lines


def verify_checksum_lines(root: Path, lines: list[str]) -> bool:
    for line in lines:
        expected, relative = line.split("  ", 1)
        if sha256(root / relative) != expected:
            return False
    return True


def mark_current_task_pass(tasks: dict[str, dict[str, object]], task_id: str, manifest: str) -> None:
    tasks[task_id] = {"status": "PASS", "manifest": manifest, "validation_passed": True}

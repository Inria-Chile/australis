from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .manifest import sha256_file


def stable_bucket(identifier: str, bucket_count: int) -> int:
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def build_file_index(
    root: Path,
    paths: Iterable[Path],
    *,
    bucket_count: int,
    include_sha256: bool = False,
) -> list[dict[str, Any]]:
    root = root.resolve()
    records = []
    for path in sorted(path.resolve() for path in paths):
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "artifact_id": path.stem,
            "relative_path": relative,
            "bucket": stable_bucket(relative, bucket_count),
            "size_bytes": path.stat().st_size,
        }
        if include_sha256:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def write_jsonl_index(records: Iterable[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

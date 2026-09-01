from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .manifest import sha256_file


def validate_parquet_dataset(
    root: Path,
    *,
    expected_rows: int | None = None,
    required_columns: set[str] | None = None,
    include_checksums: bool = False,
) -> dict[str, Any]:
    parts = sorted(root.glob("*.parquet"))
    if not parts:
        return {"status": "FAIL", "reason": "no Parquet parts", "root": str(root)}
    dataset = ds.dataset(root, format="parquet")
    rows = dataset.count_rows()
    columns = dataset.schema.names
    required = required_columns or set()
    missing_columns = sorted(required - set(columns))
    part_records = []
    metadata_rows = 0
    for path in parts:
        metadata = pq.read_metadata(path)
        metadata_rows += metadata.num_rows
        record = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
        }
        if include_checksums:
            record["sha256"] = sha256_file(path)
        part_records.append(record)
    checks = {
        "parts_present": bool(parts),
        "metadata_matches_dataset": metadata_rows == rows,
        "expected_rows": expected_rows is None or rows == expected_rows,
        "required_columns": not missing_columns,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "root": str(root),
        "rows": rows,
        "columns": columns,
        "part_count": len(parts),
        "total_size_bytes": sum(record["size_bytes"] for record in part_records),
        "missing_columns": missing_columns,
        "checks": checks,
        "parts": part_records,
    }


def write_dataset_validation(result: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

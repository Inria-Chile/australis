#!/usr/bin/env python3
"""Build and audit the extended-minus-core GPU manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ID_CANDIDATES = ("CDHit_ID", "sequence_id", "unigene_id", "id", "protein_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--expected-remainder", type=int, default=42_272)
    args = parser.parse_args()

    core = pq.read_table(args.core)
    extended = pq.read_table(args.extended)
    if core.schema != extended.schema:
        raise SystemExit("Core and extended schemas differ")

    id_column = next((name for name in ID_CANDIDATES if name in core.column_names), None)
    if id_column is None:
        raise SystemExit(f"No supported ID column in {core.column_names}")

    core_ids = core[id_column]
    extended_ids = extended[id_column]
    if pc.count_distinct(core_ids).as_py() != core.num_rows:
        raise SystemExit("Core IDs are not unique")
    if pc.count_distinct(extended_ids).as_py() != extended.num_rows:
        raise SystemExit("Extended IDs are not unique")

    in_core = pc.is_in(extended_ids, value_set=core_ids)
    core_rows_in_extended = pc.sum(pc.cast(in_core, pa.int64())).as_py()
    if core_rows_in_extended != core.num_rows:
        raise SystemExit("Core is not an exact ID subset of extended")

    remainder = extended.filter(pc.invert(in_core))
    if remainder.num_rows != args.expected_remainder:
        raise SystemExit(
            f"Expected {args.expected_remainder} remainder rows, got {remainder.num_rows}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    pq.write_table(remainder, args.output, compression="zstd")

    reread = pq.read_table(args.output)
    overlap = pc.sum(
        pc.cast(pc.is_in(reread[id_column], value_set=core_ids), pa.int64())
    ).as_py()
    report = {
        "status": "PASS",
        "id_column": id_column,
        "core_rows": core.num_rows,
        "extended_rows": extended.num_rows,
        "remainder_rows": reread.num_rows,
        "core_rows_in_extended": core_rows_in_extended,
        "core_remainder_overlap": overlap,
        "schema_equal": reread.schema == extended.schema,
        "core_sha256": sha256(args.core),
        "extended_sha256": sha256(args.extended),
        "remainder_sha256": sha256(args.output),
        "output": str(args.output.resolve()),
    }
    if overlap or not report["schema_equal"]:
        raise SystemExit("Remainder validation failed")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    args.report_md.write_text(
        "# ESM2 extended remainder manifest\n\n"
        f"- Status: **{report['status']}**\n"
        f"- Core rows: {report['core_rows']:,}\n"
        f"- Extended rows: {report['extended_rows']:,}\n"
        f"- Remainder rows: {report['remainder_rows']:,}\n"
        f"- Core/remainder overlap: {report['core_remainder_overlap']}\n"
        f"- ID column: `{id_column}`\n"
        f"- Remainder SHA256: `{report['remainder_sha256']}`\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

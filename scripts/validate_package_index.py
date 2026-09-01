#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.package_validation import validate_package_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate deterministic structural package archives")
    parser.add_argument("index", type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--reference-id-column", default="CDHit_ID")
    parser.add_argument("--expected-kind", action="append", required=True)
    parser.add_argument("--archive-checksums", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate_package_index(
        args.index,
        reference_manifest=args.reference_manifest,
        reference_id_column=args.reference_id_column,
        expected_kinds=set(args.expected_kind),
        archive_checksums=args.archive_checksums,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "archive_records"}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

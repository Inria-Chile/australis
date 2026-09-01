#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.dataset_validation import (
    validate_parquet_dataset,
    write_dataset_validation,
)

REQUIRED_COLUMNS = {
    "GENE_ID",
    "CDHit_ID",
    "AGC_ID",
    "effective_AGNOSTOS_category",
    "strict_known",
    "strict_unknown",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen ACE unigene master")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=89_739_060)
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()
    result = validate_parquet_dataset(
        args.dataset,
        expected_rows=args.expected_rows,
        required_columns=REQUIRED_COLUMNS,
        include_checksums=args.checksums,
    )
    write_dataset_validation(result, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

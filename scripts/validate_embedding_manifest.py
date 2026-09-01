#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.embedding_validation import validate_embedding_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a compact embedding manifest and shards")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--shard-path-column", required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-id-column")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-dimension", type=int)
    parser.add_argument("--expected-model")
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate_embedding_manifest(
        args.manifest,
        id_column=args.id_column,
        shard_path_column=args.shard_path_column,
        reference_manifest=args.reference_manifest,
        reference_id_column=args.reference_id_column,
        expected_rows=args.expected_rows,
        expected_dimension=args.expected_dimension,
        expected_model=args.expected_model,
        checksums=args.checksums,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "shard_records"}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

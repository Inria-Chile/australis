#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from polarfunc_repro.annotation import build_unigene_master


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the streaming ACE unigene registry")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=Path("configs/data/ace_schema.yaml"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    schema = yaml.safe_load(args.schema.read_text(encoding="utf-8"))
    result = build_unigene_master(
        args.input,
        args.output_dir,
        functional_columns=schema["functional_columns"],
        chunksize=args.chunksize,
        max_chunks=args.max_chunks,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.embedding_index import index_npz_shards, write_embedding_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a validated embedding-shard index")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pattern", default="shard_*.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-dimension", type=int)
    args = parser.parse_args()
    paths = args.root.rglob(args.pattern)
    index, summary = index_npz_shards(
        paths,
        root=args.root,
        representation=args.representation,
        model_revision=args.model_revision,
        expected_dimension=args.expected_dimension,
    )
    write_embedding_index(index, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

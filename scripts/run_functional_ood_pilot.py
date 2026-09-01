from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.ood_pilot import run_pfam_open_set_pilot
from polarfunc_repro.open_set import ClassHoldoutConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the POLAR-FUNC Pfam open-set CPU pilot")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        action="append",
        help="Embedding shard directory; repeat for split collections",
    )
    parser.add_argument(
        "--fuse-embedding-dir",
        type=Path,
        action="append",
        help="Second embedding collection to concatenate by gene ID; repeat for its shards",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-class-size", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_pfam_open_set_pilot(
        annotations_path=args.annotations,
        split_manifest_path=args.split_manifest,
        embedding_dirs=args.embedding_dir,
        fusion_embedding_dirs=args.fuse_embedding_dir,
        output_dir=args.output_dir,
        config=ClassHoldoutConfig(min_class_size=args.min_class_size, seed=args.seed),
        pca_components=args.pca_components,
        knn_k=args.knn_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

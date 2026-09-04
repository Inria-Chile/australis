#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.report_json, args.report_md, args.output_dir / "embedding_manifest.parquet"):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
    manifest = pd.read_parquet(args.manifest, columns=["CDHit_ID", "protein_length"])
    if not manifest.CDHit_ID.is_unique:
        raise ValueError("Input manifest IDs are not unique")

    rows: list[dict[str, object]] = []
    excluded_rows: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    dimensions: set[int] = set()
    nonfinite = zero_norm = 0
    checksum_failures: list[str] = []
    for shard in range(args.num_shards):
        stem = f"shard_{shard:05d}"
        archive = args.output_dir / f"{stem}.npz"
        meta_path = args.output_dir / f"{stem}.json"
        done = args.output_dir / f"{stem}.DONE"
        checksum = args.output_dir / f"{stem}.sha256"
        excluded = args.output_dir / f"{stem}.excluded.parquet"
        if not all(path.exists() for path in (archive, meta_path, done, checksum, excluded)):
            raise FileNotFoundError(f"Incomplete shard {shard}")
        for line in checksum.read_text().splitlines():
            expected, name = line.split("  ", 1)
            if sha256_file(args.output_dir / name) != expected:
                checksum_failures.append(name)
        data = np.load(archive, allow_pickle=False)
        identifiers = data["ids"].astype(str)
        embeddings = data["embeddings"]
        meta = json.loads(meta_path.read_text())
        if embeddings.ndim != 2 or len(identifiers) != embeddings.shape[0]:
            raise ValueError(f"Shape mismatch in shard {shard}")
        dimensions.add(int(embeddings.shape[1]))
        nonfinite += int((~np.isfinite(embeddings)).sum())
        zero_norm += int((np.linalg.norm(embeddings, axis=1) == 0).sum())
        rows.extend({"CDHit_ID": identifier, "shard_id": shard, "row_index": row} for row, identifier in enumerate(identifiers))
        excluded_rows.append(pd.read_parquet(excluded))
        metadata.append(meta)

    embedded = pd.DataFrame(rows)
    excluded = pd.concat(excluded_rows, ignore_index=True)
    context_lengths = {int(meta["context_length"]) for meta in metadata}
    if len(context_lengths) != 1:
        raise ValueError(f"Context length changed across shards: {context_lengths}")
    context = next(iter(context_lengths))
    expected_embedded = set(manifest.loc[manifest.protein_length.add(2).le(context), "CDHit_ID"].astype(str))
    expected_excluded = set(manifest.loc[manifest.protein_length.add(2).gt(context), "CDHit_ID"].astype(str))
    actual_embedded = set(embedded.CDHit_ID.astype(str))
    actual_excluded = set(excluded.CDHit_ID.astype(str))
    consistent_fields = ("revision", "pooling", "dtype", "embedding_dimension", "context_policy", "manifest_sha256")
    consistency = {field: sorted({str(meta.get(field)) for meta in metadata}) for field in consistent_fields}
    checks = {
        "all_shards_present": len(metadata) == args.num_shards,
        "checksums_verified": not checksum_failures,
        "embedded_ids_unique": embedded.CDHit_ID.is_unique,
        "embedded_ids_exact": actual_embedded == expected_embedded,
        "excluded_ids_unique": excluded.CDHit_ID.is_unique if len(excluded) else True,
        "excluded_ids_exact": actual_excluded == expected_excluded,
        "no_embedded_excluded_overlap": actual_embedded.isdisjoint(actual_excluded),
        "one_embedding_dimension": len(dimensions) == 1,
        "metadata_consistent": all(len(values) == 1 for values in consistency.values()),
        "no_nonfinite": nonfinite == 0,
        "no_zero_norm": zero_norm == 0,
    }
    passed = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedded.sort_values("CDHit_ID").to_parquet(args.output_dir / "embedding_manifest.parquet", index=False)
    report = {
        "status": "PASS" if passed else "BLOCKED",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "output_dir": str(args.output_dir.resolve()),
        "num_shards": args.num_shards,
        "expected_total": len(manifest),
        "embedded": len(embedded),
        "excluded_context": len(excluded),
        "missing": len(expected_embedded - actual_embedded),
        "duplicates": int(embedded.CDHit_ID.duplicated().sum()),
        "embedding_dimension": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "context_length": context,
        "consistency": consistency,
        "checksum_failures": checksum_failures,
        "checks": checks,
        "performance": {
            "inference_seconds_sum": sum(float(meta.get("inference_seconds", 0)) for meta in metadata),
            "gpu_hours_sum": sum(float(meta.get("inference_seconds", 0)) for meta in metadata) / 3600,
            "peak_gpu_memory_mib_max": max(int(meta.get("peak_gpu_memory_mib", 0)) for meta in metadata),
            "sequences_per_second_sum": sum(float(meta.get("sequences_per_second", 0)) for meta in metadata),
            "residues_per_second_sum": sum(float(meta.get("residues_per_second", 0)) for meta in metadata),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.report_md.write_text(
        "# ESM2 shard validation\n\n"
        f"Status: `{'PASS' if passed else 'BLOCKED'}`. Embedded `{len(embedded):,}`; context-excluded `{len(excluded):,}`; missing `{report['missing']:,}`; duplicates `{report['duplicates']:,}`.\n\n"
        f"Embedding dimension `{report['embedding_dimension']}`; context `{context}`.\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

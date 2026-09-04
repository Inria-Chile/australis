#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.report_json, args.report_md, args.output_dir / "embedding_manifest.parquet"):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
    expected = set(pd.read_parquet(args.manifest, columns=["CDHit_ID"]).CDHit_ID.astype(str))
    rows = []
    excluded_rows = []
    metadata = []
    dimensions = set()
    checksum_failures = []
    nonfinite = 0
    zero_norm = 0
    for shard in range(args.num_shards):
        stem = f"shard_{shard:05d}"
        paths = {
            "archive": args.output_dir / f"{stem}.npz",
            "metadata": args.output_dir / f"{stem}.json",
            "excluded": args.output_dir / f"{stem}.excluded.parquet",
            "checksum": args.output_dir / f"{stem}.sha256",
            "done": args.output_dir / f"{stem}.DONE",
        }
        if not all(path.exists() for path in paths.values()):
            raise FileNotFoundError(f"Incomplete shard {shard}")
        for line in paths["checksum"].read_text().splitlines():
            digest, name = line.split("  ", 1)
            if sha256(args.output_dir / name) != digest:
                checksum_failures.append(name)
        data = np.load(paths["archive"], allow_pickle=False)
        identifiers = data["ids"].astype(str)
        embeddings = data["embeddings"]
        meta = json.loads(paths["metadata"].read_text())
        dimensions.add(int(embeddings.shape[1]))
        nonfinite += int((~np.isfinite(embeddings)).sum())
        zero_norm += int((np.linalg.norm(embeddings, axis=1) == 0).sum())
        rows.extend({"CDHit_ID": value, "shard_id": shard, "row_index": index} for index, value in enumerate(identifiers))
        excluded_rows.append(pd.read_parquet(paths["excluded"]))
        metadata.append(meta)

    embedded = pd.DataFrame(rows)
    excluded = pd.concat(excluded_rows, ignore_index=True)
    embedded_ids = set(embedded.CDHit_ID.astype(str))
    excluded_ids = set(excluded.CDHit_ID.astype(str))
    fields = ("revision", "pooling", "dtype", "embedding_dimension", "context_length", "context_policy", "attn_implementation")
    consistency = {field: sorted({str(meta.get(field)) for meta in metadata}) for field in fields}
    checks = {
        "checksums_verified": not checksum_failures,
        "embedded_ids_unique": embedded.CDHit_ID.is_unique,
        "excluded_ids_unique": excluded.CDHit_ID.is_unique if len(excluded) else True,
        "expected_ids_accounted_exactly": embedded_ids | excluded_ids == expected,
        "embedded_excluded_disjoint": embedded_ids.isdisjoint(excluded_ids),
        "one_embedding_dimension": len(dimensions) == 1,
        "metadata_consistent": all(len(values) == 1 for values in consistency.values()),
        "no_nonfinite": nonfinite == 0,
        "no_zero_norm": zero_norm == 0,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED"
    embedded.sort_values("CDHit_ID").to_parquet(args.output_dir / "embedding_manifest.parquet", index=False)
    report = {
        "status": status, "expected_total": len(expected), "embedded": len(embedded),
        "excluded_context": len(excluded), "embedding_dimension": next(iter(dimensions)),
        "manifest_sha256": sha256(args.manifest), "checks": checks,
        "consistency": consistency, "checksum_failures": checksum_failures,
        "performance": {
            "inference_seconds_sum": sum(float(meta["inference_seconds"]) for meta in metadata),
            "sequences_per_second_sum": sum(float(meta["sequences_per_second"]) for meta in metadata),
            "peak_gpu_memory_mib_max": max(int(meta["peak_gpu_memory_mib"]) for meta in metadata),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    args.report_md.write_text(
        "# GenomeOcean smoke validation\n\n"
        f"- Status: **{status}**\n"
        f"- Embedded: {len(embedded):,}/{len(expected):,}\n"
        f"- Dimension: {report['embedding_dimension']:,}\n"
        f"- Context exclusions: {len(excluded):,}\n"
        f"- Peak VRAM: {report['performance']['peak_gpu_memory_mib_max']:,} MiB\n"
        f"- Aggregate shard throughput: {report['performance']['sequences_per_second_sum']:.2f} sequences/s\n"
    )
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

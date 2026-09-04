#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.benchmark import add_matching_bins, assert_no_split_leakage, assign_split_groups, benchmark_join_query, coarsened_exact_match, select_nested_groups
from polarfunc.provenance import sha256_file, utc_now, write_json
from polarfunc.sequences import dataframe_to_markdown


def standardized_mean_difference(frame: pd.DataFrame, column: str, left: str, right: str) -> float:
    a = frame.loc[frame.function_label == left, column].astype(float)
    b = frame.loc[frame.function_label == right, column].astype(float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze leakage-safe nested POLAR-FUNC benchmark v1.")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--quantile-bins", type=int, default=2)
    parser.add_argument("--top-phyla", type=int, default=10)
    args = parser.parse_args()
    started_at = utc_now()
    paths = {
        "groups": ROOT / "artifacts/homology/candidate_split_groups.parquet",
        "qc": ROOT / "artifacts/sequences/candidate_translation_qc.parquet",
        "candidates": ROOT / "manifests/candidate_pool_v1.parquet",
        "abundance": ROOT / "artifacts/wave2/agc_abundance_covariates_combined.parquet",
        "taxonomy": ROOT / "artifacts/wave2/agc_taxonomy.parquet",
    }
    outputs = {size: ROOT / f"manifests/benchmark_v1_{size//1000}k.parquet" for size in (50_000, 100_000, 200_000)}
    split_path = ROOT / "manifests/split_manifest_v1.parquet"
    report_path = ROOT / "reports/wave2/benchmark_balance.md"
    audit_path = ROOT / "reports/wave2/benchmark_leakage_audit.json"
    card_path = ROOT / "docs/BENCHMARK_CARD_v1.md"
    final_paths = [*outputs.values(), split_path, report_path, audit_path, card_path]
    for path in final_paths:
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise FileExistsError(f"Refusing to overwrite frozen benchmark output: {path}")

    con = duckdb.connect()
    frame = con.execute(benchmark_join_query(paths)).fetchdf()
    con.close()
    if not frame.CDHit_ID.is_unique or not frame.AGC_ID.is_unique:
        raise RuntimeError("Headline pool must contain one unique CDHit_ID per unique AGC_ID")
    binned, bin_columns = add_matching_bins(frame, quantiles=args.quantile_bins, top_phyla=args.top_phyla)
    matched = coarsened_exact_match(binned, bin_columns, seed=args.seed)
    if len(matched) < 200_000:
        raise RuntimeError(f"Coarsened exact matching retained only {len(matched):,}; do not relax matching silently")

    split_groups = assign_split_groups(matched, seed=args.seed)
    split_manifest = split_groups.copy()
    split_lookup = split_groups.set_index("split_group").split
    matched["split"] = matched.split_group.map(split_lookup)
    target_by_level = {
        50_000: {"train": 35_000, "validation": 7_500, "test": 7_500},
        100_000: {"train": 70_000, "validation": 15_000, "test": 15_000},
        200_000: {"train": 140_000, "validation": 30_000, "test": 30_000},
    }
    selected_by_level = {size: set() for size in target_by_level}
    for split in ("train", "validation", "test"):
        subset = split_groups.loc[split_groups.split == split, ["split_group", "group_size", "assignment_hash"]]
        targets = [target_by_level[size][split] for size in sorted(target_by_level)]
        nested = select_nested_groups(subset, targets)
        for size, target in zip(sorted(target_by_level), targets):
            selected_by_level[size].update(nested[target])

    audit = {"validation": {}, "levels": {}, "matching": {"input_valid_proteins": len(frame), "matched_pool": len(matched), "quantile_bins": args.quantile_bins, "top_phyla": args.top_phyla}}
    previous_ids: set[str] = set()
    benchmark_frames = {}
    for size in sorted(outputs):
        benchmark = matched.loc[matched.split_group.isin(selected_by_level[size])].copy()
        checks = assert_no_split_leakage(benchmark)
        checks["exact_size"] = len(benchmark) == size
        ids = set(benchmark.CDHit_ID)
        checks["nested_superset"] = previous_ids.issubset(ids)
        if not all(checks.values()):
            raise RuntimeError(f"Benchmark {size} validation failed: {checks}")
        atomic_parquet(benchmark, outputs[size])
        audit["levels"][str(size)] = {"rows": len(benchmark), "checks": checks, "split_counts": benchmark.split.value_counts().to_dict(), "stratum_counts": benchmark.stratum.value_counts().to_dict()}
        benchmark_frames[size] = benchmark
        previous_ids = ids
    atomic_parquet(split_manifest, split_path)
    audit["validation"] = {"passed": True, "all_levels_passed": True, "nested_50_in_100": set(benchmark_frames[50_000].CDHit_ID).issubset(set(benchmark_frames[100_000].CDHit_ID)), "nested_100_in_200": set(benchmark_frames[100_000].CDHit_ID).issubset(set(benchmark_frames[200_000].CDHit_ID))}
    write_json(audit_path, audit)

    largest = benchmark_frames[200_000]
    numeric = ["length_nt", "GC_fraction", "protein_length", "log1p_AGC_n_unigenes", "prevalence", "log1p_mean_abundance"]
    smd = pd.DataFrame({"covariate": numeric, "SMD_known_minus_unknown": [standardized_mean_difference(largest, column, "strict_known", "strict_unknown") for column in numeric]})
    distribution = largest.groupby(["split", "stratum"]).size().rename("n").reset_index()
    report_path.write_text("\n".join(["# W2-12 benchmark balance", "", f"Coarsened exact matched pool: {len(matched):,} proteins.", "", "## Standardized mean differences (200k)", "", dataframe_to_markdown(smd), "", "## Split and stratum counts (200k)", "", dataframe_to_markdown(distribution), ""]))
    card_path.write_text("\n".join(["# POLAR-FUNC Benchmark Card v1", "", "Nested protein benchmarks of 50k, 100k, and 200k ACE unigenes, with one unigene per AGC.", "", "## Leakage controls", "", "Train/validation/test assignment is made at the MMseqs remote-homology group level. CDHit IDs, AGCs, exact sequence hashes, and MMseqs split groups are disjoint across partitions.", "", "## Matching", "", f"Known and strict-unknown examples were coarsened-exact-matched within fraction and ecology class using {args.quantile_bins}-quantile bins for nucleotide length, GC, protein length, AGC size, prevalence, and mean abundance, plus the {args.top_phyla} most frequent phyla (others collapsed).", "", "## Scope", "", "This is a post-selection predictive benchmark over protein-valid candidates. It is descriptive/predictive, not a causal population estimate.", ""]))
    manifest_outputs = [*outputs.values(), split_path, report_path, audit_path, card_path]
    manifest = {"task_id": "W2-12", "status": "PASS", "started_at": started_at, "completed_at": utc_now(), "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"), "command": " ".join(sys.argv), "parameters": vars(args), "inputs": [{"name": name, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in paths.items()], "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in manifest_outputs], "validation": audit["validation"]}
    write_json(ROOT / "manifests/wave2/W2-12/manifest.json", manifest)
    print(json.dumps({"matched_pool": len(matched), "levels": {size: len(frame) for size, frame in benchmark_frames.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

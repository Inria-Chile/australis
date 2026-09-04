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
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json


def sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def union(parent: np.ndarray, left: int, right: int) -> None:
    while parent[left] != left:
        parent[left] = parent[parent[left]]
        left = int(parent[left])
    while parent[right] != right:
        parent[right] = parent[parent[right]]
        right = int(parent[right])
    if left == right:
        return
    smaller, larger = (left, right) if left < right else (right, left)
    parent[larger] = smaller


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AGC connectivity induced by multi-AGC unigenes.")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp/lvalenzuela/polarfunc/w2-02"))
    parser.add_argument("--giant-fraction-threshold", type=float, default=0.01)
    parser.add_argument("--giant-size-threshold", type=int, default=100_000)
    args = parser.parse_args()
    started_at = utc_now()
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    pairs = ROOT / "artifacts/master/cdhit_agc_pairs.parquet"
    summary = ROOT / "artifacts/wave2/unigene_multi_agc_summary.parquet"
    stats = ROOT / "artifacts/master/agc_stats.parquet"
    ecology_fl = ROOT / "artifacts/ecology/agc_ecology_FL.parquet"
    ecology_att = ROOT / "artifacts/ecology/agc_ecology_ATT.parquet"
    staged_pairs = args.temp_dir / "multi_agc_pairs.parquet"
    output = ROOT / "artifacts/wave2/agc_connectivity_components.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite validated output: {output}")
    temporary.unlink(missing_ok=True)

    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute("SET memory_limit='4TB'")
    connection.execute(f"SET temp_directory='{sql_path(args.temp_dir)}'")
    if not staged_pairs.exists():
        connection.execute(
            f"""
            COPY (
                SELECT p.CDHit_ID, p.AGC_ID
                FROM read_parquet('{sql_path(pairs)}') p
                INNER JOIN read_parquet('{sql_path(summary)}') s USING (CDHit_ID)
                WHERE s.is_multi_agc
                ORDER BY p.CDHit_ID, p.AGC_ID
            ) TO '{sql_path(staged_pairs)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
            """
        )

    pair_table = pq.read_table(staged_pairs, columns=["CDHit_ID", "AGC_ID"])
    pair_frame = pair_table.to_pandas()
    agc_codes, agc_ids = pd.factorize(pair_frame["AGC_ID"], sort=True)
    unigene_codes, _ = pd.factorize(pair_frame["CDHit_ID"], sort=False)
    boundaries = np.flatnonzero(unigene_codes[1:] != unigene_codes[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(pair_frame)]))
    pair_count = len(pair_frame)
    multi_unigene_count = len(starts)
    parent = np.arange(len(agc_ids), dtype=np.int64)
    edge_count = 0
    for start, end in zip(starts, ends, strict=True):
        anchor = int(agc_codes[start])
        for position in range(int(start) + 1, int(end)):
            union(parent, anchor, int(agc_codes[position]))
            edge_count += 1
    while True:
        compressed = parent[parent]
        if np.array_equal(compressed, parent):
            break
        parent = compressed
    roots, component_ids, component_sizes = np.unique(
        parent, return_inverse=True, return_counts=True
    )
    output_table = pa.table(
        {
            "AGC_ID": pa.array(agc_ids.astype(str), type=pa.string()),
            "component_id": pa.array(component_ids, type=pa.int64()),
            "component_size": pa.array(component_sizes[component_ids], type=pa.int64()),
            "is_cross_agc_component": pa.array(component_sizes[component_ids] > 1),
        }
    )
    pq.write_table(output_table, temporary, compression="zstd", row_group_size=500_000)
    del pair_frame, pair_table, output_table

    observed_rows = pq.ParquetFile(temporary).metadata.num_rows
    expected_rows = int(connection.execute(f"SELECT count(DISTINCT AGC_ID) FROM read_parquet('{sql_path(staged_pairs)}')").fetchone()[0])
    duplicates = int(connection.execute(f"SELECT count(*) - count(DISTINCT AGC_ID) FROM read_parquet('{sql_path(temporary)}')").fetchone()[0])
    largest = int(component_sizes.max())
    giant_fraction = largest / observed_rows
    use_components = largest <= args.giant_size_threshold and giant_fraction <= args.giant_fraction_threshold
    checks = {
        "all_participating_agcs_assigned": observed_rows == expected_rows,
        "one_component_per_agc": duplicates == 0,
        "all_components_cross_agc": bool(np.all(component_sizes >= 2)),
        "star_edge_count_matches_n_minus_one": edge_count == pair_count - multi_unigene_count,
    }
    if not all(checks.values()):
        write_json(ROOT / "reports/wave2/W2-02_discrepancy.json", {"checks": checks, "rows": observed_rows, "expected_rows": expected_rows})
        raise RuntimeError(f"W2-02 validation failure: {checks}")
    temporary.replace(output)

    percentiles = {
        str(percentile): float(np.percentile(component_sizes, percentile))
        for percentile in [50, 75, 90, 95, 99, 99.9]
    }
    category = connection.execute(
        f"""
        SELECT s.effective_AGNOSTOS_category, count(*) AS participating_agcs
        FROM read_parquet('{sql_path(output)}') c
        LEFT JOIN read_parquet('{sql_path(stats)}') s USING (AGC_ID)
        GROUP BY s.effective_AGNOSTOS_category
        ORDER BY participating_agcs DESC
        """
    ).fetchdf()
    rf_coverage = {}
    for fraction, ecology in [("FL", ecology_fl), ("ATT", ecology_att)]:
        rf_coverage[fraction] = int(
            connection.execute(
                f"""
                SELECT count(*) FROM read_parquet('{sql_path(output)}') c
                INNER JOIN read_parquet('{sql_path(ecology)}') e
                  ON e.AGC_ID = concat('AGC_', c.AGC_ID)
                """
            ).fetchone()[0]
        )
    top = pd.DataFrame(
        {"component_id": np.argsort(component_sizes)[::-1][:20], "component_size": np.sort(component_sizes)[::-1][:20]}
    )
    report = {
        "participating_agcs": observed_rows,
        "multi_agc_unigenes": multi_unigene_count,
        "star_edges": edge_count,
        "components": len(roots),
        "largest_component": largest,
        "largest_component_fraction": giant_fraction,
        "component_size_percentiles": percentiles,
        "rf_eligible_participating_agcs": rf_coverage,
        "split_policy": "USE_CONNECTIVITY_COMPONENT" if use_components else "EXCLUDE_MULTI_AGC_AND_USE_MMSEQS",
        "decision_thresholds": {"max_component_size": args.giant_size_threshold, "max_component_fraction": args.giant_fraction_threshold},
        "validation": {"passed": True, "checks": checks},
    }
    json_path = ROOT / "reports/wave2/agc_connectivity_summary.json"
    md_path = ROOT / "reports/wave2/agc_connectivity_summary.md"
    decision_path = ROOT / "reports/wave2/split_group_decision.md"
    write_json(json_path, report)
    md_path.write_text(
        "# Multi-AGC connectivity\n\n"
        f"- Participating AGCs: `{observed_rows:,}`\n"
        f"- Components: `{len(roots):,}`\n"
        f"- Largest component: `{largest:,}` ({giant_fraction:.4%})\n"
        f"- Split policy: `{report['split_policy']}`\n\n"
        "## Largest components\n\n```text\n" + top.to_string(index=False) + "\n```\n\n"
        "## Functional composition\n\n```text\n" + category.to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    decision_path.write_text(
        "# Frozen multi-AGC split policy\n\n"
        f"Decision: `{report['split_policy']}`.\n\n"
        f"Predeclared giant-component gates were size > {args.giant_size_threshold:,} or fraction > {args.giant_fraction_threshold:.1%}. "
        f"Observed maximum was {largest:,} ({giant_fraction:.4%}).\n",
        encoding="utf-8",
    )
    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None}
        for path in [output, json_path, md_path, decision_path]
    ]
    manifest = {
        "task_id": "W2-02", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": args.threads,
        "inputs": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in [pairs, summary]],
        "parameters": report["decision_thresholds"], "outputs": outputs, "validation": report["validation"], "notes": [report["split_policy"]],
    }
    write_json(ROOT / "manifests/wave2/W2-02/manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

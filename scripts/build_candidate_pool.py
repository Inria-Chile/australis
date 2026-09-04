#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json


def quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the oversized leakage-aware ACE candidate pool.")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit", default="256GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp") / os.environ.get("USER", "polarfunc") / "wave2_candidates")
    parser.add_argument("--cap-per-stratum", type=int, default=125_000)
    parser.add_argument("--minimum-low-per-stratum", type=int, default=25_000)
    parser.add_argument("--reuse-eligible-base", action="store_true")
    args = parser.parse_args()
    started_at = utc_now()
    base = ROOT / "artifacts/wave2/candidate_eligible_base.parquet"
    output = ROOT / "manifests/candidate_pool_v1.parquet"
    counts_path = ROOT / "reports/wave2/candidate_pool_counts.tsv"
    decision_path = ROOT / "reports/wave2/low_r2_threshold_decision.md"
    protected_paths = (output, counts_path, decision_path) if args.reuse_eligible_base else (base, output, counts_path, decision_path)
    for path in protected_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    if args.reuse_eligible_base and not base.exists():
        raise FileNotFoundError(f"Requested eligible-base reuse but file is absent: {base}")
    base.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={quote(args.memory_limit)}")
    connection.execute(f"SET temp_directory={quote(args.temp_dir)}")
    connection.execute("SET preserve_insertion_order=false")
    multi = ROOT / "artifacts/wave2/unigene_multi_agc_summary.parquet"
    ecology = {fraction: ROOT / f"artifacts/ecology/agc_ecology_{fraction}.parquet" for fraction in ("FL", "ATT")}
    union = " UNION ALL ".join(
        f"SELECT {quote(fraction)} AS fraction, * FROM read_parquet({quote(path)})" for fraction, path in ecology.items()
    )
    if not args.reuse_eligible_base:
        connection.execute(
            f"""
        COPY (
          WITH representatives AS (
            SELECT 'AGC_' || canonical_AGC_ID AS AGC_ID, CDHit_ID, strict_known, strict_unknown, is_multi_agc,
                   row_number() OVER (
                     PARTITION BY canonical_AGC_ID,
                       CASE WHEN strict_known THEN 'strict_known' WHEN strict_unknown THEN 'strict_unknown' ELSE 'other' END
                     ORDER BY CDHit_ID
                   ) AS candidate_rank
            FROM read_parquet({quote(multi)}) WHERE NOT is_multi_agc AND (strict_known OR strict_unknown)
          ), ecological AS (
            SELECT fraction, AGC_ID, R2_caret, env_AGC_paper, strong_env_AGC, effective_AGNOSTOS_category,
                   AGC_n_unigenes, AGC_n_strict_known_unigenes, AGC_n_strict_unknown_unigenes,
                   CASE WHEN AGC_n_strict_unknown_unigenes > AGC_n_strict_known_unigenes THEN 'strict_unknown'
                        WHEN AGC_n_strict_known_unigenes > AGC_n_strict_unknown_unigenes THEN 'strict_known' END AS function_label
            FROM ({union})
            WHERE R2_caret IS NOT NULL AND effective_AGNOSTOS_category <> 'DISC'
          ), linked AS (
            SELECT e.*, r.CDHit_ID, r.is_multi_agc,
                   row_number() OVER (PARTITION BY e.AGC_ID ORDER BY hash(e.AGC_ID || e.fraction || '2026')) AS fraction_rank
            FROM ecological e JOIN representatives r ON e.AGC_ID = r.AGC_ID
              AND ((e.function_label = 'strict_known' AND r.strict_known) OR (e.function_label = 'strict_unknown' AND r.strict_unknown))
              AND r.candidate_rank = 1
            WHERE e.function_label IS NOT NULL
          )
          SELECT * EXCLUDE (fraction_rank) FROM linked WHERE fraction_rank = 1
        ) TO {quote(base)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
        )
    base_rel = f"read_parquet({quote(base)})"
    initial_counts = connection.execute(
        f"""
        SELECT fraction, function_label,
               count_if(R2_caret <= 0) AS low_nonpositive,
               count_if(R2_caret > 0.5) AS high,
               quantile_cont(R2_caret, 0.10) AS q10,
               quantile_cont(R2_caret, 0.25) AS q25,
               quantile_cont(R2_caret, 0.50) AS q50
        FROM {base_rel} GROUP BY fraction, function_label ORDER BY fraction, function_label
        """
    ).fetchdf()
    use_nonpositive = bool(initial_counts.low_nonpositive.min() >= args.minimum_low_per_stratum)
    selected_quantile = None
    thresholds = {(row.fraction, row.function_label): 0.0 for row in initial_counts.itertuples()}
    selected_low_counts = initial_counts.low_nonpositive.astype(int).tolist()
    if not use_nonpositive:
        for quantile_name in ("q10", "q25", "q50"):
            candidate_thresholds = {
                (row.fraction, row.function_label): min(float(getattr(row, quantile_name)), 0.5)
                for row in initial_counts.itertuples()
            }
            base_threshold_case = "CASE " + " ".join(
                f"WHEN fraction={quote(fraction)} AND function_label={quote(label)} THEN {threshold}"
                for (fraction, label), threshold in candidate_thresholds.items()
            ) + " END"
            candidate_counts = connection.execute(
                f"SELECT fraction, function_label, count_if(R2_caret <= ({base_threshold_case})) AS low_count "
                f"FROM {base_rel} GROUP BY fraction, function_label ORDER BY fraction, function_label"
            ).fetchdf()
            thresholds = candidate_thresholds
            selected_low_counts = candidate_counts.low_count.astype(int).tolist()
            selected_quantile = quantile_name
            if candidate_counts.low_count.min() >= args.minimum_low_per_stratum:
                break
    threshold_case = "CASE " + " ".join(
        f"WHEN b.fraction={quote(fraction)} AND b.function_label={quote(label)} THEN {threshold}"
        for (fraction, label), threshold in thresholds.items()
    ) + " END"
    abundance = {fraction: ROOT / f"artifacts/wave2/agc_abundance_covariates_{fraction}.parquet" for fraction in ("FL", "ATT")}
    drivers = {fraction: ROOT / f"artifacts/ecology/agc_driver_labels_{fraction}.parquet" for fraction in ("FL", "ATT")}
    taxonomy = ROOT / "artifacts/wave2/agc_taxonomy.parquet"
    enriched_union = " UNION ALL ".join(
        f"""
        SELECT b.*, a.prevalence, a.mean_abundance, a.median_abundance, a.variance_abundance, a.max_abundance,
               t.Domain, t.Phylum, d.driver_A, d.driver_ambiguous,
               CASE WHEN b.R2_caret > 0.5 THEN 'high' WHEN b.R2_caret <= ({threshold_case}) THEN 'low' END AS ecology_label
        FROM {base_rel} b
        JOIN read_parquet({quote(abundance[fraction])}) a USING (AGC_ID)
        LEFT JOIN read_parquet({quote(taxonomy)}) t USING (AGC_ID)
        LEFT JOIN read_parquet({quote(drivers[fraction])}) d USING (AGC_ID)
        WHERE b.fraction={quote(fraction)}
        """
        for fraction in ("FL", "ATT")
    )
    connection.execute(
        f"""
        COPY (
          WITH labelled AS (
            SELECT *, fraction || '::' || function_label || '::' || ecology_label AS stratum
            FROM ({enriched_union}) WHERE ecology_label IS NOT NULL
          ), sampled AS (
            SELECT *, row_number() OVER (PARTITION BY stratum ORDER BY hash(CDHit_ID || 'candidate_pool_v1')) AS stratum_rank
            FROM labelled
          )
          SELECT * FROM sampled WHERE stratum_rank <= {args.cap_per_stratum}
          ORDER BY stratum, stratum_rank
        ) TO {quote(output)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    output_rel = f"read_parquet({quote(output)})"
    counts = connection.execute(
        f"SELECT fraction, function_label, ecology_label, count(*) AS candidates FROM {output_rel} GROUP BY ALL ORDER BY ALL"
    ).fetchdf()
    counts.to_csv(counts_path, sep="\t", index=False)
    rows, agcs, unigenes, missing_r2, multi_rows, strata = connection.execute(
        f"SELECT count(*), count(DISTINCT AGC_ID), count(DISTINCT CDHit_ID), count_if(R2_caret IS NULL), count_if(is_multi_agc), count(DISTINCT stratum) FROM {output_rel}"
    ).fetchone()
    checks = {
        "AGC_ID_unique": rows == agcs, "CDHit_ID_unique": rows == unigenes, "R2_nonmissing": missing_r2 == 0,
        "multi_AGC_excluded": multi_rows == 0, "all_eight_strata_present": strata == 8,
        "target_pool_at_least_200k": rows >= 200_000,
    }
    if not all(checks.values()):
        raise RuntimeError(f"W2-08 validation failure: {checks}")
    rule = "R2_caret <= 0" if use_nonpositive else f"stratum-specific {selected_quantile} quantile capped at R2_caret <= 0.5"
    decision_path.write_text(
        "# Low-R2 threshold decision\n\n"
        f"The frozen LOW rule is `{rule}`. It was selected before sequence extraction using availability only; minimum nonpositive LOW count was {int(initial_counts.low_nonpositive.min()):,}, selected LOW counts had minimum {min(selected_low_counts):,}, and the predeclared sufficiency floor was {args.minimum_low_per_stratum:,}.\n\n"
        "HIGH remains `R2_caret > 0.50`. Intermediate values are excluded. Multi-AGC unigenes are excluded under the W2-02 split policy.\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": "W2-08", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": args.threads,
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": None} for path in [multi, *ecology.values(), *abundance.values(), *drivers.values(), taxonomy]],
        "parameters": {"low_rule": rule, "thresholds": {f"{key[0]}::{key[1]}": value for key, value in thresholds.items()}, "cap_per_stratum": args.cap_per_stratum, "split_policy": "exclude multi-AGC unigenes", "reuse_eligible_base": args.reuse_eligible_base},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in [output, counts_path, decision_path]],
        "validation": {"passed": True, "checks": checks}, "notes": [f"Candidate rows: {rows}", initial_counts.to_json(orient="records")],
    }
    write_json(ROOT / "manifests/wave2/W2-08/manifest.json", manifest)
    print(json.dumps({"rows": rows, "low_rule": rule, "counts": counts.to_dict("records"), "validation": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

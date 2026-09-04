#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
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


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def header(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return handle.readline().rstrip("\n").split("\t")


def build_sample_sets(correspondence_path: Path, grouping_path: Path, matrix_columns: list[str]) -> dict[str, list[str]]:
    correspondence = pd.read_excel(correspondence_path, usecols=["Sample", "ACE_seq_name"])
    grouping = pd.read_parquet(grouping_path, columns=["ACE_seq_name", "Size_fraction"])
    mapping = correspondence.merge(grouping, on="ACE_seq_name", how="inner", validate="one_to_one")
    mapping["matrix_column"] = mapping["Sample"].astype(str) + "_SORTED"
    if not set(mapping["matrix_column"]).issubset(matrix_columns):
        raise ValueError("Environment-to-matrix sample mapping is incomplete")
    fl = mapping.loc[mapping["Size_fraction"].isin(["0.2-3 µm", "0.2-40 µm"]), "matrix_column"].tolist()
    att = mapping.loc[mapping["Size_fraction"].eq(">3 µm"), "matrix_column"].tolist()
    if len(fl) != 127 or len(att) != 80 or set(fl) & set(att):
        raise ValueError(f"Unexpected sample partition: FL={len(fl)}, ATT={len(att)}")
    return {"FL": fl, "ATT": att}


def relation(path: Path, columns: list[str], max_rows: int | None) -> str:
    schema = ",".join(f"{quote(column)}:{quote('VARCHAR' if column == 'AGC_ID' else 'DOUBLE')}" for column in columns)
    base = (
        f"read_csv({quote(path)}, delim='\\t', header=true, columns={{ {schema} }}, "
        "auto_detect=false, parallel=false, strict_mode=false)"
    )
    return f"(SELECT * FROM {base} LIMIT {max_rows})" if max_rows is not None else base


def thresholded(sample: str) -> str:
    column = identifier(sample)
    return f"CASE WHEN d.{column} >= 0.6 THEN coalesce(c.{column}, 0.0) ELSE 0.0 END"


def metric_expressions(samples: list[str], fraction: str) -> list[str]:
    values = [thresholded(sample) for sample in samples]
    total = " + ".join(f"({value})" for value in values)
    squares = " + ".join(f"power(({value}), 2)" for value in values)
    nonzero = " + ".join(f"CASE WHEN ({value}) > 0 THEN 1 ELSE 0 END" for value in values)
    value_list = "list_value(" + ",".join(values) + ")"
    n = len(samples)
    return [
        f"{n}::USMALLINT AS n_samples_{fraction}",
        f"({nonzero})::USMALLINT AS n_nonzero_samples_{fraction}",
        f"({nonzero})::DOUBLE / {n} AS prevalence_{fraction}",
        f"({total}) / {n} AS mean_abundance_{fraction}",
        f"CASE WHEN ({nonzero}) <= {n // 2} THEN 0.0 ELSE list_median({value_list}) END AS median_abundance_{fraction}",
        f"greatest(({squares}) / {n} - power(({total}) / {n}, 2), 0.0) AS variance_abundance_{fraction}",
        f"greatest({','.join(values)}) AS max_abundance_{fraction}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TMAX60 coverage-derived AGC abundance covariates.")
    project = ROOT.parent
    parser.add_argument("--coverage", type=Path, default=project / "data/AGNOSTOS_CLSTRLVL_GENE_MAT_COV.tsv.gz")
    parser.add_argument("--detection", type=Path, default=project / "data/AGNOSTOS_CLSTRLVL_GENE_MAT_DET_Maximum.tsv.gz")
    parser.add_argument("--correspondence", type=Path, default=project / "data/ACEsamples_CorrespondanceTable.xlsx")
    parser.add_argument("--grouping", type=Path, default=ROOT / "artifacts/environment/environment_grouping.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/wave2")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/wave2/agc_abundance_covariates_validation.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit", default="256GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp") / os.environ.get("USER", "polarfunc") / "wave2_abundance")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    started_at = utc_now()
    columns = header(args.coverage)
    if columns != header(args.detection) or len(columns) != 219:
        raise ValueError("Coverage and detection headers are not aligned at 218 samples")
    samples = build_sample_sets(args.correspondence, args.grouping, columns[1:])
    suffix = "_probe" if args.max_rows is not None else ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / f"agc_abundance_covariates_combined{suffix}.parquet"
    outputs = {fraction: args.output_dir / f"agc_abundance_covariates_{fraction}{suffix}.parquet" for fraction in ("FL", "ATT")}
    if args.max_rows is not None and args.report == ROOT / "reports/wave2/agc_abundance_covariates_validation.json":
        args.report = ROOT / "reports/wave2/agc_abundance_covariates_validation_probe.json"
    for path in [combined, *outputs.values(), args.report]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={quote(args.memory_limit)}")
    connection.execute(f"SET temp_directory={quote(args.temp_dir)}")
    connection.execute("SET preserve_insertion_order=false")
    cov = relation(args.coverage, columns, args.max_rows)
    det = relation(args.detection, columns, args.max_rows)
    metrics = metric_expressions(samples["FL"], "FL") + metric_expressions(samples["ATT"], "ATT")
    connection.execute(
        f"""
        COPY (
          SELECT 'AGC_' || c.AGC_ID AS AGC_ID, c.AGC_ID = d.AGC_ID AS matrix_ids_aligned,
                 {', '.join(metrics)}
          FROM {cov} c POSITIONAL JOIN {det} d
        ) TO {quote(combined)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    metric_names = ["n_samples", "n_nonzero_samples", "prevalence", "mean_abundance", "median_abundance", "variance_abundance", "max_abundance"]
    for fraction, path in outputs.items():
        selections = ", ".join(f"{name}_{fraction} AS {name}" for name in metric_names)
        connection.execute(
            f"COPY (SELECT AGC_ID, {selections}, 'coverage_TMAX60_detection_ge_0.6_not_DESeq_normalized' AS representation "
            f"FROM read_parquet({quote(combined)})) TO {quote(path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)"
        )
    combined_rel = f"read_parquet({quote(combined)})"
    rows, unique_rows, mismatches = connection.execute(
        f"SELECT count(*), count(DISTINCT AGC_ID), count_if(NOT matrix_ids_aligned) FROM {combined_rel}"
    ).fetchone()
    result: dict[str, object] = {
        "representation": "coverage-derived TMAX60; coverage zeroed when maximum detection <0.6; not DESeq-normalized",
        "matrix_rows": rows, "matrix_unique_AGCs": unique_rows, "matrix_id_mismatches": mismatches,
        "excluded_samples_without_model_environment": 11, "fractions": {}, "probe": args.max_rows is not None,
    }
    checks = {"matrix_ids_aligned": mismatches == 0, "AGC_ID_unique": rows == unique_rows}
    for fraction, path in outputs.items():
        relation_out = f"read_parquet({quote(path)})"
        ecology = f"read_parquet({quote(ROOT / f'artifacts/ecology/agc_ecology_{fraction}.parquet')})"
        n_samples, invalid, joined = connection.execute(
            f"SELECT min(n_samples), count_if(prevalence < 0 OR prevalence > 1 OR NOT isfinite(mean_abundance) "
            f"OR NOT isfinite(median_abundance) OR NOT isfinite(variance_abundance) OR NOT isfinite(max_abundance)), "
            f"(SELECT count(*) FROM {relation_out} a JOIN {ecology} e USING (AGC_ID)) FROM {relation_out}"
        ).fetchone()
        result["fractions"][fraction] = {"rows": rows, "samples": n_samples, "invalid_covariate_rows": invalid, "RF_universe_joined_AGCs": joined}
        checks[f"{fraction}_sample_count"] = n_samples == len(samples[fraction])
        checks[f"{fraction}_finite_and_bounded"] = invalid == 0
        checks[f"{fraction}_RF_join_nonempty"] = joined > 0
    validation = {"passed": all(checks.values()), "checks": checks}
    result["validation"] = validation
    write_json(args.report, result)
    if not validation["passed"]:
        raise RuntimeError(f"W2-04 validation failure: {checks}")
    manifest_path = args.manifest or ROOT / "manifests/wave2" / ("W2-04-PROBE" if args.max_rows is not None else "W2-04") / "manifest.json"
    manifest = {
        "task_id": "W2-04-PROBE" if args.max_rows is not None else "W2-04", "status": "PASS",
        "started_at": started_at, "completed_at": utc_now(), "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "environment_prefix": os.environ.get("CONDA_PREFIX"), "command": " ".join(sys.argv), "threads": args.threads,
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": None} for path in (args.coverage, args.detection, args.correspondence, args.grouping)],
        "parameters": {"threshold": 0.6, "sample_counts": {key: len(value) for key, value in samples.items()}, "max_rows": args.max_rows},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in [combined, *outputs.values(), args.report]],
        "validation": validation,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

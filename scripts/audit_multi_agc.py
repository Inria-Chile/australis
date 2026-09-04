#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json


EXPECTED_UNIGENES = 89_739_060
EXPECTED_MULTI = 3_572_202
EXPECTED_MAX = 168


def sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit canonical unigenes associated with multiple AGCs.")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp/lvalenzuela/polarfunc/w2-01"))
    parser.add_argument("--reuse-validated-output", action="store_true")
    args = parser.parse_args()
    started_at = utc_now()
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    pairs = ROOT / "artifacts/master/cdhit_agc_pairs.parquet"
    canonical = ROOT / "artifacts/master/cdhit_agc_canonical.parquet"
    master = ROOT / "artifacts/master/unigene_master/*.parquet"
    ecology_fl = ROOT / "artifacts/ecology/agc_ecology_FL.parquet"
    ecology_att = ROOT / "artifacts/ecology/agc_ecology_ATT.parquet"
    output = ROOT / "artifacts/wave2/unigene_multi_agc_summary.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.reuse_validated_output:
        raise FileExistsError(f"Refusing to overwrite validated output: {output}")
    if not args.reuse_validated_output:
        temporary.unlink(missing_ok=True)

    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute("SET memory_limit='4TB'")
    connection.execute(f"SET temp_directory='{sql_path(args.temp_dir)}'")
    query = f"""
        WITH grouped AS (
            SELECT CDHit_ID, count(*)::INTEGER AS n_AGCs
            FROM read_parquet('{sql_path(pairs)}')
            GROUP BY CDHit_ID
        ), annotations AS (
            SELECT CDHit_ID, strict_known, strict_unknown, effective_AGNOSTOS_category
            FROM read_parquet('{sql_path(master)}')
        )
        SELECT
            g.CDHit_ID,
            c.canonical_AGC_ID,
            g.n_AGCs,
            (g.n_AGCs > 1) AS is_multi_agc,
            a.strict_known,
            a.strict_unknown,
            a.effective_AGNOSTOS_category
        FROM grouped g
        INNER JOIN read_parquet('{sql_path(canonical)}') c USING (CDHit_ID)
        INNER JOIN annotations a USING (CDHit_ID)
    """
    if not args.reuse_validated_output:
        connection.execute(
            f"COPY ({query}) TO '{sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)"
        )
    working_output = output if args.reuse_validated_output else temporary

    totals = connection.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT CDHit_ID) AS unique_unigenes,
               count(*) FILTER (WHERE is_multi_agc) AS multi_unigenes,
               max(n_AGCs) AS max_n_agcs
        FROM read_parquet('{sql_path(working_output)}')
        """
    ).fetchone()
    missing_membership = connection.execute(
        f"""
        SELECT count(*)
        FROM read_parquet('{sql_path(working_output)}') o
        LEFT JOIN read_parquet('{sql_path(pairs)}') p
          ON o.CDHit_ID = p.CDHit_ID AND o.canonical_AGC_ID = p.AGC_ID
        WHERE p.CDHit_ID IS NULL
        """
    ).fetchone()[0]
    histogram = connection.execute(
        f"SELECT n_AGCs, count(*) AS unigenes FROM read_parquet('{sql_path(working_output)}') GROUP BY n_AGCs ORDER BY n_AGCs"
    ).fetchdf()
    categories = connection.execute(
        f"""
        SELECT effective_AGNOSTOS_category,
               count(*) AS unigenes,
               count(*) FILTER (WHERE is_multi_agc) AS multi_unigenes,
               avg(is_multi_agc::INTEGER) AS multi_prevalence
        FROM read_parquet('{sql_path(working_output)}')
        GROUP BY effective_AGNOSTOS_category
        ORDER BY effective_AGNOSTOS_category
        """
    ).fetchdf()
    rf = {}
    for fraction, ecology in [("FL", ecology_fl), ("ATT", ecology_att)]:
        row = connection.execute(
            f"""
            SELECT count(*) AS unigenes,
                   count(*) FILTER (WHERE o.is_multi_agc) AS multi_unigenes,
                   avg(o.is_multi_agc::INTEGER) AS multi_prevalence
            FROM read_parquet('{sql_path(working_output)}') o
            INNER JOIN read_parquet('{sql_path(ecology)}') e
              ON e.AGC_ID = concat('AGC_', o.canonical_AGC_ID)
            """
        ).fetchone()
        rf[fraction] = {"unigenes": row[0], "multi_unigenes": row[1], "multi_prevalence": row[2]}

    checks = {
        "rows_match_expected_unigenes": totals[0] == EXPECTED_UNIGENES,
        "unique_ids_match_expected": totals[1] == EXPECTED_UNIGENES,
        "multi_count_matches_wave1": totals[2] == EXPECTED_MULTI,
        "maximum_matches_wave1": totals[3] == EXPECTED_MAX,
        "canonical_agc_always_associated": missing_membership == 0,
    }
    if not all(checks.values()):
        discrepancy = {
            "observed": {"rows": totals[0], "unique": totals[1], "multi": totals[2], "max": totals[3], "missing_membership": missing_membership},
            "expected": {"rows": EXPECTED_UNIGENES, "multi": EXPECTED_MULTI, "max": EXPECTED_MAX},
            "checks": checks,
        }
        write_json(ROOT / "reports/wave2/W2-01_discrepancy.json", discrepancy)
        raise RuntimeError(f"W2-01 discrepancy: {json.dumps(discrepancy)}")

    if not args.reuse_validated_output:
        temporary.replace(output)
    report_tsv = ROOT / "reports/wave2/multi_agc_distribution.tsv"
    report_md = ROOT / "reports/wave2/multi_agc_distribution.md"
    report_tsv.parent.mkdir(parents=True, exist_ok=True)
    histogram.to_csv(report_tsv, sep="\t", index=False)
    tail = {f"gt_{threshold}": int(histogram.loc[histogram.n_AGCs > threshold, "unigenes"].sum()) for threshold in [5, 10, 20, 50, 100]}
    report_md.write_text(
        "# Multi-AGC unigene audit\n\n"
        f"- Canonical unigenes: `{totals[0]:,}`\n"
        f"- Multi-AGC unigenes: `{totals[2]:,}`\n"
        f"- Maximum AGCs per unigene: `{totals[3]}`\n"
        f"- Canonical membership failures: `{missing_membership}`\n\n"
        "## Tail counts\n\n"
        + "\n".join(f"- {key}: `{value:,}`" for key, value in tail.items())
        + "\n\n## Functional categories\n\n"
        + "```text\n"
        + categories.to_string(index=False)
        + "\n```"
        + "\n\n## RF-eligible canonical universes\n\n"
        + "\n".join(f"- {fraction}: `{values}`" for fraction, values in rf.items())
        + "\n",
        encoding="utf-8",
    )

    input_records = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [pairs, canonical, ecology_fl, ecology_att]
    ]
    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None}
        for path in [output, report_tsv, report_md]
    ]
    manifest = {
        "task_id": "W2-01",
        "status": "PASS",
        "started_at": started_at,
        "completed_at": utc_now(),
        "host": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv),
        "threads": args.threads,
        "inputs": input_records + [{"path": str(master), "sha256": "recorded-per-shard-in-Wave-1-P05"}],
        "parameters": {"group_count": "count over validated unique pair table", "reuse_validated_output": args.reuse_validated_output, "ecology_id_normalization": "AGC_ + canonical_AGC_ID"},
        "outputs": outputs,
        "validation": {"passed": True, "checks": checks},
        "notes": [f"RF eligibility summaries: {rf}", f"Tail counts: {tail}"],
    }
    write_json(ROOT / "manifests/wave2/W2-01/manifest.json", manifest)
    print(json.dumps({"totals": totals, "tail": tail, "rf": rf, "validation": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

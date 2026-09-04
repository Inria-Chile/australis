#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest

MISSING_SQL = "trim(coalesce(AGC_ID, '')) NOT IN ('', '-', 'NA', 'nan')"


def sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def annotation_relation(path: Path, columns: list[str], max_rows: int | None) -> str:
    types = ", ".join(f"'{name.replace(chr(39), chr(39) * 2)}': 'VARCHAR'" for name in columns)
    relation = (
        f"read_csv({sql_string(path)}, delim='\\t', header=true, "
        f"columns={{ {types} }}, auto_detect=false, parallel=true, strict_mode=false, "
        "null_padding=true, max_line_size=10000000)"
    )
    if max_rows is not None:
        return f"(SELECT * FROM {relation} LIMIT {int(max_rows)})"
    return relation


def configure(connection: duckdb.DuckDBPyConnection, threads: int, memory_limit: str, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads={int(threads)}")
    connection.execute(f"SET memory_limit={sql_string(memory_limit)}")
    connection.execute(f"SET temp_directory={sql_string(temp_dir)}")
    connection.execute("SET preserve_insertion_order=false")


def build_pairs(
    connection: duckdb.DuckDBPyConnection,
    annotation_path: Path,
    columns: list[str],
    pairs_path: Path,
    canonical_path: Path,
    *,
    max_rows: int | None,
) -> dict[str, object]:
    relation = annotation_relation(annotation_path, columns, max_rows)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
          SELECT CDHit_ID, coalesce(AGC_ID, '') AS AGC_ID,
                 count(*)::UBIGINT AS n_ORFs,
                 bool_or(GENE_ID = CDHit_ID) AS has_representative
          FROM {relation}
          GROUP BY CDHit_ID, coalesce(AGC_ID, '')
        ) TO {sql_string(pairs_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    pairs = f"read_parquet({sql_string(pairs_path)})"
    summary_row = connection.execute(
        f"""
        WITH cardinality AS (
          SELECT CDHit_ID, count(*) FILTER (WHERE {MISSING_SQL}) AS n_AGCs
          FROM {pairs} GROUP BY CDHit_ID
        )
        SELECT count(*) AS unigenes,
               count_if(n_AGCs = 1) AS unique_mapping,
               count_if(n_AGCs > 1) AS multi_mapping,
               count_if(n_AGCs = 0) AS missing_agc,
               max(n_AGCs) AS max_agcs_per_unigene
        FROM cardinality
        """
    ).fetchone()
    summary = dict(zip(["unigenes", "unique_mapping", "multi_mapping", "missing_agc", "max_agcs_per_unigene"], map(int, summary_row)))
    examples = connection.execute(
        f"""
        WITH multi AS (
          SELECT CDHit_ID FROM {pairs}
          WHERE {MISSING_SQL}
          GROUP BY CDHit_ID HAVING count(*) > 1
          ORDER BY CDHit_ID LIMIT 20
        )
        SELECT p.* FROM {pairs} p JOIN multi USING (CDHit_ID)
        ORDER BY CDHit_ID, AGC_ID
        """
    ).fetchdf()
    summary["multi_mapping_examples"] = examples.to_dict("records")
    connection.execute(
        f"""
        COPY (
          WITH scored AS (
            SELECT *, ({MISSING_SQL}) AS valid_agc,
                   max(n_ORFs) FILTER (WHERE {MISSING_SQL}) OVER (PARTITION BY CDHit_ID) AS max_support,
                   count(*) FILTER (WHERE has_representative AND {MISSING_SQL}) OVER (PARTITION BY CDHit_ID) AS n_rep_pairs,
                   row_number() OVER (
                     PARTITION BY CDHit_ID
                     ORDER BY ({MISSING_SQL}) DESC, has_representative DESC, n_ORFs DESC, AGC_ID ASC
                   ) AS rank
            FROM {pairs}
          )
          SELECT CDHit_ID, CASE WHEN valid_agc THEN AGC_ID ELSE '' END AS canonical_AGC_ID,
                 CASE
                   WHEN NOT valid_agc THEN 'missing_agc'
                   WHEN has_representative AND n_rep_pairs = 1 THEN 'representative_row'
                   WHEN has_representative THEN 'representative_tiebreak'
                   WHEN n_ORFs = max_support AND
                        count(*) FILTER (WHERE valid_agc AND n_ORFs = max_support) OVER (PARTITION BY CDHit_ID) = 1
                     THEN 'largest_orf_support'
                   ELSE 'lexical_tiebreak'
                 END AS selection_reason
          FROM scored WHERE rank = 1
        ) TO {sql_string(canonical_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    summary["pair_rows"] = int(connection.execute(f"SELECT count(*) FROM {pairs}").fetchone()[0])
    summary["canonical_rows"] = int(connection.execute(f"SELECT count(*) FROM read_parquet({sql_string(canonical_path)})").fetchone()[0])
    return summary


def build_stats(
    connection: duckdb.DuckDBPyConnection,
    pairs_path: Path,
    canonical_path: Path,
    unigene_glob: str,
    output_path: Path,
) -> dict[str, object]:
    pairs = f"read_parquet({sql_string(pairs_path)})"
    canonical = f"read_parquet({sql_string(canonical_path)})"
    unigenes = f"read_parquet({sql_string(unigene_glob)})"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
          WITH orfs AS (
            SELECT AGC_ID, sum(n_ORFs)::UBIGINT AS AGC_n_ORFs
            FROM {pairs} WHERE {MISSING_SQL} GROUP BY AGC_ID
          ), assigned AS (
            SELECT c.canonical_AGC_ID AS AGC_ID, u.strict_known, u.strict_unknown,
                   u.AGC_Cat, u.effective_AGNOSTOS_category
            FROM {canonical} c JOIN {unigenes} u USING (CDHit_ID)
            WHERE trim(c.canonical_AGC_ID) <> ''
          ), unigenes_by_agc AS (
            SELECT AGC_ID,
                   count(*)::UBIGINT AS AGC_n_unigenes,
                   count_if(strict_known)::UBIGINT AS AGC_n_strict_known_unigenes,
                   count_if(strict_unknown)::UBIGINT AS AGC_n_strict_unknown_unigenes,
                   mode(AGC_Cat) AS AGC_category,
                   mode(effective_AGNOSTOS_category) AS effective_AGNOSTOS_category
            FROM assigned GROUP BY AGC_ID
          )
          SELECT o.AGC_ID, o.AGC_n_ORFs, coalesce(u.AGC_n_unigenes, 0)::UBIGINT AS AGC_n_unigenes,
                 coalesce(u.AGC_n_unigenes, 0)::DOUBLE / o.AGC_n_ORFs AS AGC_unigene_ORF_ratio,
                 u.AGC_category, u.effective_AGNOSTOS_category,
                 coalesce(u.AGC_n_strict_known_unigenes, 0)::UBIGINT AS AGC_n_strict_known_unigenes,
                 coalesce(u.AGC_n_strict_unknown_unigenes, 0)::UBIGINT AS AGC_n_strict_unknown_unigenes,
                 CASE WHEN coalesce(u.AGC_n_unigenes, 0) = 0 THEN NULL
                      ELSE u.AGC_n_strict_unknown_unigenes::DOUBLE / u.AGC_n_unigenes END
                   AS AGC_fraction_strict_unknown
          FROM orfs o LEFT JOIN unigenes_by_agc u USING (AGC_ID)
          ORDER BY o.AGC_ID
        ) TO {sql_string(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    row = connection.execute(
        f"""SELECT count(*), sum(AGC_n_ORFs), sum(AGC_n_unigenes),
                   sum(AGC_n_strict_known_unigenes), sum(AGC_n_strict_unknown_unigenes)
            FROM read_parquet({sql_string(output_path)})"""
    ).fetchone()
    keys = ["AGCs", "ORFs", "canonical_unigenes", "strict_known_unigenes", "strict_unknown_unigenes"]
    return dict(zip(keys, map(int, row)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ACE CD-HIT/AGNOSTOS mappings and AGC statistics.")
    project = ROOT.parent
    scratch_default = Path("/tmp") / os.environ.get("USER", "polarfunc") / "polarfunc" / os.environ.get("OAR_JOB_ID", "no_oar") / "duckdb"
    parser.add_argument("--mode", choices=["pairs", "stats", "all"], default="all")
    parser.add_argument("--input", type=Path, default=project / "data" / "Annotation_Table_AGN_CDH_Tax_KEGG_EGG.tsv.gz")
    parser.add_argument("--pairs", type=Path, default=ROOT / "artifacts" / "master" / "cdhit_agc_pairs.parquet")
    parser.add_argument("--canonical", type=Path, default=ROOT / "artifacts" / "master" / "cdhit_agc_canonical.parquet")
    parser.add_argument("--unigene-glob", default=str(ROOT / "artifacts" / "master" / "unigene_master" / "*.parquet"))
    parser.add_argument("--stats", type=Path, default=ROOT / "artifacts" / "master" / "agc_stats.parquet")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit", default="128GB")
    parser.add_argument("--temp-dir", type=Path, default=scratch_default)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    started_at = utc_now()
    started = time.monotonic()
    registry = yaml.safe_load((ROOT / "config" / "schema_registry.yaml").read_text(encoding="utf-8"))
    columns = registry["tables"]["annotation"]["columns"]
    connection = duckdb.connect()
    configure(connection, args.threads, args.memory_limit, args.temp_dir)
    report: dict[str, object] = {"mode": args.mode, "probe": args.max_rows is not None}
    if args.mode in {"pairs", "all"}:
        if args.pairs.exists() or args.canonical.exists():
            raise FileExistsError("P06 output already exists; refusing to overwrite it")
        report["P06"] = build_pairs(connection, args.input, columns, args.pairs, args.canonical, max_rows=args.max_rows)
        p06_report = ROOT / "reports" / ("cdhit_agc_cardinality_probe.json" if args.max_rows else "cdhit_agc_cardinality.json")
        write_json(p06_report, report["P06"])
        p06_validation = {
            "passed": report["P06"]["canonical_rows"] == report["P06"]["unigenes"]
            and report["P06"]["unique_mapping"]
            + report["P06"]["multi_mapping"]
            + report["P06"]["missing_agc"]
            == report["P06"]["unigenes"],
            "checks": {
                "one_canonical_row_per_unigene": report["P06"]["canonical_rows"] == report["P06"]["unigenes"],
                "cardinality_classes_exhaustive": report["P06"]["unique_mapping"]
                + report["P06"]["multi_mapping"]
                + report["P06"]["missing_agc"]
                == report["P06"]["unigenes"],
                "probe_only": args.max_rows is not None,
            },
        }
        p06_outputs = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (args.pairs, args.canonical, p06_report)
        ]
        write_task_manifest(
            ROOT,
            "P06_PROBE" if args.max_rows else "P06",
            started_at=started_at,
            command=" ".join(sys.argv),
            inputs=[{"path": str(args.input), "size_bytes": args.input.stat().st_size, "sha256": None}],
            outputs=p06_outputs,
            parameters={"threads": args.threads, "memory_limit": args.memory_limit, "max_rows": args.max_rows},
            validation=p06_validation,
        )
        report["P06"]["validation"] = p06_validation
    if args.mode in {"stats", "all"}:
        if args.stats.exists():
            raise FileExistsError("P07 output already exists; refusing to overwrite it")
        report["P07"] = build_stats(connection, args.pairs, args.canonical, args.unigene_glob, args.stats)
        p07_report = ROOT / "reports" / ("agc_stats_probe.json" if args.max_rows else "agc_stats.json")
        write_json(p07_report, report["P07"])
        p07_validation = {
            "passed": report["P07"]["AGCs"] > 0
            and report["P07"]["ORFs"] > 0
            and report["P07"]["canonical_unigenes"] > 0,
            "checks": {
                "nonempty_agc_universe": report["P07"]["AGCs"] > 0,
                "nonempty_orf_universe": report["P07"]["ORFs"] > 0,
                "nonempty_unigene_universe": report["P07"]["canonical_unigenes"] > 0,
                "probe_only": args.max_rows is not None,
            },
        }
        p07_outputs = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (args.stats, p07_report)
        ]
        write_task_manifest(
            ROOT,
            "P07_PROBE" if args.max_rows else "P07",
            started_at=started_at,
            command=" ".join(sys.argv),
            inputs=[
                {"path": str(args.pairs), "size_bytes": args.pairs.stat().st_size, "sha256": None},
                {"path": str(args.canonical), "size_bytes": args.canonical.stat().st_size, "sha256": None},
                {"path": args.unigene_glob, "size_bytes": None, "sha256": None},
            ],
            outputs=p07_outputs,
            parameters={"threads": args.threads, "memory_limit": args.memory_limit},
            validation=p07_validation,
        )
        report["P07"]["validation"] = p07_validation
    report["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

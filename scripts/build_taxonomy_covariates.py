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

RANKS = ("Domain", "Phylum", "Class", "Order", "Family")


def quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical ACE unigene and AGC taxonomy covariates.")
    project = ROOT.parent
    parser.add_argument("--master-glob", default=str(ROOT / "artifacts/master/unigene_master/*.parquet"))
    parser.add_argument("--canonical", type=Path, default=ROOT / "artifacts/master/cdhit_agc_canonical.parquet")
    parser.add_argument("--gtdb", type=Path, default=project / "data/contig_GTDBlineage_kraken.tsv.gz")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit", default="256GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp") / os.environ.get("USER", "polarfunc") / "wave2_taxonomy")
    args = parser.parse_args()
    started_at = utc_now()
    out_unigene = ROOT / "artifacts/wave2/unigene_taxonomy.parquet"
    out_agc = ROOT / "artifacts/wave2/agc_taxonomy.parquet"
    report_tsv = ROOT / "reports/wave2/taxonomy_coverage.tsv"
    report_md = ROOT / "reports/wave2/taxonomy_coverage.md"
    for path in (out_unigene, out_agc, report_tsv, report_md):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    out_unigene.parent.mkdir(parents=True, exist_ok=True)
    report_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={quote(args.memory_limit)}")
    connection.execute(f"SET temp_directory={quote(args.temp_dir)}")
    connection.execute("SET preserve_insertion_order=false")
    master = f"read_parquet({quote(args.master_glob)})"
    canonical = f"read_parquet({quote(args.canonical)})"
    gtdb = (
        f"read_csv({quote(args.gtdb)}, delim='\\t', header=true, "
        "columns={'contig_id':'VARCHAR','taxid':'VARCHAR','lineage':'VARCHAR'}, auto_detect=false)"
    )
    connection.execute(
        f"""
        COPY (
          WITH gt AS (
            SELECT contig_id, taxid, lineage AS GTDB_lineage_raw,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 2)) END AS GTDB_Domain,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 3)) END AS Phylum,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 4)) END AS Class,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 5)) END AS Order_rank,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 6)) END AS Family,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 7)) END AS Genus,
                   CASE WHEN lineage IN ('', 'root', 'unclassified') THEN '' ELSE trim(list_extract(string_split(lineage, ';'), 8)) END AS Species
            FROM {gtdb}
          ), u AS (
            SELECT m.CDHit_ID, m.GENE_ID, m.DOMAIN AS Domain, m.strict_known, m.strict_unknown,
                   m.effective_AGNOSTOS_category,
                   regexp_extract(m.GENE_ID, '^ACE_(G[^_]+_[^_]+)_[^_]+$', 1) AS contig_id,
                   c.canonical_AGC_ID
            FROM {master} m JOIN {canonical} c USING (CDHit_ID)
          )
          SELECT u.*, CASE WHEN canonical_AGC_ID = '' THEN '' ELSE 'AGC_' || canonical_AGC_ID END AS AGC_ID,
                 gt.taxid AS GTDB_taxid, gt.GTDB_lineage_raw, gt.GTDB_Domain,
                 gt.Phylum, gt.Class, gt.Order_rank AS "Order", gt.Family, gt.Genus, gt.Species,
                 CASE WHEN trim(coalesce(u.Domain, '')) = '' OR trim(coalesce(gt.GTDB_Domain, '')) = '' THEN false
                      WHEN lower(u.Domain) = lower(gt.GTDB_Domain) THEN false
                      WHEN lower(u.Domain) = 'prokaryote' AND lower(gt.GTDB_Domain) IN ('bacteria', 'archaea') THEN false
                      WHEN lower(u.Domain) = 'eukaryote' AND lower(gt.GTDB_Domain) IN ('eukaryota', 'eukarya') THEN false
                      ELSE true END AS Domain_conflict,
                 CASE WHEN gt.contig_id IS NULL THEN false ELSE true END AS GTDB_joined
          FROM u LEFT JOIN gt USING (contig_id)
        ) TO {quote(out_unigene)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    unigene = f"read_parquet({quote(out_unigene)})"
    rank_expr = []
    for rank in RANKS:
        identifier = f'"{rank}"'
        rank_expr.extend(
            [
                f"mode({identifier}) FILTER (WHERE trim(coalesce({identifier}, '')) <> '') AS {identifier}",
                f"count_if(trim(coalesce({identifier}, '')) <> '')::DOUBLE / count(*) AS {rank}_coverage",
                f"max(rank_count_{rank}) FILTER (WHERE trim(coalesce({identifier}, '')) <> '')::DOUBLE / nullif(count_if(trim(coalesce({identifier}, '')) <> ''), 0) AS {rank}_purity",
            ]
        )
    count_windows = ",\n".join(
        f"count(*) OVER (PARTITION BY AGC_ID, \"{rank}\") AS rank_count_{rank}" for rank in RANKS
    )
    connection.execute(
        f"""
        COPY (
          WITH valid AS (SELECT * FROM {unigene} WHERE AGC_ID <> ''), counted AS (
            SELECT *, {count_windows} FROM valid
          )
          SELECT AGC_ID, count(*)::UBIGINT AS n_unigenes,
                 {', '.join(rank_expr)},
                 avg(GTDB_joined::INTEGER) AS GTDB_contig_coverage,
                 avg(Domain_conflict::INTEGER) AS Domain_conflict_fraction
          FROM counted GROUP BY AGC_ID ORDER BY AGC_ID
        ) TO {quote(out_agc)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    coverage_queries = []
    for rank in RANKS:
        identifier = f'"{rank}"'
        coverage_queries.append(
            f"SELECT 'unigene' AS unit, 'ALL' AS universe, 'ALL' AS category, {quote(rank)} AS rank, count(*) AS n, "
            f"count_if(trim(coalesce({identifier}, '')) <> '') AS annotated FROM {unigene}"
        )
        coverage_queries.append(
            f"SELECT 'unigene', 'ALL', CASE WHEN strict_known THEN 'strict_known' WHEN strict_unknown THEN 'strict_unknown' ELSE 'other' END, "
            f"{quote(rank)}, count(*), count_if(trim(coalesce({identifier}, '')) <> '') FROM {unigene} GROUP BY 3"
        )
        for fraction in ("FL", "ATT"):
            ecology = f"read_parquet({quote(ROOT / f'artifacts/ecology/agc_ecology_{fraction}.parquet')})"
            agc_taxonomy = f"read_parquet({quote(out_agc)})"
            coverage_queries.append(
                f"SELECT 'AGC', {quote(fraction)}, "
                "CASE WHEN e.AGC_n_strict_known_unigenes > e.AGC_n_strict_unknown_unigenes THEN 'strict_known_dominated' "
                "WHEN e.AGC_n_strict_unknown_unigenes > e.AGC_n_strict_known_unigenes THEN 'strict_unknown_dominated' ELSE 'other' END, "
                f"{quote(rank)}, count(*), count_if(trim(coalesce(t.{identifier}, '')) <> '') "
                f"FROM {ecology} e LEFT JOIN {agc_taxonomy} t USING (AGC_ID) GROUP BY 3"
            )
    coverage = connection.execute(" UNION ALL ".join(coverage_queries)).fetchdf()
    coverage["coverage"] = coverage["annotated"] / coverage["n"]
    coverage.to_csv(report_tsv, sep="\t", index=False)
    counts = connection.execute(
        f"SELECT count(*), count(DISTINCT CDHit_ID), count_if(GTDB_joined), count_if(Domain_conflict) FROM {unigene}"
    ).fetchone()
    agc_count = connection.execute(f"SELECT count(*), count(DISTINCT AGC_ID) FROM read_parquet({quote(out_agc)})").fetchone()
    validation = {
        "one_row_per_canonical_unigene": counts[0] == counts[1] == 89_739_060,
        "agc_output_unique": agc_count[0] == agc_count[1],
        "no_gtbd_join_multiplication": counts[0] == 89_739_060,
        "coverage_bounded": bool(coverage["coverage"].between(0, 1).all()),
    }
    if not all(validation.values()):
        raise RuntimeError(f"W2-03 validation failure: {validation}")
    report_md.write_text(
        "# Canonical taxonomy coverage\n\n"
        "Primary Domain source: representative ACE annotation row. Higher ranks: GTDB/Kraken contig lineage only after explicit `ACE_<contig>_<ORF>` linkage.\n\n"
        f"- Canonical unigenes: {counts[0]:,}\n- GTDB-linked unigenes: {counts[2]:,} ({counts[2] / counts[0]:.2%})\n"
        f"- Domain conflicts among all unigenes: {counts[3]:,}\n- Canonical AGCs represented: {agc_count[0]:,}\n",
        encoding="utf-8",
    )
    outputs = [out_unigene, out_agc, report_tsv, report_md]
    manifest = {
        "task_id": "W2-03", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "environment_prefix": os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV"),
        "command": " ".join(sys.argv), "threads": args.threads,
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": None} for path in (args.canonical, args.gtdb)],
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                     "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "parameters": {"taxonomy_hierarchy": ["annotation Domain", "GTDB/Kraken contig ranks"], "linkage_regex": "^ACE_(G[^_]+_[^_]+)_[^_]+$"},
        "validation": {"passed": True, "checks": validation},
    }
    write_json(ROOT / "manifests/wave2/W2-03/manifest.json", manifest)
    print(json.dumps({"canonical_unigenes": counts[0], "GTDB_linked": counts[2], "AGCs": agc_count[0], "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

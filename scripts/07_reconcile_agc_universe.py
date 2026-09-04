#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest


def counts_by_value(connection: duckdb.DuckDBPyConnection, path: Path, column: str) -> dict[str, int]:
    rows = connection.execute(
        f"SELECT coalesce(cast({column} AS VARCHAR), '<MISSING>'), count(*) "
        "FROM read_parquet(?) GROUP BY 1 ORDER BY 2 DESC",
        [str(path)],
    ).fetchall()
    return {str(value): int(count) for value, count in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile the ACE AGC universes without modifying data.")
    project = ROOT.parent
    supplementary = project / "data" / "suplemmentary" / "figshare_29821949_v1"
    parser.add_argument("--agc-stats", type=Path, default=ROOT / "artifacts" / "master" / "agc_stats.parquet")
    parser.add_argument("--rf-fl", type=Path, default=ROOT / "artifacts" / "rf" / "rf_FL.parquet")
    parser.add_argument("--rf-att", type=Path, default=ROOT / "artifacts" / "rf" / "rf_ATT.parquet")
    parser.add_argument("--cag-fl", type=Path, default=supplementary / "CAGs_T60MAX_FL_RSquared10.txt")
    parser.add_argument("--cag-att", type=Path, default=supplementary / "CAGs_T60MAX_ATT_RSquared15.txt")
    parser.add_argument("--paper-agcs", type=int, default=30_123_228)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "agc_universe_reconciliation.json")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    started = utc_now()
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    stats_count, distinct_stats_count, annotation_orfs, canonical_unigenes = map(
        int,
        connection.execute(
            "SELECT count(*), count(DISTINCT AGC_ID), sum(AGC_n_ORFs), sum(AGC_n_unigenes) FROM read_parquet(?)",
            [str(args.agc_stats)],
        ).fetchone(),
    )
    categories = counts_by_value(connection, args.agc_stats, "AGC_category")
    effective_categories = counts_by_value(connection, args.agc_stats, "effective_AGNOSTOS_category")
    rf_fl = int(connection.execute("SELECT count(DISTINCT AGC_ID) FROM read_parquet(?)", [str(args.rf_fl)]).fetchone()[0])
    rf_att = int(connection.execute("SELECT count(DISTINCT AGC_ID) FROM read_parquet(?)", [str(args.rf_att)]).fetchone()[0])
    rf_union = int(
        connection.execute(
            "SELECT count(DISTINCT AGC_ID) FROM ("
            "SELECT AGC_ID FROM read_parquet(?) UNION ALL SELECT AGC_ID FROM read_parquet(?))",
            [str(args.rf_fl), str(args.rf_att)],
        ).fetchone()[0]
    )
    rf_intersection = int(
        connection.execute(
            "SELECT count(*) FROM (SELECT AGC_ID FROM read_parquet(?) INTERSECT SELECT AGC_ID FROM read_parquet(?))",
            [str(args.rf_fl), str(args.rf_att)],
        ).fetchone()[0]
    )
    for view, path in (("cag_fl", args.cag_fl), ("cag_att", args.cag_att)):
        connection.read_csv(
            str(path),
            delimiter="\t",
            header=False,
            columns={"CAG_ID": "VARCHAR", "AGC_ID": "VARCHAR"},
        ).create_view(view, replace=True)
    cag = {}
    for fraction in ("fl", "att"):
        row = connection.execute(
            f"SELECT count(*), count(DISTINCT CAG_ID), count(DISTINCT AGC_ID) FROM cag_{fraction}"
        ).fetchone()
        cag[fraction.upper()] = {"mapping_rows": int(row[0]), "CAGs": int(row[1]), "AGCs": int(row[2])}
    cag_union = int(
        connection.execute(
            "SELECT count(DISTINCT AGC_ID) FROM (SELECT AGC_ID FROM cag_fl UNION ALL SELECT AGC_ID FROM cag_att)"
        ).fetchone()[0]
    )
    disc = categories.get("DISC", 0)
    missing_category = categories.get("<MISSING>", 0)
    report = {
        "published_AGCs": args.paper_agcs,
        "downloaded_annotation_AGCs": stats_count,
        "published_minus_downloaded": args.paper_agcs - stats_count,
        "annotation_ORFs": annotation_orfs,
        "canonical_unigenes": canonical_unigenes,
        "AGNOSTOS_categories": categories,
        "effective_AGNOSTOS_categories": effective_categories,
        "DISC_AGCs": disc,
        "AGCs_with_missing_category": missing_category,
        "non_DISC_proxy_including_missing_category": stats_count - disc,
        "non_DISC_with_explicit_category": stats_count - disc - missing_category,
        "RF": {
            "FL_AGCs": rf_fl,
            "ATT_AGCs": rf_att,
            "union_AGCs": rf_union,
            "intersection_AGCs": rf_intersection,
        },
        "CAG": {**cag, "union_mapped_AGCs": cag_union},
        "good_quality_definition": {
            "status": "not_reconstructible_from_category_alone",
            "faure_pipeline": "maximum detection >=60%, DISC removal, then downstream NZV and fraction-specific R2 filters",
            "non_DISC_proxy_is_not_headline_GQ_count": True,
            "RF_and_CAG_counts_are_downstream_observable_subsets": True,
        },
        "decision": "Preserve the downloaded 30,123,227-row AGC universe; do not synthesize the one missing published AGC.",
    }
    validation = {
        "passed": stats_count == distinct_stats_count
        and annotation_orfs == 175_336_776
        and canonical_unigenes == 89_739_060
        and rf_union == rf_fl + rf_att - rf_intersection,
        "checks": {
            "agc_ids_unique": stats_count == distinct_stats_count,
            "annotation_orfs_exact": annotation_orfs == 175_336_776,
            "canonical_unigenes_exact": canonical_unigenes == 89_739_060,
            "rf_set_arithmetic": rf_union == rf_fl + rf_att - rf_intersection,
            "no_synthetic_agc_added": stats_count == 30_123_227,
        },
    }
    report["validation"] = validation
    write_json(args.output, report)
    write_task_manifest(
        ROOT,
        "P08",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": None}
            for path in (args.agc_stats, args.rf_fl, args.rf_att, args.cag_fl, args.cag_att)
        ],
        outputs=[{"path": str(args.output), "size_bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)}],
        parameters={"paper_agcs": args.paper_agcs, "threads": args.threads},
        validation=validation,
    )
    print(json.dumps(report, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

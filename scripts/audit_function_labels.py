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


FUNCTIONAL_COLUMNS = (
    "KEGG_KO", "seed_ortholog", "eggNOG_OGs", "narr_OG_desc", "best_OG_desc",
    "Preferred_name", "CAZy", "BiGG_Reaction", "PFAMs",
)
MISSING = "('', '-', 'NA', 'nan')"


def quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ACE unigene functional-label semantics.")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--memory-limit", default="256GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp") / os.environ.get("USER", "polarfunc") / "w2c01")
    args = parser.parse_args()
    started = utc_now()
    master = ROOT / "artifacts/master/unigene_master/*.parquet"
    report_dir = ROOT / "reports/corrective"
    artifact_dir = ROOT / "artifacts/corrective"
    manifest_path = ROOT / "manifests/corrective/W2C-01/manifest.json"
    crosswalk = artifact_dir / "unigene_function_label_crosswalk.parquet"
    report_json = report_dir / "function_label_audit.json"
    report_md = report_dir / "function_label_audit.md"
    corrected = report_dir / "INACH_CPU_NUMBERS_label_corrected.json"
    outputs = (crosswalk, report_json, report_md, corrected, manifest_path)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    present_terms = [f"trim(coalesce({column}, '')) NOT IN {MISSING}" for column in FUNCTIONAL_COLUMNS]
    function_present = "(" + " OR ".join(present_terms) + ")"
    effective = "CASE WHEN AGC_Cat='SINGL' THEN AGC_Singl_Cat ELSE AGC_Cat END"
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={quoted(Path(args.memory_limit))}")
    connection.execute(f"SET temp_directory={quoted(args.temp_dir)}")
    connection.execute("SET preserve_insertion_order=false")
    source = f"read_parquet({quoted(master)})"
    connection.execute(
        f"""
        COPY (
          WITH recomputed AS (
            SELECT {function_present} AS function_present,
                   {effective} AS effective_AGNOSTOS_category_recomputed,
                   broad_unknown AS broad_unknown_current,
                   strict_known AS strict_known_current,
                   strict_unknown AS strict_unknown_current
            FROM {source}
          ), labelled AS (
            SELECT *, NOT function_present AS broad_unknown_recomputed,
                   function_present AND effective_AGNOSTOS_category_recomputed='K' AS strict_known_recomputed,
                   NOT function_present AND effective_AGNOSTOS_category_recomputed IN ('GU','EU') AS strict_unknown_recomputed
            FROM recomputed
          )
          SELECT function_present, effective_AGNOSTOS_category_recomputed,
                 broad_unknown_current, broad_unknown_recomputed,
                 strict_known_current, strict_known_recomputed,
                 strict_unknown_current, strict_unknown_recomputed,
                 count(*)::UBIGINT AS unigenes
          FROM labelled GROUP BY ALL ORDER BY ALL
        ) TO {quoted(crosswalk)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    relation = f"read_parquet({quoted(crosswalk)})"
    row = connection.execute(
        f"""
        SELECT sum(unigenes),
               sum(unigenes) FILTER (WHERE broad_unknown_recomputed),
               sum(unigenes) FILTER (WHERE strict_known_recomputed),
               sum(unigenes) FILTER (WHERE strict_unknown_recomputed),
               sum(unigenes) FILTER (WHERE broad_unknown_current <> broad_unknown_recomputed),
               sum(unigenes) FILTER (WHERE strict_known_current <> strict_known_recomputed),
               sum(unigenes) FILTER (WHERE strict_unknown_current <> strict_unknown_recomputed),
               sum(unigenes) FILTER (WHERE strict_known_recomputed AND strict_unknown_recomputed),
               sum(unigenes) FILTER (WHERE strict_unknown_recomputed AND NOT broad_unknown_recomputed),
               sum(unigenes) FILTER (WHERE strict_known_recomputed AND (NOT function_present OR effective_AGNOSTOS_category_recomputed <> 'K')),
               sum(unigenes) FILTER (WHERE strict_unknown_recomputed AND effective_AGNOSTOS_category_recomputed NOT IN ('GU','EU'))
        FROM {relation}
        """
    ).fetchone()
    keys = (
        "total_unigenes", "broad_unknown_recomputed", "strict_known_recomputed", "strict_unknown_recomputed",
        "broad_flag_mismatches", "strict_known_mismatches", "strict_unknown_mismatches", "strict_overlap",
        "strict_unknown_not_broad", "strict_known_definition_violations", "strict_unknown_category_violations",
    )
    counts = {key: int(value or 0) for key, value in zip(keys, row)}
    current_numbers_path = ROOT / "reports/INACH_CPU_NUMBERS.json"
    current_numbers = json.loads(current_numbers_path.read_text())
    reported_broad = int(current_numbers["catalogue_funnel"]["broad_unknown"])
    checks = {
        "total_exact": counts["total_unigenes"] == 89_739_060,
        "current_flags_match_recomputation": counts["broad_flag_mismatches"] == counts["strict_known_mismatches"] == counts["strict_unknown_mismatches"] == 0,
        "strict_labels_mutually_exclusive": counts["strict_overlap"] == 0,
        "strict_unknown_subset_broad": counts["strict_unknown_not_broad"] == 0,
        "strict_known_definition_valid": counts["strict_known_definition_violations"] == 0,
        "strict_unknown_categories_valid": counts["strict_unknown_category_violations"] == 0,
        "functional_partition_reconciles": counts["broad_unknown_recomputed"] <= counts["total_unigenes"],
        "reported_broad_unknown_is_wrong": reported_broad != counts["broad_unknown_recomputed"],
        "reported_value_is_complement_of_strict_known": reported_broad == counts["total_unigenes"] - counts["strict_known_recomputed"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"W2C-01 validation failed: counts={counts}, checks={checks}")
    decision = "REPORT_ONLY_BUG"
    corrected_numbers = current_numbers.copy()
    corrected_numbers["catalogue_funnel"] = dict(current_numbers["catalogue_funnel"])
    corrected_numbers["catalogue_funnel"]["broad_unknown"] = counts["broad_unknown_recomputed"]
    corrected_numbers["catalogue_funnel"]["definition"] = "representative unigene row has no nonmissing explicit functional annotation field"
    corrected_numbers["corrective_status"] = "W2C-01 REPORT_ONLY_BUG; downstream strict labels unchanged"
    write_json(corrected, corrected_numbers)
    payload = {
        "task_id": "W2C-01", "status": "PASS", "scientific_decision": decision,
        "definitions": {
            "broad_unknown": "No nonmissing value across the nine configured explicit functional fields on the CD-HIT representative row.",
            "effective_AGNOSTOS_category": "AGC_Cat, except SINGL uses AGC_Singl_Cat.",
            "strict_unknown": "broad_unknown AND effective category in {GU, EU}.",
            "strict_known": "function_present AND effective category K.",
            "annotation_scope": "CD-HIT representative row (the frozen production unigene-master definition).",
            "any_member_scope": "Not precomputed; raw ORF-level source exists but is not used by the frozen benchmark labels.",
        },
        "functional_columns": list(FUNCTIONAL_COLUMNS), "counts": counts,
        "current_inach_broad_unknown": reported_broad, "checks": checks,
        "affected_previous_tasks": ["W2-19 report only"], "stale_sequence_artifacts": [],
    }
    write_json(report_json, payload)
    report_md.write_text("\n".join([
        "# Functional-label semantic audit", "", f"Decision: `{decision}`.", "",
        f"- Canonical unigenes: `{counts['total_unigenes']:,}`",
        f"- Recomputed broad unknown: `{counts['broad_unknown_recomputed']:,}`",
        f"- Previously reported broad unknown: `{reported_broad:,}`",
        f"- Strict known: `{counts['strict_known_recomputed']:,}`",
        f"- Strict unknown: `{counts['strict_unknown_recomputed']:,}`", "",
        "The unigene-master flags match independent recomputation exactly. The erroneous INACH value was generated as total minus strict-known; candidate labels and sequence artifacts remain valid.", "",
        "The frozen production definition is based on the CD-HIT representative annotation row. An any-member ORF definition is a separate estimand and is not silently substituted.", "",
    ]), encoding="utf-8")
    manifest = {
        "task_id": "W2C-01", "status": "PASS", "started_at": started, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "git_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(), "command": " ".join(sys.argv),
        "inputs": [{"path": "artifacts/master/unigene_master/*.parquet", "sha256": None, "rows": counts["total_unigenes"]}, {"path": "reports/INACH_CPU_NUMBERS.json", "sha256": sha256_file(current_numbers_path), "rows": None}],
        "parameters": {"threads": args.threads, "memory_limit": args.memory_limit, "annotation_scope": "representative_row"},
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in (crosswalk, report_json, report_md, corrected)],
        "validation": {"passed": True, "checks": checks}, "scientific_decision": decision,
        "affected_previous_tasks": ["W2-19"], "notes": ["W2-08 through W2-12 remain valid; no sequence rebuild required."],
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "decision": decision, "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

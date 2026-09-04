#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.ecology import build_rf_ecology, validate_environment_groups
from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create ACE AGC ecological phenotype tables.")
    parser.add_argument("--rf-dir", type=Path, default=ROOT / "artifacts" / "rf")
    parser.add_argument("--agc-stats", type=Path, default=ROOT / "artifacts" / "master" / "agc_stats.parquet")
    parser.add_argument("--groups", type=Path, default=ROOT / "config" / "environment_groups.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "ecology")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    started_at = utc_now()
    started = time.monotonic()
    config = yaml.safe_load(args.groups.read_text(encoding="utf-8"))
    groups = config["groups"]
    importance_prefix = config["importance_prefix"]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_columns = [
        "AGC_ID",
        "AGC_n_ORFs",
        "AGC_n_unigenes",
        "AGC_unigene_ORF_ratio",
        "AGC_category",
        "effective_AGNOSTOS_category",
        "AGC_n_strict_known_unigenes",
        "AGC_n_strict_unknown_unigenes",
        "AGC_fraction_strict_unknown",
    ]
    stats = pd.read_parquet(args.agc_stats, columns=stats_columns)
    stats["AGC_join"] = "AGC_" + stats.pop("AGC_ID").astype("string")
    report: dict[str, object] = {"fractions": {}, "probe": args.max_rows is not None}
    outputs: list[dict[str, object]] = []
    group_validation: dict[str, object] = {}
    for fraction in ("FL", "ATT"):
        source = args.rf_dir / f"rf_{fraction}.parquet"
        frame = pd.read_parquet(source)
        if args.max_rows is not None:
            frame = frame.head(args.max_rows).copy()
        importance_columns = [column for column in frame if column.startswith(importance_prefix)]
        validation = validate_environment_groups(importance_columns, groups)
        group_validation[fraction] = validation
        ecology = build_rf_ecology(
            frame,
            fraction=fraction,
            environment_groups=groups,
            importance_prefix=importance_prefix,
        )
        ecology["AGC_join"] = ecology["AGC_ID"].astype("string")
        ecology = ecology.merge(stats, on="AGC_join", how="left", validate="many_to_one")
        ecology = ecology.drop(columns=["AGC_join"])
        output = args.output_dir / f"agc_ecology_{fraction}.parquet"
        ecology.to_parquet(output, index=False, compression="zstd")
        matched = int(ecology["AGC_n_ORFs"].notna().sum())
        summary = {
            "rows": len(ecology),
            "matched_agc_stats": matched,
            "unmatched_agc_stats": len(ecology) - matched,
            "env_AGC_paper": int(ecology["env_AGC_paper"].sum()),
            "strong_env_AGC": int(ecology["strong_env_AGC"].sum()),
            "dominant_driver_groups": {
                str(key): int(value) for key, value in ecology["dominant_driver_group"].value_counts().items()
            },
        }
        report["fractions"][fraction] = summary
        outputs.append({"path": str(output), "size_bytes": output.stat().st_size, "sha256": sha256_file(output)})
    group_report = ROOT / "reports" / "environment_groups_validation.json"
    group_passed = all(not any(result.values()) for result in group_validation.values())
    write_json(group_report, {"passed": group_passed, "fractions": group_validation, "groups": groups})
    report["elapsed_seconds"] = time.monotonic() - started
    report_path = ROOT / "reports" / ("agc_ecology_probe.json" if args.max_rows else "agc_ecology.json")
    write_json(report_path, report)
    outputs.extend(
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (report_path, group_report)
    )
    all_rows_match = all(
        value["rows"] == value["matched_agc_stats"] for value in report["fractions"].values()
    )
    p09_validation = {
        "passed": group_passed and all_rows_match,
        "checks": {
            "environment_groups_exact": group_passed,
            "all_rf_agcs_joined_to_stats": all_rows_match,
            "probe_only": args.max_rows is not None,
        },
    }
    write_task_manifest(
        ROOT,
        "P09_PROBE" if args.max_rows else "P09",
        started_at=started_at,
        command=" ".join(sys.argv),
        inputs=[
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": None}
            for path in (args.rf_dir / "rf_FL.parquet", args.rf_dir / "rf_ATT.parquet", args.agc_stats, args.groups)
        ],
        outputs=outputs,
        parameters={"max_rows": args.max_rows, "groups": groups},
        validation=p09_validation,
    )
    write_task_manifest(
        ROOT,
        "P10_PROBE" if args.max_rows else "P10",
        started_at=started_at,
        command=" ".join(sys.argv),
        inputs=[{"path": str(args.groups), "size_bytes": args.groups.stat().st_size, "sha256": sha256_file(args.groups)}],
        outputs=[{"path": str(group_report), "size_bytes": group_report.stat().st_size, "sha256": sha256_file(group_report)}],
        parameters={"importance_prefix": importance_prefix},
        validation={"passed": group_passed, "checks": {"environment_groups_exact": group_passed}},
    )
    print(json.dumps({"report": report, "validation": p09_validation}, indent=2))
    return 0 if p09_validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

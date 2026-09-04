#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.ecology import validate_environment_groups
from polarfunc.environment_drivers import build_driver_labels
from polarfunc.provenance import sha256_file, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build robust ACE environmental-driver targets.")
    parser.add_argument("--groups", type=Path, default=ROOT / "config/environment_groups.yaml")
    parser.add_argument("--ecology-dir", type=Path, default=ROOT / "artifacts/ecology")
    args = parser.parse_args()
    started_at = utc_now()
    config = yaml.safe_load(args.groups.read_text(encoding="utf-8"))
    groups = config["groups"]
    prefix = config.get("importance_prefix", "Importance.")
    outputs: list[Path] = []
    report: dict[str, object] = {"headline_rule": "DRIVER_A", "fractions": {}}
    checks: dict[str, bool] = {}

    for fraction in ["FL", "ATT"]:
        source = args.ecology_dir / f"agc_ecology_{fraction}.parquet"
        schema_columns = pq.ParquetFile(source).schema_arrow.names
        importance_columns = [column for column in schema_columns if column.startswith(prefix)]
        mapping_check = validate_environment_groups(importance_columns, groups)
        columns = ["AGC_ID", "R2_caret", "env_AGC_paper", "strong_env_AGC", *importance_columns]
        frame = pd.read_parquet(source, columns=columns)
        labels = build_driver_labels(frame, groups, importance_prefix=prefix)
        labels.insert(0, "fraction", fraction)
        labels.insert(0, "AGC_ID", frame["AGC_ID"].astype("string"))
        labels["R2_caret"] = frame["R2_caret"].to_numpy()
        labels["env_AGC_paper"] = frame["env_AGC_paper"].to_numpy()
        labels["strong_env_AGC"] = frame["strong_env_AGC"].to_numpy()
        output = args.ecology_dir / f"agc_driver_labels_{fraction}.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite validated output: {output}")
        labels.to_parquet(temporary, index=False, compression="zstd")
        observed_rows = pq.ParquetFile(temporary).metadata.num_rows
        fraction_checks = {
            "all_importance_columns_mapped_once": not any(mapping_check.values()),
            "row_count_preserved": observed_rows == len(frame),
            "agc_ids_unique": labels["AGC_ID"].is_unique,
            "headline_unassigned_only_when_nonpositive": labels.loc[labels.driver_A.isna(), "driver_ambiguous"].all(),
        }
        checks.update({f"{fraction}_{name}": bool(value) for name, value in fraction_checks.items()})
        if not all(fraction_checks.values()):
            raise RuntimeError(f"W2-07 {fraction} validation failure: {fraction_checks}")
        temporary.replace(output)
        outputs.append(output)
        assigned = labels.dropna(subset=["driver_A"])
        agreement = {
            "A_equals_B": float((assigned.driver_A == assigned.driver_B).mean()),
            "A_equals_C": float((assigned.driver_A == assigned.driver_C).mean()),
            "B_equals_C": float((assigned.driver_B == assigned.driver_C).mean()),
        }
        distributions = {
            rule: labels[rule].fillna("UNASSIGNED").value_counts().to_dict()
            for rule in ["driver_A", "driver_B", "driver_C"]
        }
        report["fractions"][fraction] = {
            "rows": len(labels),
            "importance_columns": len(importance_columns),
            "unassigned": int(labels.driver_A.isna().sum()),
            "ambiguous": int(labels.driver_ambiguous.sum()),
            "agreement": agreement,
            "class_distributions": distributions,
            "mapping_validation": mapping_check,
        }

    report["validation"] = {"passed": all(checks.values()), "checks": checks}
    json_path = ROOT / "reports/wave2/driver_label_robustness.json"
    md_path = ROOT / "reports/wave2/driver_label_robustness.md"
    write_json(json_path, report)
    md_path.write_text(
        "# Environmental-driver target robustness\n\n"
        "The frozen Phase-I headline is `DRIVER_A`: the group containing the single largest raw permutation importance.\n\n"
        + "\n".join(
            f"- {fraction}: N={values['rows']:,}; variables={values['importance_columns']}; "
            f"A/B agreement={values['agreement']['A_equals_B']:.3f}; A/C agreement={values['agreement']['A_equals_C']:.3f}; "
            f"ambiguous={values['ambiguous']:,}."
            for fraction, values in report["fractions"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.extend([json_path, md_path])
    input_paths = [args.groups, args.ecology_dir / "agc_ecology_FL.parquet", args.ecology_dir / "agc_ecology_ATT.parquet"]
    manifest = {
        "task_id": "W2-07", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": 1,
        "inputs": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in input_paths],
        "parameters": {"headline_rule": "DRIVER_A", "importance_prefix": prefix},
        "outputs": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "validation": report["validation"], "notes": ["All nonpositive rows remain unassigned."],
    }
    write_json(ROOT / "manifests/wave2/W2-07/manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

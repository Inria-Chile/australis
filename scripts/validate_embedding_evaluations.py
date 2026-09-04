#!/usr/bin/env python3
"""Validate completed GenomeOcean evaluations and published reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evaluation(directory: Path, report_copy: Path) -> dict[str, object]:
    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "PASS":
        raise ValueError(f"Evaluation did not pass: {directory}")
    if report.get("model_label") != "GenomeOcean":
        raise ValueError(f"Incorrect model label: {directory}")
    if not (directory / "DONE").is_file() or (directory / ".incomplete").exists():
        raise ValueError(f"Completion markers are inconsistent: {directory}")
    for name, expected in report["outputs"].items():
        actual = sha256(directory / name)
        if actual != expected:
            raise ValueError(f"Hash mismatch for {directory / name}: {actual} != {expected}")
    for name in ("results_long.parquet", "model_selection.parquet"):
        frame = pd.read_parquet(directory / name)
        if not frame.representation.astype(str).str.startswith("GenomeOcean_").all():
            raise ValueError(f"Incorrect representation labels in {directory / name}")
    summary = pd.read_csv(directory / "summary.tsv", sep="\t")
    if not summary.representation.astype(str).str.startswith("GenomeOcean_").all():
        raise ValueError(f"Incorrect summary labels: {directory}")
    if report_copy.read_bytes() != report_path.read_bytes():
        raise ValueError(f"Published report differs from evaluation: {report_copy}")
    return {
        "dataset": report["dataset"],
        "rows": report["embedded_rows"],
        "dimension": report["native_dimension"],
        "outputs": len(report["outputs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--extended-dir", type=Path, required=True)
    parser.add_argument("--core-report", type=Path, required=True)
    parser.add_argument("--extended-report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()

    evaluations = [
        validate_evaluation(args.core_dir, args.core_report),
        validate_evaluation(args.extended_dir, args.extended_report),
    ]
    comparison = pd.read_csv(args.comparison, sep="\t")
    if len(comparison) != 24:
        raise ValueError(f"Expected 24 comparison rows, found {len(comparison)}")
    if comparison.duplicated(["scope", "representation_type", "task"]).any():
        raise ValueError("Duplicate comparison rows")
    result = {
        "status": "PASS",
        "evaluations": evaluations,
        "comparison_rows": len(comparison),
        "genomeocean_auroc_wins": int((comparison.delta_AUROC_mean > 0).sum()),
        "genomeocean_auprc_wins": int((comparison.delta_AUPRC_mean > 0).sum()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

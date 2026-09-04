#!/usr/bin/env python3
"""Relabel a completed embedding evaluation without changing its metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


OUTPUTS = (
    "results_long.parquet",
    "model_selection.parquet",
    "null_controls.parquet",
    "summary.tsv",
    "null_summary.tsv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_prefix(series: pd.Series, old: str, new: str) -> pd.Series:
    values = series.astype(str)
    unexpected = values[~values.str.startswith(f"{old}_")].unique().tolist()
    if unexpected:
        raise ValueError(f"Unexpected representation labels: {unexpected}")
    return values.str.replace(f"{old}_", f"{new}_", n=1, regex=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-label", required=True)
    parser.add_argument("--to-label", required=True)
    args = parser.parse_args()

    directory = args.output_dir.resolve()
    if not (directory / "DONE").is_file() or (directory / ".incomplete").exists():
        raise ValueError(f"Evaluation is not complete: {directory}")

    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_text())
    before = {name: sha256(directory / name) for name in OUTPUTS}
    if report.get("outputs") != before:
        raise ValueError("Recorded output hashes do not match files before relabeling")

    results = pd.read_parquet(directory / "results_long.parquet")
    selections = pd.read_parquet(directory / "model_selection.parquet")
    summary = pd.read_csv(directory / "summary.tsv", sep="\t")
    for frame in (results, selections, summary):
        frame["representation"] = replace_prefix(
            frame["representation"], args.from_label, args.to_label
        )

    results.to_parquet(directory / "results_long.parquet", index=False, compression="zstd")
    selections.to_parquet(directory / "model_selection.parquet", index=False, compression="zstd")
    summary.to_csv(directory / "summary.tsv", sep="\t", index=False)

    null_summary = pd.read_csv(directory / "null_summary.tsv", sep="\t")
    report["model_label"] = args.to_label
    report["outputs"] = {name: sha256(directory / name) for name in OUTPUTS}
    report["label_normalization"] = {
        "from": args.from_label,
        "to": args.to_label,
        "metrics_recomputed": False,
        "before_sha256": before,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (directory / "evaluation.md").write_text(
        f"# {args.to_label} evaluation: {report['dataset']}\n\n"
        f"Status: **{report['status']}**\n\n"
        "The frozen protocol uses train-only scaling/PCA, validation-only C "
        "selection, five retained seeds, and separate FL/ATT cohorts.\n\n"
        + summary.to_csv(sep="\t", index=False)
        + "\n## Null controls\n\n"
        + null_summary.to_csv(sep="\t", index=False)
    )

    after = {name: sha256(directory / name) for name in OUTPUTS}
    if report["outputs"] != after:
        raise RuntimeError("Output hash update failed")
    print(json.dumps({"status": "PASS", "directory": str(directory), "outputs": after}, indent=2))


if __name__ == "__main__":
    main()

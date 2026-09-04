#!/usr/bin/env python3
"""Compare frozen ESM2 and GenomeOcean evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "AUROC_mean",
    "AUPRC_mean",
    "MCC_mean",
    "balanced_accuracy_mean",
    "F1_mean",
    "Brier_mean",
)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_summary(path: Path, model: str, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame["model"] = model
    frame["scope"] = scope
    frame["representation_type"] = frame.representation.str.extract(r"_(native|PCA256)$")[0]
    if frame.representation_type.isna().any():
        raise ValueError(f"Unexpected representation labels in {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm2-core", type=Path, required=True)
    parser.add_argument("--esm2-extended", type=Path, required=True)
    parser.add_argument("--genomeocean-core", type=Path, required=True)
    parser.add_argument("--genomeocean-extended", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    frames = [
        load_summary(args.esm2_core, "ESM2", "core"),
        load_summary(args.esm2_extended, "ESM2", "extended"),
        load_summary(args.genomeocean_core, "GenomeOcean", "core"),
        load_summary(args.genomeocean_extended, "GenomeOcean", "extended"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    keys = ["scope", "representation_type", "task"]
    duplicate = combined.duplicated(keys + ["model"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate model/scope/representation/task rows")

    wide = combined.pivot(index=keys, columns="model", values=list(METRICS))
    expected = {(metric, model) for metric in METRICS for model in ("ESM2", "GenomeOcean")}
    if set(wide.columns) != expected:
        raise ValueError("Comparison matrix is incomplete")

    rows = []
    for index, values in wide.iterrows():
        row = dict(zip(keys, index))
        for metric in METRICS:
            esm2 = float(values[(metric, "ESM2")])
            genomeocean = float(values[(metric, "GenomeOcean")])
            row[f"ESM2_{metric}"] = esm2
            row[f"GenomeOcean_{metric}"] = genomeocean
            row[f"delta_{metric}"] = genomeocean - esm2
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)

    output_tsv = args.output_prefix.with_suffix(".tsv")
    output_md = args.output_prefix.with_suffix(".md")
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_tsv, sep="\t", index=False)

    auroc_wins = int((comparison.delta_AUROC_mean > 0).sum())
    auprc_wins = int((comparison.delta_AUPRC_mean > 0).sum())
    total = len(comparison)
    display = comparison[
        keys
        + [
            "ESM2_AUROC_mean",
            "GenomeOcean_AUROC_mean",
            "delta_AUROC_mean",
            "ESM2_AUPRC_mean",
            "GenomeOcean_AUPRC_mean",
            "delta_AUPRC_mean",
        ]
    ].copy()
    for column in display.columns[3:]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    best = comparison.loc[comparison.delta_AUROC_mean.idxmax()]
    worst = comparison.loc[comparison.delta_AUROC_mean.idxmin()]
    output_md.write_text(
        "# ESM2 versus GenomeOcean frozen evaluation\n\n"
        "Both models use the same frozen splits, tasks, validation-only C selection, "
        "train-only scaling/PCA, seeds, and null-control protocol. Positive deltas "
        "favor GenomeOcean.\n\n"
        f"- GenomeOcean AUROC wins: {auroc_wins}/{total}\n"
        f"- GenomeOcean AUPRC wins: {auprc_wins}/{total}\n"
        f"- Largest AUROC gain: {best.delta_AUROC_mean:+.4f} "
        f"({best.scope}, {best.representation_type}, {best.task})\n"
        f"- Largest AUROC loss: {worst.delta_AUROC_mean:+.4f} "
        f"({worst.scope}, {worst.representation_type}, {worst.task})\n\n"
        + markdown_table(display)
        + "\n"
    )
    print(json.dumps({"status": "PASS", "rows": total, "auroc_wins": auroc_wins, "auprc_wins": auprc_wins}, indent=2))


if __name__ == "__main__":
    main()

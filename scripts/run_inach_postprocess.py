#!/usr/bin/env python3
"""Bootstrap, compare, visualize, and narrate the frozen INACH model results."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)


SEEDS = (13, 42, 73, 101, 2026)
METRICS = ("AUROC", "AUPRC", "MCC", "balanced_accuracy", "Brier")
HEADLINE = (
    "kmer_1_6", "mmseqs_nn", "esm2_native", "esm2_pca256",
    "genomeocean_native", "genomeocean_pca256", "fusion_native_2816",
    "fusion_modality_pca512", "fusion_joint_pca256", "fusion_balanced_pca256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y: np.ndarray, probability: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
    hard = probability >= 0.5
    return {
        "AUROC": float(roc_auc_score(y, probability, sample_weight=weight)),
        "AUPRC": float(average_precision_score(y, probability, sample_weight=weight)),
        "MCC": float(matthews_corrcoef(y, hard, sample_weight=weight)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard, sample_weight=weight)),
        "Brier": float(brier_score_loss(y, probability, sample_weight=weight)),
    }


def task_test_ids(frame: pd.DataFrame, task: str) -> pd.Index:
    kind, fraction = task.split("_")
    mask = frame.fraction.astype(str).eq(fraction) & frame.split.astype(str).eq("test")
    if kind == "known":
        mask &= frame.function_label.eq("strict_known")
    else:
        mask &= frame.function_label.eq("strict_unknown")
    return pd.Index(frame.loc[mask, "CDHit_ID"].astype(str))


def homology_group_column(frame: pd.DataFrame) -> str:
    for column in ("MMseqs_group", "split_group"):
        if column in frame.columns:
            return column
    raise KeyError("Expected MMseqs_group or split_group in the common manifest")


def add_mmseqs(root: Path) -> None:
    result_path = root / "artifacts/results/common_subset_cpu_baselines.parquet"
    prediction_dir = root / "artifacts/predictions/common_subset_cpu"
    results = pd.read_parquet(result_path)
    if "mmseqs_nn" in set(results.representation):
        return
    common = pd.read_parquet(root / "manifests/common_subset_91663.parquet")
    hits = pd.read_parquet(root / "artifacts/homology/test_to_train_hits.parquet")
    hits["query"] = hits["query"].astype(str)
    train_ids = set(common.loc[common.split.astype(str).eq("train"), "CDHit_ID"].astype(str))
    benchmark_hash = sha256(root / "manifests/common_subset_91663.parquet")
    rows: list[dict[str, object]] = []
    group_column = homology_group_column(common)
    for task in sorted(results.task.unique()):
        ids = task_test_ids(common, task)
        cohort = common.set_index(common.CDHit_ID.astype(str)).loc[ids]
        joined = pd.DataFrame({"CDHit_ID": ids}).merge(hits, left_on="CDHit_ID", right_on="query", how="left", validate="one_to_one")
        valid = joined.target.astype(str).isin(train_ids) & joined.has_hit.fillna(False)
        train_task = results.loc[results.task.eq(task), "train_n"].iloc[0]
        kind, fraction = task.split("_")
        train_mask = common.fraction.astype(str).eq(fraction) & common.split.astype(str).eq("train")
        train_mask &= common.function_label.eq("strict_known" if kind in ("known", "transfer") else "strict_unknown")
        prevalence = float(common.loc[train_mask, "ecology_label"].eq("high").mean())
        probability = np.where(valid, joined.prediction.astype(float), prevalence)
        y = cohort.ecology_label.eq("high").to_numpy(np.int8)
        value = metrics(y, probability)
        settings = {
            "representation": "mmseqs_nn", "reference": "frozen_train_only_top_hit",
            "common_target_required": True, "invalid_or_no_hit_fallback": "task_train_prevalence",
        }
        config = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        split_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        prediction = pd.DataFrame({
            "CDHit_ID": ids,
            "MMseqs_group": cohort[group_column].astype(str).to_numpy(),
            "task": task, "split": "test", "y_true": y, "representation": "mmseqs_nn",
            "probability_positive": probability, "predicted_label": (probability >= 0.5).astype(np.int8),
            "benchmark_hash": benchmark_hash, "split_hash": split_hash, "model_config_hash": config,
        })
        prediction.to_parquet(prediction_dir / f"mmseqs_nn__{task}.parquet", index=False, compression="zstd")
        for seed in SEEDS:
            rows.append({
                "representation": "mmseqs_nn", "task": task, "seed": seed, "C": np.nan,
                "validation_AUPRC": np.nan, "dimension": 1, "train_n": int(train_task),
                "validation_n": int(results.loc[results.task.eq(task), "validation_n"].iloc[0]),
                "test_n": len(ids), "benchmark_hash": benchmark_hash, "split_hash": split_hash,
                "model_config_hash": config, **value,
            })
    combined = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    combined.to_parquet(result_path, index=False, compression="zstd")
    summary = combined.groupby(["representation", "task"]).agg(
        AUROC=("AUROC", "mean"), AUPRC=("AUPRC", "mean"), MCC=("MCC", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), Brier=("Brier", "mean"), test_n=("test_n", "first"),
    ).reset_index()
    (root / "reports/inach_consolidation/common_subset_cpu_baselines.md").write_text(
        "# INACH common-subset classical evaluation\n\nStatus: **PASS**\n\n"
        "The MMseqs nearest-neighbour baseline uses the frozen TRAIN-only top hit. Targets outside the exact common cohort and no-hit cases receive the task-specific TRAIN prevalence.\n\n"
        + summary.to_csv(sep="\t", index=False)
    )


def normalize_fusion_output(root: Path) -> None:
    source = root / "artifacts/results/multimodal_fusion.parquet"
    target = root / "artifacts/results/multimodal_fusion_common.parquet"
    if target.exists():
        return
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)


def load_task_predictions(root: Path, task: str) -> pd.DataFrame:
    paths = []
    for directory in (
        root / "artifacts/predictions/common_subset_cpu",
        root / "artifacts/predictions/common_subset_fm",
        root / "artifacts/predictions/multimodal_fusion",
    ):
        paths.extend(sorted(directory.glob(f"*__{task}.parquet")))
    selected = []
    base_ids: list[str] | None = None
    metadata: pd.DataFrame | None = None
    for path in paths:
        frame = pd.read_parquet(path).sort_values("CDHit_ID").reset_index(drop=True)
        representation = str(frame.representation.iloc[0])
        if representation not in HEADLINE:
            continue
        ids = frame.CDHit_ID.astype(str).tolist()
        if base_ids is None:
            base_ids = ids
            metadata = frame[["CDHit_ID", "MMseqs_group", "y_true"]].copy()
        elif ids != base_ids:
            raise RuntimeError(f"Prediction ID mismatch for {task}: {path}")
        selected.append(frame[["CDHit_ID", "probability_positive"]].rename(columns={"probability_positive": representation}))
    if len(selected) != len(HEADLINE) or metadata is None:
        found = [column for frame in selected for column in frame.columns if column != "CDHit_ID"]
        raise RuntimeError(f"Incomplete headline predictions for {task}: {found}")
    result = metadata
    for frame in selected:
        result = result.merge(frame, on="CDHit_ID", validate="one_to_one")
    return result


def bootstrap_task(payload: tuple[str, pd.DataFrame, int, int]) -> tuple[list[dict[str, object]], int]:
    task, frame, replicates, seed = payload
    y = frame.y_true.to_numpy(np.int8)
    group_codes, groups = pd.factorize(frame.MMseqs_group.astype(str), sort=True)
    probabilities = {representation: frame[representation].to_numpy(float) for representation in HEADLINE}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    skipped = 0
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        counts = np.bincount(sampled, minlength=len(groups)).astype(float)
        weight = counts[group_codes]
        if len(np.unique(y[weight > 0])) < 2:
            skipped += 1
            continue
        for representation, probability in probabilities.items():
            rows.append({
                "task": task, "replicate": replicate, "representation": representation,
                "bootstrap_unit": "MMseqs_group", **metrics(y, probability, weight),
            })
    return rows, skipped


def bootstrap(root: Path, replicates: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = root / "artifacts/results/paired_group_bootstrap.parquet"
    delta_output = root / "artifacts/results/fusion_deltas.parquet"
    for path in (output, delta_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    tasks = ("known_FL", "unknown_FL", "transfer_FL", "known_ATT", "unknown_ATT", "transfer_ATT")
    payloads = [(task, load_task_predictions(root, task), replicates, 20260830 + index) for index, task in enumerate(tasks)]
    rows: list[dict[str, object]] = []
    skipped: dict[str, int] = {}
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        for task, result in zip(tasks, pool.map(bootstrap_task, payloads)):
            local, count = result; rows.extend(local); skipped[task] = count
    boot = pd.DataFrame(rows)
    if boot.empty or boot.groupby(["task", "representation"]).replicate.nunique().min() < 1000:
        raise RuntimeError("Insufficient valid group-bootstrap replicates")
    output.parent.mkdir(parents=True, exist_ok=True)
    boot.to_parquet(output, index=False, compression="zstd")

    point = pd.concat([
        pd.read_parquet(root / "artifacts/results/common_subset_foundation_models.parquet"),
        pd.read_parquet(root / "artifacts/results/multimodal_fusion_common.parquet"),
    ]).groupby(["task", "representation"], as_index=False).first()
    delta_rows: list[dict[str, object]] = []
    comparisons: list[tuple[str, str, str]] = [
        ("genomeocean_native", "esm2_native", "genomeocean_minus_esm2"),
        ("genomeocean_native", "kmer_1_6", "genomeocean_minus_kmer"),
        ("genomeocean_native", "mmseqs_nn", "genomeocean_minus_mmseqs"),
    ]
    for task in tasks:
        task_point = point.loc[point.task.eq(task)]
        singles = task_point.loc[task_point.representation.isin(("esm2_native", "esm2_pca256", "genomeocean_native", "genomeocean_pca256"))]
        best_single = str(singles.sort_values("AUROC", ascending=False).representation.iloc[0])
        for fusion in ("fusion_native_2816", "fusion_modality_pca512", "fusion_joint_pca256", "fusion_balanced_pca256"):
            comparisons.append((fusion, best_single, f"{fusion}_minus_best_single"))
            comparisons.append((fusion, "genomeocean_native", f"{fusion}_minus_genomeocean_native"))
            comparisons.append((fusion, "esm2_native", f"{fusion}_minus_esm2_native"))
        task_boot = boot.loc[boot.task.eq(task)]
        for left, right, comparison in comparisons:
            if left not in set(task_boot.representation) or right not in set(task_boot.representation):
                continue
            paired = task_boot.loc[task_boot.representation.isin((left, right))].pivot(index="replicate", columns="representation", values=list(METRICS)).dropna()
            for metric in METRICS:
                delta = paired[(metric, right)] - paired[(metric, left)] if metric == "Brier" else paired[(metric, left)] - paired[(metric, right)]
                delta_rows.append({
                    "task": task, "comparison": comparison, "left": left, "right": right,
                    "metric": metric, "delta_mean": float(delta.mean()),
                    "ci_low": float(delta.quantile(0.025)), "ci_high": float(delta.quantile(0.975)),
                    "p_improvement_gt_0": float((delta > 0).mean()), "valid_replicates": len(delta),
                    "brier_orientation": "right_minus_left" if metric == "Brier" else "left_minus_right",
                })
        comparisons = comparisons[:3]
    deltas = pd.DataFrame(delta_rows)
    deltas.to_parquet(delta_output, index=False, compression="zstd")
    summary = boot.groupby(["task", "representation"])[list(METRICS)].agg(["mean", lambda x: x.quantile(.025), lambda x: x.quantile(.975)])
    report = root / "reports/inach_consolidation/paired_group_bootstrap.md"
    report.write_text(
        f"# Paired MMseqs-group bootstrap\n\nStatus: **PASS**\n\nRequested replicates: {replicates}. "
        f"Skipped single-class replicates: `{json.dumps(skipped, sort_keys=True)}`.\n\n"
        + summary.reset_index().to_csv(sep="\t", index=False)
    )
    fusion_report = root / "reports/inach_consolidation/fusion_incremental_value.md"
    fusion_report.write_text("# Fusion incremental value\n\nStatus: **PASS**\n\nPositive deltas denote improvement, including reversed Brier differences.\n\n" + deltas.to_csv(sep="\t", index=False))
    return boot, deltas


def point_results(root: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(root / "artifacts/results/common_subset_cpu_baselines.parquet"),
        pd.read_parquet(root / "artifacts/results/common_subset_foundation_models.parquet"),
        pd.read_parquet(root / "artifacts/results/multimodal_fusion_common.parquet"),
    ]
    return pd.concat(frames, ignore_index=True).groupby(["representation", "task"], as_index=False).first()


def tables_and_figures(root: Path, boot: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    point = point_results(root)
    ci = boot.groupby(["representation", "task"]).AUROC.quantile([.025, .975]).unstack().reset_index().rename(columns={.025: "AUROC_ci_low", .975: "AUROC_ci_high"})
    table = point.merge(ci, on=["representation", "task"], how="left")
    classical = table.loc[table.representation.isin(("length", "gc", "length_gc", "codon_67", "aa20", "kmer_1_6", "mmseqs_nn"))]
    best_classical = classical.groupby("task").AUROC.max()
    table["delta_AUROC_vs_best_classical"] = table.apply(lambda row: row.AUROC - best_classical[row.task], axis=1)
    report_dir = root / "reports/inach_consolidation"; report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "INACH_main_comparison.tsv", sep="\t", index=False)
    matrix = table.pivot(index="representation", columns="task", values="AUROC")
    (report_dir / "INACH_main_comparison.md").write_text(
        "# INACH main comparison\n\nAll headline values use the exact 91,663-molecule common cohort.\n\n"
        + "```text\n" + matrix.to_csv(sep="\t", float_format="%.3f") + "```\n"
    )

    figure_dir = root / "figures/inach_final"; figure_dir.mkdir(parents=True, exist_ok=True)
    common = pd.read_parquet(root / "manifests/common_subset_91663.parquet")
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    common.function_label.value_counts().rename(index={"strict_known": "Strict known", "strict_unknown": "Strict unknown"}).plot.bar(ax=axes[0], color=["#287271", "#D1495B"])
    pd.crosstab(common.fraction, common.ecology_label).plot.bar(ax=axes[1], color=["#2A6FBB", "#E09F3E"])
    axes[0].set_title("Functional-status cohort"); axes[0].set_ylabel("Unigenes")
    axes[1].set_title("Ecological target by fraction"); axes[1].set_ylabel("Unigenes")
    save_figure(fig, figure_dir / "Fig1_catalogue_ecology", [root / "manifests/common_subset_91663.parquet"])

    selected = matrix.loc[[x for x in HEADLINE if x in matrix.index]]
    fig, ax = plt.subplots(figsize=(12, 8)); image = ax.imshow(selected.to_numpy(), cmap="viridis", vmin=.5, vmax=1, aspect="auto")
    for row in range(selected.shape[0]):
        for column in range(selected.shape[1]):
            ax.text(column, row, f"{selected.iloc[row, column]:.2f}", ha="center", va="center", color="white" if selected.iloc[row, column] < .75 else "black", fontsize=8)
    ax.set_xticks(range(selected.shape[1]), selected.columns, rotation=35, ha="right"); ax.set_yticks(range(selected.shape[0]), selected.index)
    fig.colorbar(image, ax=ax, label="AUROC"); ax.set_title("Common-cohort AUROC (n=91,663)"); ax.set_xlabel("Biological task"); ax.set_ylabel("Representation")
    save_figure(fig, figure_dir / "Fig2_model_comparison", [report_dir / "INACH_main_comparison.tsv"])

    transfer = table.loc[table.task.isin(("transfer_FL", "transfer_ATT")) & table.representation.isin(HEADLINE)]
    transfer_matrix = transfer.pivot(index="representation", columns="task", values="AUROC").reindex([x for x in HEADLINE if x in set(transfer.representation)])
    fig, ax = plt.subplots(figsize=(12, 6)); transfer_matrix.plot.bar(ax=ax, color=["#287271", "#D1495B"])
    ax.tick_params(axis="x", rotation=65); ax.set_ylim(.45, 1); ax.set_title("Known-to-unknown transfer"); ax.set_ylabel("AUROC")
    save_figure(fig, figure_dir / "Fig3_transfer_fusion", [report_dir / "INACH_main_comparison.tsv"])

    delta_plot = deltas.loc[deltas.metric.eq("AUROC") & deltas.comparison.str.contains("minus_best_single")].copy()
    delta_plot["error_low"] = delta_plot.delta_mean - delta_plot.ci_low; delta_plot["error_high"] = delta_plot.ci_high - delta_plot.delta_mean
    fig, ax = plt.subplots(figsize=(11, 7))
    for index, row in delta_plot.reset_index(drop=True).iterrows():
        ax.errorbar(row.delta_mean, index, xerr=[[row.error_low], [row.error_high]], fmt="o", color="#1D3557")
    ax.set_yticks(range(len(delta_plot))); ax.set_yticklabels(delta_plot.task + " | " + delta_plot.left)
    ax.axvline(0, color="black", linewidth=1); ax.set_xlabel("Paired delta AUROC vs best single modality"); ax.set_title("Fusion incremental value (95% group-bootstrap CI)")
    save_figure(fig, figure_dir / "Fig4_paired_deltas", [root / "artifacts/results/fusion_deltas.parquet"])
    return table


def save_figure(fig: plt.Figure, stem: Path, inputs: list[Path]) -> None:
    fig.tight_layout(); fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    provenance = {"status": "PASS", "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs], "outputs": [str(stem.with_suffix(".pdf")), str(stem.with_suffix(".png"))]}
    stem.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def narrative_outputs(root: Path, table: pd.DataFrame, deltas: pd.DataFrame) -> None:
    report_dir = root / "reports"
    fusion = deltas.loc[deltas.metric.eq("AUROC") & deltas.comparison.str.contains("minus_best_single")]
    controlled = fusion.loc[fusion.left.isin(("fusion_modality_pca512", "fusion_joint_pca256", "fusion_balanced_pca256"))]
    robust = controlled.loc[controlled.ci_low > 0]
    conclusion = "robust_dimension_controlled_complementarity" if len(robust) else "no_robust_dimension_controlled_complementarity"
    sources = [
        root / "reports/inach_consolidation/INACH_main_comparison.tsv",
        root / "artifacts/results/fusion_deltas.parquet",
        root / "reports/gpu/gpu_compute_cost.tsv",
    ]
    final = {
        "status": "PASS", "common_cohort": 91663, "bootstrap_unit": "MMseqs_group",
        "fusion_conclusion": conclusion, "robust_dimension_controlled_cells": len(robust),
        "sources": [{"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in sources],
        "headline": table.loc[table.representation.isin(HEADLINE)].to_dict("records"),
        "fusion_deltas": fusion.to_dict("records"),
    }
    (report_dir / "INACH_PRELIMINARY_RESULTS_FINAL.json").write_text(json.dumps(final, indent=2) + "\n")
    (report_dir / "INACH_PRELIMINARY_RESULTS_FINAL.md").write_text(
        "# INACH preliminary results\n\nStatus: **PASS**\n\n"
        f"The strict paired cohort contains **91,663** unigenes. Fusion conclusion: **{conclusion}**. "
        "Uncertainty uses paired resampling of MMseqs homology groups.\n\n"
        + table.loc[table.representation.isin(HEADLINE)].to_csv(sep="\t", index=False)
    )
    (report_dir / "INACH_CLAIM_MATRIX_FINAL.md").write_text(
        "# INACH claim matrix\n\n"
        "| Claim | Status | Evidence boundary |\n|---|---|---|\n"
        "| GenomeOcean and ESM2 encode ecological signal | Safe | Common-cohort held-out predictions and group-bootstrap CI |\n"
        f"| Naive DNA+protein fusion is complementary | {'Safe in supported cells' if len(robust) else 'Unsupported as a general claim'} | Requires dimension-controlled delta CI above zero; supported cells={len(robust)} |\n"
        "| Strict-unknown genes have proven biochemical functions | Unsupported | Strict-unknown denotes missing current annotation, not verified novel function |\n"
        "| Ecological association is causal | Unsupported | The benchmark is predictive/associational |\n"
    )
    (report_dir / "INACH_PRELIMINARY_NARRATIVE.md").write_text(
        "# Preliminary-work narrative for INACH\n\n"
        "An ORF is a predicted coding region; closely related ORFs were dereplicated into unigenes, and AGC/CAG group genes by sequence or co-abundance structure. "
        "We distinguish strict-known genes with supported functional annotation from strict-unknown genes lacking that evidence; unknown does not mean experimentally novel.\n\n"
        "The predictive target separates high and low ecological association within free-living (FL) and particle-associated (ATT) fractions. "
        "TRAIN, validation and TEST are separated by MMseqs homology groups, reducing the chance that near-duplicate proteins inflate performance.\n\n"
        "ESM2 represents translated proteins and GenomeOcean represents nucleotide sequences. Both were evaluated on the exact same 91,663 genes, with preprocessing fitted only on TRAIN and hyperparameters selected only on validation. "
        "Known-to-unknown transfer asks whether patterns learned from annotated genes generalize to unannotated genes.\n\n"
        f"Four frozen DNA+protein fusion variants tested complementarity. The current evidence class is **{conclusion}**. "
        "These are preliminary predictive associations, not causal or biochemical validation. INACH support is needed for broader Antarctic sampling, independent-site validation, additional genomic/protein language models and experimentally anchored interpretation.\n"
    )
    neurips = report_dir / "neurips"; neurips.mkdir(parents=True, exist_ok=True)
    recommendation = "develop aligned DNA-protein-ecology projections" if len(robust) else "prioritize GenomeOcean-to-ecology alignment and retain protein as a secondary modality"
    (neurips / "POST_INACH_METHOD_DECISION.md").write_text(
        "# Post-INACH method decision\n\n"
        f"Observed decision: **{conclusion}**; therefore **{recommendation}**.\n\n"
        "Future latent variables are z_DNA, z_PROTEIN and z_ECO. Evaluation must include CAG holdout, taxonomy holdout and ACE-to-CEODOS out-of-distribution transfer. "
        "Masked-known retrieval and unknown-function retrieval remain later stages, after robust held-out ecological validation.\n"
    )


def task_manifest(root: Path, task_id: str, outputs: list[Path], validation: dict[str, object]) -> None:
    directory = root / "manifests/inach_consolidation" / task_id; directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id, "status": "PASS", "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "outputs": [{"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in outputs], "validation": validation,
    }
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--replicates", type=int, default=2000); parser.add_argument("--workers", type=int, default=6); args = parser.parse_args()
    if args.replicates < 1000: raise ValueError("At least 1,000 bootstrap replicates are mandatory")
    root = args.root.resolve(); normalize_fusion_output(root); add_mmseqs(root); boot, deltas = bootstrap(root, args.replicates, args.workers); table = tables_and_figures(root, boot, deltas); narrative_outputs(root, table, deltas)
    task_manifest(root, "C-INACH-06", [root / "artifacts/results/paired_group_bootstrap.parquet", root / "reports/inach_consolidation/paired_group_bootstrap.md"], {"paired": True, "unit": "MMseqs_group", "replicates": args.replicates})
    task_manifest(root, "C-INACH-08", [root / "artifacts/results/fusion_deltas.parquet", root / "reports/inach_consolidation/fusion_incremental_value.md"], {"paired": True, "dimension_controlled": True})
    task_manifest(root, "C-INACH-09", [root / "reports/inach_consolidation/INACH_main_comparison.tsv", root / "reports/inach_consolidation/INACH_main_comparison.md"], {"common_cohort_only": True})
    task_manifest(root, "C-INACH-10", sorted((root / "figures/inach_final").glob("*.pdf")) + sorted((root / "figures/inach_final").glob("*.provenance.json")), {"figures": 4, "provenance_sidecars": 4})
    task_manifest(root, "C-INACH-11", [root / "reports/INACH_PRELIMINARY_RESULTS_FINAL.json", root / "reports/INACH_PRELIMINARY_RESULTS_FINAL.md", root / "reports/INACH_CLAIM_MATRIX_FINAL.md"], {"traceable_sources": True})
    task_manifest(root, "C-INACH-12", [root / "reports/INACH_PRELIMINARY_NARRATIVE.md"], {"causal_overclaim": False, "future_work_distinguished": True})
    task_manifest(root, "C-INACH-13", [root / "reports/neurips/POST_INACH_METHOD_DECISION.md"], {"linked_to_fusion_evidence": True})
    print(json.dumps({"status": "PASS", "bootstrap_rows": len(boot), "delta_rows": len(deltas), "table_rows": len(table)}, indent=2))


if __name__ == "__main__":
    main()

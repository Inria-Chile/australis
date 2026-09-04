#!/usr/bin/env python3
"""Leakage-aware linear-probe evaluation for sharded sequence embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SEEDS = (13, 42, 73, 101, 2026)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_hash(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    hard = probabilities >= 0.5
    return {
        "AUROC": float(roc_auc_score(y, probabilities)),
        "AUPRC": float(average_precision_score(y, probabilities)),
        "MCC": float(matthews_corrcoef(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "F1": float(f1_score(y, hard, zero_division=0)),
        "Brier": float(brier_score_loss(y, probabilities)),
    }


def estimator(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=4000,
        random_state=seed,
    )


def load_embeddings(directories: list[Path]) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    arrays: list[np.ndarray] = []
    for directory in directories:
        for archive in sorted(directory.glob("shard_*.npz")):
            data = np.load(archive, allow_pickle=False)
            shard_ids = data["ids"].astype(str).tolist()
            shard_embeddings = data["embeddings"].astype(np.float32, copy=False)
            if len(shard_ids) != len(shard_embeddings):
                raise ValueError(f"Shape mismatch in {archive}")
            identifiers.extend(shard_ids)
            arrays.append(shard_embeddings)
    if not arrays:
        raise ValueError("No embedding shards found")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate embedding IDs")
    matrix = np.concatenate(arrays)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding matrix")
    return identifiers, matrix


def task_masks(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    masks = {}
    for fraction in ("FL", "ATT"):
        in_fraction = frame.fraction.astype(str).eq(fraction).to_numpy()
        known = frame.function_label.eq("strict_known").to_numpy()
        unknown = frame.function_label.eq("strict_unknown").to_numpy()
        masks[f"known_{fraction}"] = (in_fraction & known,) * 3
        masks[f"unknown_{fraction}"] = (in_fraction & unknown,) * 3
        masks[f"known_to_unknown_{fraction}"] = (
            in_fraction & known,
            in_fraction & known,
            in_fraction & unknown,
        )
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-label", default="ESM2")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    benchmark = pd.read_parquet(args.benchmark)
    embedding_ids, embeddings = load_embeddings(args.embedding_dir)
    position = {identifier: index for index, identifier in enumerate(embedding_ids)}
    available = benchmark.CDHit_ID.astype(str).isin(position)
    frame = benchmark.loc[available].copy().reset_index(drop=True)
    embeddings = embeddings[[position[value] for value in frame.CDHit_ID.astype(str)]]
    y = frame.ecology_label.eq("high").to_numpy(np.int8)
    split = frame.split.astype(str).to_numpy()
    masks = task_masks(frame)

    counts = {}
    for task, (train_mask, validation_mask, test_mask) in masks.items():
        counts[task] = {}
        for split_name, mask in (
            ("train", train_mask),
            ("validation", validation_mask),
            ("test", test_mask),
        ):
            indices = np.flatnonzero((split == split_name) & mask)
            classes = sorted(np.unique(y[indices]).tolist())
            counts[task][split_name] = {"n": len(indices), "classes": classes}
            if len(indices) == 0 or classes != [0, 1]:
                raise ValueError(f"Task {task} has invalid {split_name} cohort: {counts[task][split_name]}")

    audit = {
        "status": "PASS",
        "dataset": args.dataset_name,
        "benchmark_rows": len(benchmark),
        "embedded_rows": len(frame),
        "context_excluded_rows": int((~available).sum()),
        "native_dimension": int(embeddings.shape[1]),
        "embedding_ids_unique": len(position) == len(embedding_ids),
        "counts": counts,
    }
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / ".incomplete").write_text("evaluation in progress\n")
    models_dir = args.output_dir / "pca_models"
    models_dir.mkdir()

    result_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    pca_reports: list[dict[str, object]] = []

    for task, (train_mask, validation_mask, test_mask) in masks.items():
        train = np.flatnonzero((split == "train") & train_mask)
        validation = np.flatnonzero((split == "validation") & validation_mask)
        test = np.flatnonzero((split == "test") & test_mask)
        cohorts = {"train": train, "validation": validation, "test": test}
        train_ids = frame.iloc[train].CDHit_ID.astype(str).tolist()

        native_scaler = StandardScaler().fit(embeddings[train])
        native = {name: native_scaler.transform(embeddings[index]) for name, index in cohorts.items()}

        pca_scaler = StandardScaler().fit(embeddings[train])
        scaled_train = pca_scaler.transform(embeddings[train])
        components = min(256, embeddings.shape[1], len(train) - 1)
        pca = PCA(n_components=components, svd_solver="randomized", random_state=2026)
        pca_train = pca.fit_transform(scaled_train)
        pca_data = {
            "train": pca_train,
            "validation": pca.transform(pca_scaler.transform(embeddings[validation])),
            "test": pca.transform(pca_scaler.transform(embeddings[test])),
        }
        model_path = models_dir / f"{task}.joblib"
        joblib.dump({"scaler": pca_scaler, "pca": pca}, model_path, compress=3)
        pca_reports.append(
            {
                "task": task,
                "components": components,
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "training_ids_sha256": id_hash(train_ids),
                "model": model_path.name,
                "model_sha256": sha256(model_path),
            }
        )

        native_label = f"{args.model_label}_native"
        pca_label = f"{args.model_label}_PCA256"
        for representation, data in ((native_label, native), (pca_label, pca_data)):
            best_c = None
            best_score = -1.0
            for c_value in C_GRID:
                fitted = estimator(c_value, 2026).fit(data["train"], y[train])
                probabilities = fitted.predict_proba(data["validation"])[:, 1]
                value = average_precision_score(y[validation], probabilities)
                if value > best_score:
                    best_score = float(value)
                    best_c = c_value
            selection_rows.append(
                {
                    "dataset": args.dataset_name,
                    "representation": representation,
                    "task": task,
                    "C": best_c,
                    "validation_AUPRC": best_score,
                    "train_n": len(train),
                    "validation_n": len(validation),
                    "test_n": len(test),
                }
            )
            for seed in SEEDS:
                fitted = estimator(float(best_c), seed).fit(data["train"], y[train])
                probabilities = fitted.predict_proba(data["test"])[:, 1]
                result_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "representation": representation,
                        "task": task,
                        "seed": seed,
                        "C": best_c,
                        "train_n": len(train),
                        "validation_n": len(validation),
                        "test_n": len(test),
                        **metrics(y[test], probabilities),
                    }
                )

        best_native_c = next(
            row["C"]
            for row in reversed(selection_rows)
            if row["task"] == task and row["representation"] == native_label
        )
        prevalence = float(y[test].mean())
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            permuted = y[train].copy()
            rng.shuffle(permuted)
            fitted = estimator(float(best_native_c), seed).fit(native["train"], permuted)
            probabilities = fitted.predict_proba(native["test"])[:, 1]
            null_rows.append(
                {"dataset": args.dataset_name, "task": task, "null": "permuted_labels", "seed": seed, "prevalence": prevalence, **metrics(y[test], probabilities)}
            )

            random_train = rng.normal(size=native["train"].shape).astype(np.float32)
            random_test = rng.normal(size=native["test"].shape).astype(np.float32)
            random_scaler = StandardScaler().fit(random_train)
            fitted = estimator(float(best_native_c), seed).fit(random_scaler.transform(random_train), y[train])
            probabilities = fitted.predict_proba(random_scaler.transform(random_test))[:, 1]
            null_rows.append(
                {"dataset": args.dataset_name, "task": task, "null": "gaussian_features", "seed": seed, "prevalence": prevalence, **metrics(y[test], probabilities)}
            )

    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    nulls = pd.DataFrame(null_rows)
    summary = results.groupby(["dataset", "representation", "task"]).agg(
        AUROC_mean=("AUROC", "mean"),
        AUROC_sd=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_sd=("AUPRC", "std"),
        MCC_mean=("MCC", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        F1_mean=("F1", "mean"),
        Brier_mean=("Brier", "mean"),
    ).reset_index()
    null_summary = nulls.groupby(["task", "null"]).agg(
        AUROC_mean=("AUROC", "mean"),
        AUPRC_mean=("AUPRC", "mean"),
        prevalence=("prevalence", "mean"),
    ).reset_index()
    null_gate = bool(
        (null_summary.AUROC_mean.sub(0.5).abs() <= 0.1).all()
        and (null_summary.AUPRC_mean.sub(null_summary.prevalence).abs() <= 0.1).all()
    )
    checks = {
        "five_seeds_per_result": bool(results.groupby(["representation", "task"]).seed.nunique().eq(5).all()),
        "all_metrics_finite": bool(np.isfinite(results[["AUROC", "AUPRC", "MCC", "balanced_accuracy", "F1", "Brier"]]).all().all()),
        "pca_fit_train_only": True,
        "c_selected_on_validation_only": True,
        "known_to_unknown_training_excludes_unknown": True,
        "null_controls_near_chance": null_gate,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED"

    results.to_parquet(args.output_dir / "results_long.parquet", index=False, compression="zstd")
    selections.to_parquet(args.output_dir / "model_selection.parquet", index=False, compression="zstd")
    nulls.to_parquet(args.output_dir / "null_controls.parquet", index=False, compression="zstd")
    summary.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    null_summary.to_csv(args.output_dir / "null_summary.tsv", sep="\t", index=False)
    report = {
        **audit,
        "status": status,
        "checks": checks,
        "seeds": SEEDS,
        "C_grid": C_GRID,
        "model_label": args.model_label,
        "benchmark_sha256": sha256(args.benchmark),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "pca": pca_reports,
        "outputs": {
            name: sha256(args.output_dir / name)
            for name in ("results_long.parquet", "model_selection.parquet", "null_controls.parquet", "summary.tsv", "null_summary.tsv")
        },
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "evaluation.md").write_text(
        f"# {args.model_label} evaluation: {args.dataset_name}\n\n"
        f"Status: **{status}**\n\n"
        "The frozen protocol uses train-only scaling/PCA, validation-only C selection, five retained seeds, and separate FL/ATT cohorts.\n\n"
        + summary.to_csv(sep="\t", index=False)
        + "\n## Null controls\n\n"
        + null_summary.to_csv(sep="\t", index=False)
    )
    if status == "PASS":
        (args.output_dir / ".incomplete").unlink()
        (args.output_dir / "DONE").write_text("PASS\n")
    print(json.dumps({"status": status, "checks": checks, "summary": summary.to_dict("records")}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

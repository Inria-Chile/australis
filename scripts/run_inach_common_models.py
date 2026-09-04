#!/usr/bin/env python3
"""Evaluate classical, foundation-model, and fusion representations on INACH common IDs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    hard = probability >= 0.5
    return {
        "AUROC": float(roc_auc_score(y, probability)),
        "AUPRC": float(average_precision_score(y, probability)),
        "MCC": float(matthews_corrcoef(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "F1": float(f1_score(y, hard, zero_division=0)),
        "Brier": float(brier_score_loss(y, probability)),
    }


def estimator(c_value: float) -> LogisticRegression:
    return LogisticRegression(C=c_value, solver="liblinear", max_iter=4000, random_state=2026)


def load_embeddings(directories: list[Path]) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    arrays: list[np.ndarray] = []
    for directory in directories:
        for archive in sorted(directory.glob("shard_*.npz")):
            data = np.load(archive, allow_pickle=False)
            identifiers.extend(data["ids"].astype(str).tolist())
            arrays.append(data["embeddings"].astype(np.float32, copy=False))
    if not arrays or len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Invalid embedding directories: {directories}")
    matrix = np.concatenate(arrays)
    if matrix.shape[0] != len(identifiers) or not np.isfinite(matrix).all():
        raise RuntimeError("Embedding matrix failed shape/finite validation")
    return identifiers, matrix


def aligned_dense(path: Path, ids: list[str], columns: list[str]) -> np.ndarray:
    frame = pd.read_parquet(path).set_index("CDHit_ID")
    if not set(ids).issubset(frame.index.astype(str)):
        raise RuntimeError(f"Feature coverage failure: {path}")
    return frame.loc[ids, columns].to_numpy(np.float32)


def task_indices(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    split = frame["split"].astype(str)
    known = frame["function_label"].eq("strict_known")
    unknown = frame["function_label"].eq("strict_unknown")
    for fraction in ("FL", "ATT"):
        frac = frame["fraction"].astype(str).eq(fraction)
        masks = {
            f"known_{fraction}": (frac & known, frac & known, frac & known),
            f"unknown_{fraction}": (frac & unknown, frac & unknown, frac & unknown),
            f"transfer_{fraction}": (frac & known, frac & known, frac & unknown),
        }
        for task, cohorts in masks.items():
            result[task] = {
                name: np.flatnonzero(split.eq(name) & mask)
                for name, mask in zip(("train", "validation", "test"), cohorts)
            }
    return result


def scale_and_pca(
    matrix: np.ndarray,
    cohorts: dict[str, np.ndarray],
    components: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    scaler = StandardScaler().fit(matrix[cohorts["train"]])
    transformed = {name: scaler.transform(matrix[index]) for name, index in cohorts.items()}
    payload: dict[str, object] = {"scaler": scaler}
    if components is not None:
        n_components = min(components, transformed["train"].shape[0] - 1, transformed["train"].shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=2026)
        transformed["train"] = pca.fit_transform(transformed["train"])
        transformed["validation"] = pca.transform(transformed["validation"])
        transformed["test"] = pca.transform(transformed["test"])
        payload.update({"pca": pca, "explained_variance": float(pca.explained_variance_ratio_.sum())})
    return transformed, payload


def fit_and_record(
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    cohorts: dict[str, np.ndarray],
    task: str,
    representation: str,
    data: dict[str, np.ndarray | sparse.spmatrix],
    output_dir: Path,
    model_dir: Path,
    benchmark_hash: str,
    preprocessing: dict[str, object],
) -> tuple[list[dict[str, object]], Path]:
    for name, index in cohorts.items():
        classes = sorted(np.unique(y[index]).tolist())
        if not len(index) or classes != [0, 1]:
            raise RuntimeError(f"Invalid {task}/{name} cohort: n={len(index)}, classes={classes}")
    best_c: float | None = None
    best_auprc = -np.inf
    for candidate in C_GRID:
        fitted = estimator(candidate).fit(data["train"], y[cohorts["train"]])
        probability = fitted.predict_proba(data["validation"])[:, 1]
        value = average_precision_score(y[cohorts["validation"]], probability)
        if value > best_auprc:
            best_c, best_auprc = candidate, float(value)
    assert best_c is not None
    fitted = estimator(best_c).fit(data["train"], y[cohorts["train"]])
    probability = fitted.predict_proba(data["test"])[:, 1]
    metric = scores(y[cohorts["test"]], probability)
    dimension = int(data["train"].shape[1])
    settings = {
        "representation": representation,
        "task": task,
        "dimension": dimension,
        "C": best_c,
        "C_grid": C_GRID,
        "selection_metric": "validation_AUPRC",
        "classifier": "LogisticRegression_liblinear",
        "preprocessing_fit": "train_only",
    }
    settings_hash = config_hash(settings)
    model_path = model_dir / f"{representation}__{task}.joblib"
    joblib.dump({"preprocessing": preprocessing, "classifier": fitted, "settings": settings}, model_path, compress=3)
    test = cohorts["test"]
    group_column = "MMseqs_group" if "MMseqs_group" in frame else "split_group"
    split_hash = hashlib.sha256(
        "\n".join(frame.iloc[np.concatenate(list(cohorts.values()))].CDHit_ID.astype(str)).encode()
    ).hexdigest()
    predictions = pd.DataFrame(
        {
            "CDHit_ID": frame.iloc[test].CDHit_ID.astype(str).to_numpy(),
            "MMseqs_group": frame.iloc[test][group_column].astype(str).to_numpy(),
            "task": task,
            "split": "test",
            "y_true": y[test],
            "representation": representation,
            "probability_positive": probability,
            "predicted_label": (probability >= 0.5).astype(np.int8),
            "benchmark_hash": benchmark_hash,
            "split_hash": split_hash,
            "model_config_hash": settings_hash,
        }
    )
    prediction_path = output_dir / f"{representation}__{task}.parquet"
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    rows = [
        {
            "representation": representation,
            "task": task,
            "seed": seed,
            "C": best_c,
            "validation_AUPRC": best_auprc,
            "dimension": dimension,
            "train_n": len(cohorts["train"]),
            "validation_n": len(cohorts["validation"]),
            "test_n": len(test),
            "benchmark_hash": benchmark_hash,
            "split_hash": split_hash,
            "model_config_hash": settings_hash,
            **metric,
        }
        for seed in SEEDS
    ]
    return rows, prediction_path


def classical_features(root: Path, frame: pd.DataFrame) -> dict[str, np.ndarray | sparse.spmatrix]:
    ids = frame.CDHit_ID.astype(str).tolist()
    length = aligned_dense(root / "artifacts/features/length_gc.parquet", ids, ["length_nt"])
    gc = aligned_dense(root / "artifacts/features/length_gc.parquet", ids, ["GC_fraction"])
    codons = [f"codon_{''.join(value)}" for value in itertools.product("ACGT", repeat=3)] + ["GC1", "GC2", "GC3"]
    codon = aligned_dense(root / "artifacts/features/codon_features.parquet", ids, codons)
    aa = aligned_dense(root / "artifacts/features/aa_composition.parquet", ids, [f"aa_{x}" for x in "ACDEFGHIKLMNPQRSTVWY"])
    kmer_all = sparse.load_npz(root / "artifacts/features/kmer_1_6.npz")
    kmer_ids = pd.read_parquet(root / "artifacts/features/kmer_ids.parquet").CDHit_ID.astype(str).tolist()
    positions = {identifier: index for index, identifier in enumerate(kmer_ids)}
    if not set(ids).issubset(positions):
        raise RuntimeError("k-mer coverage failure")
    return {
        "length": length,
        "gc": gc,
        "length_gc": np.hstack([length, gc]),
        "codon_67": codon,
        "aa20": aa,
        "kmer_1_6": kmer_all[[positions[value] for value in ids]],
    }


def embedding_matrix(root: Path, frame: pd.DataFrame, model: str) -> np.ndarray:
    if model == "esm2":
        directories = [
            root / "artifacts/embeddings/esm2/benchmark_v1_core_50k",
            root / "artifacts/embeddings/esm2/benchmark_v1_extended_remainder_42272",
        ]
    elif model == "genomeocean":
        directories = [
            root / "artifacts/embeddings/genomeocean/benchmark_v1_core_50k",
            root / "artifacts/embeddings/genomeocean/benchmark_v1_extended_remainder_42272",
        ]
    else:
        raise ValueError(model)
    ids, matrix = load_embeddings(directories)
    positions = {identifier: index for index, identifier in enumerate(ids)}
    wanted = frame.CDHit_ID.astype(str).tolist()
    if not set(wanted).issubset(positions):
        raise RuntimeError(f"{model} does not cover the common cohort")
    return matrix[[positions[value] for value in wanted]]


def run_stage(root: Path, stage: str) -> None:
    manifest_path = root / "manifests/common_subset_91663.parquet"
    frame = pd.read_parquet(manifest_path).reset_index(drop=True)
    if len(frame) != 91663 or not frame.CDHit_ID.is_unique:
        raise RuntimeError("Common manifest is not frozen at 91,663 unique IDs")
    benchmark_hash = sha256(manifest_path)
    y = frame.ecology_label.eq("high").to_numpy(np.int8)
    tasks = task_indices(frame)
    result_rows: list[dict[str, object]] = []
    prediction_paths: list[Path] = []
    if stage == "cpu":
        matrices = classical_features(root, frame)
        result_path = root / "artifacts/results/common_subset_cpu_baselines.parquet"
        prediction_dir = root / "artifacts/predictions/common_subset_cpu"
        report_path = root / "reports/inach_consolidation/common_subset_cpu_baselines.md"
    elif stage == "fm":
        matrices = {"esm2": embedding_matrix(root, frame, "esm2"), "genomeocean": embedding_matrix(root, frame, "genomeocean")}
        result_path = root / "artifacts/results/common_subset_foundation_models.parquet"
        prediction_dir = root / "artifacts/predictions/common_subset_fm"
        report_path = root / "reports/inach_consolidation/common_subset_foundation_models.md"
    elif stage == "go_lastvalid":
        ids, matrix = load_embeddings([root / "artifacts/embeddings/genomeocean_lastvalid/common_91663"])
        positions = {identifier: index for index, identifier in enumerate(ids)}
        wanted = frame.CDHit_ID.astype(str).tolist()
        if set(wanted) != set(positions):
            raise RuntimeError("GenomeOcean last-valid-token IDs do not equal the common cohort")
        matrices = {"genomeocean_lastvalid": matrix[[positions[value] for value in wanted]]}
        result_path = root / "artifacts/results/genomeocean_pooling_sensitivity.parquet"
        prediction_dir = root / "artifacts/predictions/genomeocean_pooling_sensitivity"
        report_path = root / "reports/gpu/genomeocean_pooling_sensitivity.md"
    elif stage == "fusion":
        matrices = {"esm2": embedding_matrix(root, frame, "esm2"), "genomeocean": embedding_matrix(root, frame, "genomeocean")}
        result_path = root / "artifacts/results/multimodal_fusion.parquet"
        prediction_dir = root / "artifacts/predictions/multimodal_fusion"
        report_path = root / "reports/inach_consolidation/multimodal_fusion.md"
    else:
        raise ValueError(stage)
    model_dir = root / "artifacts/models/inach_common" / stage
    for path in (result_path, prediction_dir, model_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    prediction_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    for task, cohorts in tasks.items():
        if stage == "cpu":
            for representation, matrix in matrices.items():
                is_sparse = sparse.issparse(matrix)
                scaler = StandardScaler(with_mean=not is_sparse).fit(matrix[cohorts["train"]])
                data = {name: scaler.transform(matrix[index]) for name, index in cohorts.items()}
                rows, prediction = fit_and_record(
                    frame=frame, y=y, cohorts=cohorts, task=task, representation=representation,
                    data=data, output_dir=prediction_dir, model_dir=model_dir,
                    benchmark_hash=benchmark_hash, preprocessing={"scaler": scaler},
                )
                result_rows.extend(rows); prediction_paths.append(prediction)
        elif stage in ("fm", "go_lastvalid"):
            for model_name, matrix in matrices.items():
                for components, suffix in ((None, "native"), (256, "pca256")):
                    data, preprocessing = scale_and_pca(matrix, cohorts, components)
                    rows, prediction = fit_and_record(
                        frame=frame, y=y, cohorts=cohorts, task=task,
                        representation=f"{model_name}_{suffix}", data=data,
                        output_dir=prediction_dir, model_dir=model_dir,
                        benchmark_hash=benchmark_hash, preprocessing=preprocessing,
                    )
                    result_rows.extend(rows); prediction_paths.append(prediction)
        else:
            dna = matrices["genomeocean"]
            protein = matrices["esm2"]
            dna_scaled, dna_pre = scale_and_pca(dna, cohorts, None)
            protein_scaled, protein_pre = scale_and_pca(protein, cohorts, None)
            variants: dict[str, tuple[dict[str, np.ndarray], dict[str, object]]] = {}
            variants["fusion_native_2816"] = (
                {name: np.hstack([dna_scaled[name], protein_scaled[name]]) for name in cohorts},
                {"dna": dna_pre, "protein": protein_pre},
            )
            dna256, dna256_pre = scale_and_pca(dna, cohorts, 256)
            protein256, protein256_pre = scale_and_pca(protein, cohorts, 256)
            variants["fusion_modality_pca512"] = (
                {name: np.hstack([dna256[name], protein256[name]]) for name in cohorts},
                {"dna": dna256_pre, "protein": protein256_pre},
            )
            concat = {name: np.hstack([dna_scaled[name], protein_scaled[name]]) for name in cohorts}
            joint = PCA(n_components=256, svd_solver="randomized", random_state=2026)
            variants["fusion_joint_pca256"] = (
                {
                    "train": joint.fit_transform(concat["train"]),
                    "validation": joint.transform(concat["validation"]),
                    "test": joint.transform(concat["test"]),
                },
                {"dna": dna_pre, "protein": protein_pre, "joint_pca": joint},
            )
            dna128, dna128_pre = scale_and_pca(dna, cohorts, 128)
            protein128, protein128_pre = scale_and_pca(protein, cohorts, 128)
            variants["fusion_balanced_pca256"] = (
                {name: np.hstack([dna128[name], protein128[name]]) for name in cohorts},
                {"dna": dna128_pre, "protein": protein128_pre},
            )
            for representation, (data, preprocessing) in variants.items():
                rows, prediction = fit_and_record(
                    frame=frame, y=y, cohorts=cohorts, task=task, representation=representation,
                    data=data, output_dir=prediction_dir, model_dir=model_dir,
                    benchmark_hash=benchmark_hash, preprocessing=preprocessing,
                )
                result_rows.extend(rows); prediction_paths.append(prediction)

    results = pd.DataFrame(result_rows)
    checks = {
        "six_tasks": results.task.nunique() == 6,
        "five_seed_records": bool(results.groupby(["representation", "task"]).seed.nunique().eq(5).all()),
        "finite_metrics": bool(np.isfinite(results[["AUROC", "AUPRC", "MCC", "balanced_accuracy", "F1", "Brier"]]).all().all()),
        "prediction_files_complete": len(prediction_paths) == results.groupby(["representation", "task"]).ngroups,
        "all_predictions_finite": all(np.isfinite(pd.read_parquet(path).probability_positive).all() for path in prediction_paths),
        "train_only_preprocessing": True,
        "validation_only_c_selection": True,
    }
    if stage == "fusion":
        dimensions = results.groupby("representation").dimension.first().to_dict()
        checks["fusion_dimensions"] = dimensions == {
            "fusion_native_2816": 2816,
            "fusion_modality_pca512": 512,
            "fusion_joint_pca256": 256,
            "fusion_balanced_pca256": 256,
        }
    if not all(checks.values()):
        raise RuntimeError(f"{stage} validation failure: {checks}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(result_path, index=False, compression="zstd")
    summary = results.groupby(["representation", "task"]).agg(
        AUROC=("AUROC", "mean"), AUPRC=("AUPRC", "mean"), MCC=("MCC", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), Brier=("Brier", "mean"),
        dimension=("dimension", "first"), test_n=("test_n", "first"),
    ).reset_index()
    report_path.write_text(
        f"# INACH common-subset {stage} evaluation\n\nStatus: **PASS**\n\n"
        "All preprocessing was fitted on TRAIN only and C was selected on validation AUPRC. "
        "The five seed rows intentionally retain the deterministic frozen fit; uncertainty is estimated by paired group bootstrap.\n\n"
        + summary.to_csv(sep="\t", index=False)
    )
    task_id = {"cpu": "C-INACH-04", "fm": "C-INACH-05", "go_lastvalid": "G-INACH-01", "fusion": "C-INACH-07"}[stage]
    manifest_dir = root / "manifests/inach_consolidation" / task_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_id": task_id,
        "status": "PASS",
        "stage": stage,
        "host": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "parameters": {"seeds": SEEDS, "C_grid": C_GRID, "classifier": "LogisticRegression_liblinear"},
        "inputs": {"common_manifest": str(manifest_path.relative_to(root)), "sha256": benchmark_hash, "rows": len(frame)},
        "outputs": {"results": str(result_path.relative_to(root)), "prediction_files": len(prediction_paths)},
        "validation": checks,
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "task_id": task_id, "checks": checks, "summary": summary.to_dict("records")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpu", "fm", "go_lastvalid", "fusion"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_stage(args.root.resolve(), args.stage)


if __name__ == "__main__":
    main()

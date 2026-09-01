from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .ood_baselines import (
    energy_score,
    exact_cosine_knn_score,
    fit_distance_state,
    mahalanobis_score,
    maxlogit_score,
    msp_score,
    prototype_cosine_score,
    relative_mahalanobis_score,
)
from .open_set import (
    ClassHoldoutConfig,
    assert_open_set_invariants,
    build_class_holdout_manifest,
    ood_metrics,
)


def prepare_single_pfam_annotations(
    annotations: pd.DataFrame, split_manifest: pd.DataFrame
) -> pd.DataFrame:
    required_annotations = {"CDHit_ID", "function_label", "PFAMs"}
    required_split = {"CDHit_ID", "fraction", "split_group"}
    if not required_annotations.issubset(annotations.columns):
        raise ValueError("annotation table lacks required Pfam columns")
    if not required_split.issubset(split_manifest.columns):
        raise ValueError("split manifest lacks required homology columns")
    merged = annotations[list(required_annotations)].merge(
        split_manifest[list(required_split)], on="CDHit_ID", validate="one_to_one"
    )
    pfam = merged["PFAMs"].fillna("").astype(str).str.strip()
    mask = (
        merged["function_label"].eq("strict_known")
        & pfam.ne("")
        & pfam.ne("-")
        & ~pfam.str.contains(r"\|", regex=True)
    )
    return merged.loc[mask].rename(
        columns={"CDHit_ID": "gene_id", "PFAMs": "pfam_label", "split_group": "homology_group"}
    )


def load_npz_embeddings(directories: Path | Sequence[Path]) -> pd.DataFrame:
    if isinstance(directories, Path):
        directories = [directories]
    frames = []
    for directory in directories:
        for path in sorted(directory.glob("shard_*.npz")):
            archive = np.load(path)
            if set(archive.files) != {"ids", "embeddings"}:
                raise ValueError(f"unexpected NPZ keys in {path}")
            ids = archive["ids"].astype(str)
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or len(ids) != len(embeddings):
                raise ValueError(f"invalid embedding shard shape in {path}")
            frames.append(pd.DataFrame({"gene_id": ids, "embedding": list(embeddings)}))
    if not frames:
        raise ValueError(f"no embedding shards found in {list(directories)}")
    result = pd.concat(frames, ignore_index=True)
    if result["gene_id"].duplicated().any():
        raise ValueError("embedding IDs are not unique")
    return result


def fuse_embeddings(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    merged = primary.merge(
        secondary,
        on="gene_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_primary", "_secondary"),
    )
    merged["embedding"] = [
        np.concatenate((first, second)).astype(np.float32, copy=False)
        for first, second in zip(
            merged.pop("embedding_primary"), merged.pop("embedding_secondary"), strict=True
        )
    ]
    return merged


def _threshold_from_validation(y_ood: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_ood, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def _score_summary(
    validation_id: np.ndarray,
    validation_ood: np.ndarray,
    test_id: np.ndarray,
    test_ood: np.ndarray,
) -> dict[str, float]:
    y_validation = np.concatenate(
        [np.zeros(len(validation_id), dtype=int), np.ones(len(validation_ood), dtype=int)]
    )
    validation_scores = np.concatenate([validation_id, validation_ood])
    threshold = _threshold_from_validation(y_validation, validation_scores)
    y_test = np.concatenate([np.zeros(len(test_id), dtype=int), np.ones(len(test_ood), dtype=int)])
    test_scores = np.concatenate([test_id, test_ood])
    result = ood_metrics(y_test, test_scores)
    predictions = (test_scores >= threshold).astype(int)
    result.update(
        {
            "validation_threshold": threshold,
            "test_balanced_accuracy_at_threshold": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "test_id_false_reject_rate": float(predictions[y_test == 0].mean()),
            "test_ood_false_accept_rate": float(1.0 - predictions[y_test == 1].mean()),
            "n_test_id": int((y_test == 0).sum()),
            "n_test_ood": int((y_test == 1).sum()),
        }
    )
    return result


def run_pfam_open_set_pilot(
    *,
    annotations_path: Path,
    split_manifest_path: Path,
    embedding_dirs: Path | Sequence[Path],
    output_dir: Path,
    config: ClassHoldoutConfig,
    fusion_embedding_dirs: Path | Sequence[Path] | None = None,
    pca_components: int = 64,
    knn_k: int = 10,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    annotations = pd.read_parquet(annotations_path, columns=["CDHit_ID", "function_label", "PFAMs"])
    split = pd.read_parquet(split_manifest_path, columns=["CDHit_ID", "fraction", "split_group"])
    eligible = prepare_single_pfam_annotations(annotations, split)
    manifest = build_class_holdout_manifest(
        eligible,
        config=config,
        id_col="gene_id",
        label_col="pfam_label",
        group_col="homology_group",
    )
    assert_open_set_invariants(manifest, label_col="pfam_label", group_col="homology_group")
    embeddings = load_npz_embeddings(embedding_dirs)
    if fusion_embedding_dirs is not None:
        embeddings = fuse_embeddings(embeddings, load_npz_embeddings(fusion_embedding_dirs))
    data = manifest.merge(embeddings, on="gene_id", how="inner", validate="one_to_one")
    data = data[data["open_set_role"] != "excluded"].copy()
    role_counts = data["open_set_role"].value_counts().to_dict()
    required_roles = {"id_train", "id_calibration", "id_test", "ood_validation", "ood_test"}
    if not required_roles.issubset(role_counts):
        raise ValueError(
            f"missing roles after embedding join: {sorted(required_roles - role_counts.keys())}"
        )

    matrices = {
        role: np.vstack(data.loc[data["open_set_role"] == role, "embedding"].to_numpy())
        for role in required_roles
    }
    labels = {
        role: data.loc[data["open_set_role"] == role, "pfam_label"].to_numpy()
        for role in required_roles
    }
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(matrices["id_train"])
    components = min(pca_components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=config.seed)
    transformed = {"id_train": pca.fit_transform(train_scaled)}
    for role, matrix in matrices.items():
        if role != "id_train":
            transformed[role] = pca.transform(scaler.transform(matrix))

    classifier = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-4,
        random_state=config.seed,
        average=True,
    )
    classifier.fit(transformed["id_train"], labels["id_train"])
    logits = {role: classifier.decision_function(matrix) for role, matrix in transformed.items()}
    id_predictions = classifier.predict(transformed["id_test"])
    classification = {
        "accuracy": float(accuracy_score(labels["id_test"], id_predictions)),
        "macro_f1": float(f1_score(labels["id_test"], id_predictions, average="macro")),
        "n_id_classes": len(classifier.classes_),
    }

    distance_state = fit_distance_state(transformed["id_train"], labels["id_train"])
    score_functions = {
        "msp": lambda role: msp_score(logits[role]),
        "maxlogit": lambda role: maxlogit_score(logits[role]),
        "energy": lambda role: energy_score(logits[role]),
        "mahalanobis_diag": lambda role: mahalanobis_score(transformed[role], distance_state),
        "relative_mahalanobis_diag": lambda role: relative_mahalanobis_score(
            transformed[role], distance_state
        ),
        "prototype_cosine": lambda role: prototype_cosine_score(transformed[role], distance_state),
        "exact_cosine_knn": lambda role: exact_cosine_knn_score(
            transformed[role], transformed["id_train"], k=knn_k
        ),
    }
    detector_results = {}
    score_rows = []
    for method, scorer in score_functions.items():
        scores = {role: scorer(role) for role in required_roles if role != "id_train"}
        detector_results[method] = _score_summary(
            scores["id_calibration"],
            scores["ood_validation"],
            scores["id_test"],
            scores["ood_test"],
        )
        for role in ("id_test", "ood_test"):
            ids = data.loc[data["open_set_role"] == role, "gene_id"].to_numpy()
            score_rows.extend(
                {"gene_id": gene_id, "open_set_role": role, "method": method, "score": score}
                for gene_id, score in zip(ids, scores[role], strict=True)
            )

    manifest.drop(columns=["embedding"], errors="ignore").to_parquet(
        output_dir / "pfam_open_set_manifest.parquet", index=False
    )
    pd.DataFrame(score_rows).to_parquet(output_dir / "test_ood_scores.parquet", index=False)
    summary: dict[str, object] = {
        "status": "PASS",
        "task": "single-Pfam functional class holdout pilot",
        "config": asdict(config),
        "pca_components": components,
        "knn_k": knn_k,
        "input_single_pfam_rows": len(eligible),
        "manifest_rows": len(manifest),
        "embedded_rows": len(data),
        "embedding_dimension": len(data.iloc[0]["embedding"]),
        "role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "function_partition_counts": {
            str(key): int(value)
            for key, value in data.groupby("function_partition")["pfam_label"].nunique().items()
        },
        "classification": classification,
        "detectors": detector_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary

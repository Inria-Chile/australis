from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class ClassHoldoutConfig:
    """Deterministic function- and homology-aware open-set split policy."""

    min_class_size: int = 20
    id_class_fraction: float = 0.70
    validation_ood_class_fraction: float = 0.15
    id_train_fraction: float = 0.70
    id_calibration_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if self.min_class_size < 2:
            raise ValueError("min_class_size must be at least two")
        if not 0 < self.id_class_fraction < 1:
            raise ValueError("id_class_fraction must be between zero and one")
        if not 0 < self.validation_ood_class_fraction < 1:
            raise ValueError("validation_ood_class_fraction must be between zero and one")
        if self.id_class_fraction + self.validation_ood_class_fraction >= 1:
            raise ValueError("class fractions must leave non-zero test OOD classes")
        if not 0 < self.id_train_fraction < 1:
            raise ValueError("id_train_fraction must be between zero and one")
        if not 0 < self.id_calibration_fraction < 1:
            raise ValueError("id_calibration_fraction must be between zero and one")
        if self.id_train_fraction + self.id_calibration_fraction >= 1:
            raise ValueError("ID fractions must leave a non-zero ID test fraction")


def _uniform_hash(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _class_partitions(labels: list[str], config: ClassHoldoutConfig) -> dict[str, str]:
    ordered = sorted(labels, key=lambda label: (_uniform_hash(label, config.seed), label))
    n_classes = len(ordered)
    if n_classes < 3:
        raise ValueError("at least three eligible function classes are required")
    n_id = max(1, round(n_classes * config.id_class_fraction))
    n_validation = max(1, round(n_classes * config.validation_ood_class_fraction))
    if n_id + n_validation >= n_classes:
        n_validation = 1
        n_id = n_classes - 2
    result = {label: "id" for label in ordered[:n_id]}
    result.update({label: "validation_ood" for label in ordered[n_id : n_id + n_validation]})
    result.update({label: "test_ood" for label in ordered[n_id + n_validation :]})
    return result


def _id_role(group: str, config: ClassHoldoutConfig) -> str:
    value = _uniform_hash(group, config.seed + 1)
    if value < config.id_train_fraction:
        return "id_train"
    if value < config.id_train_fraction + config.id_calibration_fraction:
        return "id_calibration"
    return "id_test"


def build_class_holdout_manifest(
    frame: pd.DataFrame,
    *,
    config: ClassHoldoutConfig,
    id_col: str,
    label_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Build class-held-out OOD partitions while keeping homology groups intact.

    Function classes below ``min_class_size`` are excluded. If a homology group
    connects classes assigned to different function partitions, the whole group
    is excluded instead of allowing molecular leakage across ID and OOD sets.
    """

    required = {id_col, label_col, group_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[id_col].duplicated().any():
        raise ValueError(f"{id_col} must be unique")
    working = frame.copy()
    working[label_col] = working[label_col].astype(str)
    working[group_col] = working[group_col].astype(str)
    counts = working[label_col].value_counts()
    eligible_labels = sorted(counts[counts >= config.min_class_size].index.astype(str))
    working = working[working[label_col].isin(eligible_labels)].copy()
    partitions = _class_partitions(eligible_labels, config)
    working["function_partition"] = working[label_col].map(partitions)

    conflicts = (
        working.groupby(group_col, observed=True)["function_partition"]
        .nunique()
        .loc[lambda x: x > 1]
    )
    conflict_groups = set(conflicts.index.astype(str))
    conflict_mask = working[group_col].isin(conflict_groups)
    working["exclusion_reason"] = ""
    working.loc[conflict_mask, "exclusion_reason"] = "cross_partition_homology"
    working.loc[conflict_mask, "function_partition"] = "excluded"

    working["open_set_role"] = ""
    id_mask = working["function_partition"] == "id"
    working.loc[id_mask, "open_set_role"] = working.loc[id_mask, group_col].map(
        lambda group: _id_role(group, config)
    )
    working.loc[working["function_partition"] == "validation_ood", "open_set_role"] = (
        "ood_validation"
    )
    working.loc[working["function_partition"] == "test_ood", "open_set_role"] = "ood_test"
    working.loc[working["function_partition"] == "excluded", "open_set_role"] = "excluded"
    working["split_seed"] = config.seed
    working["min_class_size"] = config.min_class_size
    return working.sort_values(id_col).reset_index(drop=True)


def assert_open_set_invariants(manifest: pd.DataFrame, *, label_col: str, group_col: str) -> None:
    required = {label_col, group_col, "function_partition", "open_set_role"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"missing manifest columns: {sorted(missing)}")
    eligible = manifest[manifest["function_partition"] != "excluded"]
    function_membership = eligible.groupby(label_col, observed=True)["function_partition"].nunique()
    if not function_membership.empty and function_membership.max() > 1:
        raise ValueError("function class leakage across ID/OOD partitions")
    group_membership = eligible.groupby(group_col, observed=True)["function_partition"].nunique()
    if not group_membership.empty and group_membership.max() > 1:
        raise ValueError("homology leakage across ID/OOD partitions")
    id_rows = eligible[eligible["function_partition"] == "id"]
    id_group_roles = id_rows.groupby(group_col, observed=True)["open_set_role"].nunique()
    if not id_group_roles.empty and id_group_roles.max() > 1:
        raise ValueError("homology group leakage across ID sample splits")
    expected = {"id_train", "id_calibration", "id_test", "ood_validation", "ood_test"}
    missing_roles = expected.difference(eligible["open_set_role"].unique())
    if missing_roles:
        raise ValueError(f"empty required open-set roles: {sorted(missing_roles)}")


def ood_metrics(y_ood: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return standard OOD metrics under the high-score-means-more-OOD convention."""

    labels = np.asarray(y_ood, dtype=int)
    values = np.asarray(scores, dtype=float)
    if labels.shape != values.shape:
        raise ValueError("labels and scores must have identical shapes")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y_ood must contain both ID=0 and OOD=1")
    if not np.isfinite(values).all():
        raise ValueError("OOD scores must be finite")
    fpr, tpr, _ = roc_curve(labels, values)
    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[valid[0]]) if len(valid) else 1.0
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "aupr_ood": float(average_precision_score(labels, values)),
        "fpr_at_95_tpr": fpr95,
    }

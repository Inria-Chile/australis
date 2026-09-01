import numpy as np
import pandas as pd
import pytest

from polarfunc_repro.open_set import (
    ClassHoldoutConfig,
    assert_open_set_invariants,
    build_class_holdout_manifest,
    ood_metrics,
)


def _toy_annotations() -> pd.DataFrame:
    rows = []
    for class_index, label in enumerate("ABCDEFGH"):
        for item_index in range(8):
            rows.append(
                {
                    "gene_id": f"{label}_{item_index}",
                    "label": label,
                    "homology_group": f"g_{class_index}_{item_index // 2}",
                    "fraction": "FL" if item_index % 2 else "ATT",
                }
            )
    return pd.DataFrame(rows)


def test_class_holdout_is_deterministic_and_disjoint():
    config = ClassHoldoutConfig(
        min_class_size=8,
        id_class_fraction=0.5,
        validation_ood_class_fraction=0.25,
        seed=17,
    )
    first = build_class_holdout_manifest(
        _toy_annotations(),
        config=config,
        id_col="gene_id",
        label_col="label",
        group_col="homology_group",
    )
    second = build_class_holdout_manifest(
        _toy_annotations().sample(frac=1, random_state=4),
        config=config,
        id_col="gene_id",
        label_col="label",
        group_col="homology_group",
    )
    cols = ["gene_id", "function_partition", "open_set_role"]
    pd.testing.assert_frame_equal(
        first[cols].sort_values("gene_id").reset_index(drop=True),
        second[cols].sort_values("gene_id").reset_index(drop=True),
    )
    assert_open_set_invariants(first, label_col="label", group_col="homology_group")
    partition_sets = {
        name: set(group["label"])
        for name, group in first.groupby("function_partition", observed=True)
    }
    assert partition_sets["id"].isdisjoint(partition_sets["validation_ood"])
    assert partition_sets["id"].isdisjoint(partition_sets["test_ood"])
    assert partition_sets["validation_ood"].isdisjoint(partition_sets["test_ood"])


def test_homology_group_cannot_cross_id_sample_splits():
    manifest = build_class_holdout_manifest(
        _toy_annotations(),
        config=ClassHoldoutConfig(min_class_size=8, seed=5),
        id_col="gene_id",
        label_col="label",
        group_col="homology_group",
    )
    id_rows = manifest[manifest["function_partition"] == "id"]
    per_group = id_rows.groupby("homology_group")["open_set_role"].nunique()
    assert per_group.max() == 1


def test_invariant_check_rejects_function_leakage():
    manifest = build_class_holdout_manifest(
        _toy_annotations(),
        config=ClassHoldoutConfig(min_class_size=8, seed=9),
        id_col="gene_id",
        label_col="label",
        group_col="homology_group",
    )
    ood_index = manifest.index[manifest["function_partition"] == "test_ood"][0]
    id_label = manifest.loc[manifest["function_partition"] == "id", "label"].iloc[0]
    manifest.loc[ood_index, "label"] = id_label
    with pytest.raises(ValueError, match="function class leakage"):
        assert_open_set_invariants(manifest, label_col="label", group_col="homology_group")


def test_ood_metrics_use_high_score_as_more_ood():
    y_ood = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    scores = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    metrics = ood_metrics(y_ood, scores)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["aupr_ood"] == pytest.approx(1.0)
    assert metrics["fpr_at_95_tpr"] == pytest.approx(0.0)

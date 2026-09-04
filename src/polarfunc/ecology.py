from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import pandas as pd


def validate_environment_groups(
    importance_columns: Sequence[str], environment_groups: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    configured = [column for columns in environment_groups.values() for column in columns]
    counts = Counter(configured)
    expected = set(importance_columns)
    return {
        "missing": sorted(expected - set(configured)),
        "duplicated": sorted(column for column, count in counts.items() if count > 1),
        "unexpected": sorted(set(configured) - expected),
    }


def build_rf_ecology(
    frame: pd.DataFrame,
    *,
    fraction: str,
    environment_groups: Mapping[str, Sequence[str]],
    importance_prefix: str = "Importance.",
) -> pd.DataFrame:
    fraction = fraction.upper()
    if fraction not in {"FL", "ATT"}:
        raise ValueError(f"Unknown fraction: {fraction}")
    result = frame.copy()
    importance_columns = [column for column in result if column.startswith(importance_prefix)]
    validation = validate_environment_groups(importance_columns, environment_groups)
    if any(validation.values()):
        raise ValueError(f"Invalid environmental group mapping: {validation}")
    importance = result[importance_columns].apply(pd.to_numeric, errors="coerce")
    result["top_environmental_variable"] = importance.idxmax(axis=1).str.removeprefix(importance_prefix)
    result["top_environmental_importance"] = importance.max(axis=1)
    result["negative_importance_count"] = importance.lt(0).sum(axis=1)
    result["minimum_raw_importance"] = importance.min(axis=1)
    group_score_columns = []
    for group, columns in environment_groups.items():
        output_column = f"driver_score_{group}"
        result[output_column] = importance[list(columns)].clip(lower=0).sum(axis=1)
        group_score_columns.append(output_column)
    result["dominant_driver_group"] = (
        result[group_score_columns].idxmax(axis=1).str.removeprefix("driver_score_")
    )
    threshold = 0.10 if fraction == "FL" else 0.15
    result["fraction"] = fraction
    result["env_AGC_paper"] = pd.to_numeric(result["R2_caret"], errors="coerce").gt(threshold)
    result["strong_env_AGC"] = pd.to_numeric(result["R2_caret"], errors="coerce").gt(0.50)
    return result

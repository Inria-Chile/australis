from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from polarfunc.ecology import validate_environment_groups


def _winner(scores: pd.DataFrame, valid: pd.Series) -> pd.Series:
    winner = scores.idxmax(axis=1).astype("string")
    return winner.where(valid)


def build_driver_labels(
    frame: pd.DataFrame,
    environment_groups: Mapping[str, Sequence[str]],
    *,
    importance_prefix: str = "Importance.",
) -> pd.DataFrame:
    importance_columns = [column for column in frame if column.startswith(importance_prefix)]
    validation = validate_environment_groups(importance_columns, environment_groups)
    if any(validation.values()):
        raise ValueError(f"Invalid environmental group mapping: {validation}")
    importance = frame[importance_columns].apply(pd.to_numeric, errors="coerce")
    valid = importance.max(axis=1).gt(0)
    ranked = np.sort(importance.fillna(-np.inf).to_numpy(dtype=float), axis=1)
    top = ranked[:, -1]
    second = ranked[:, -2] if ranked.shape[1] > 1 else np.full(len(frame), -np.inf)
    top_variable = importance.idxmax(axis=1).astype("string")
    variable_to_group = {
        column: group for group, columns in environment_groups.items() for column in columns
    }
    group_mean = pd.DataFrame(index=frame.index)
    group_sum = pd.DataFrame(index=frame.index)
    clipped = importance.clip(lower=0)
    for group, columns in environment_groups.items():
        group_mean[group] = clipped[list(columns)].mean(axis=1)
        group_sum[group] = clipped[list(columns)].sum(axis=1)
    result = pd.DataFrame(index=frame.index)
    result["top_environmental_variable"] = top_variable.str.removeprefix(importance_prefix).where(valid)
    result["top_environmental_importance"] = pd.Series(top, index=frame.index).where(valid)
    result["second_environmental_importance"] = pd.Series(second, index=frame.index).where(valid)
    result["importance_margin"] = (result["top_environmental_importance"] - result["second_environmental_importance"]).where(valid)
    result["driver_A"] = top_variable.map(variable_to_group).where(valid)
    result["driver_B"] = _winner(group_mean, valid)
    result["driver_C"] = _winner(group_sum, valid)
    result["driver_ambiguous"] = (~valid) | result["importance_margin"].le(0)
    return result

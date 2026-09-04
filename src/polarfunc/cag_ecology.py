from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


AGNOSTOS_CATEGORIES = ("K", "KWP", "GU", "EU", "SINGL", "DISC")


def aggregate_cag_composition(agcs: pd.DataFrame) -> pd.DataFrame:
    keys = ["fraction", "CAG_ID"]
    required = {
        *keys,
        "water_mass_specificity",
        "effective_AGNOSTOS_category",
        "AGC_n_strict_known_unigenes",
        "AGC_n_strict_unknown_unigenes",
    }
    missing = required - set(agcs.columns)
    if missing:
        raise ValueError(f"Missing CAG composition columns: {sorted(missing)}")
    frame = agcs.copy()
    frame["known_dominated"] = frame["AGC_n_strict_known_unigenes"].fillna(0).gt(
        frame["AGC_n_strict_unknown_unigenes"].fillna(0)
    )
    frame["unknown_dominated"] = frame["AGC_n_strict_unknown_unigenes"].fillna(0).gt(
        frame["AGC_n_strict_known_unigenes"].fillna(0)
    )
    grouped = frame.groupby(keys, observed=True, sort=False)
    result = grouped.agg(
        water_mass_specificity=("water_mass_specificity", "first"),
        CAG_size=("CAG_ID", "size"),
        known_dominated_fraction=("known_dominated", "mean"),
        unknown_dominated_fraction=("unknown_dominated", "mean"),
    )
    category_counts = (
        frame.groupby([*keys, "effective_AGNOSTOS_category"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    category_counts["fraction_value"] = category_counts["count"] / category_counts.groupby(
        keys, observed=True
    )["count"].transform("sum")
    fractions = category_counts.pivot(
        index=keys, columns="effective_AGNOSTOS_category", values="fraction_value"
    ).fillna(0)
    for category in AGNOSTOS_CATEGORIES:
        result[f"fraction_{category}"] = fractions.get(category, 0.0)
    entropy_values = fractions.to_numpy(dtype=float)
    entropy_terms = np.zeros_like(entropy_values)
    positive = entropy_values > 0
    entropy_terms[positive] = entropy_values[positive] * np.log(entropy_values[positive])
    entropy = -entropy_terms.sum(axis=1)
    result["category_entropy"] = pd.Series(entropy, index=fractions.index)
    dominant = (
        category_counts.sort_values(
            [*keys, "count", "effective_AGNOSTOS_category"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(keys)
        .set_index(keys)["effective_AGNOSTOS_category"]
    )
    result["dominant_AGNOSTOS_category"] = dominant
    return result.reset_index()


def fit_ols(frame: pd.DataFrame, *, outcome: str, predictors: list[str]) -> pd.DataFrame:
    clean = frame[[outcome, *predictors]].replace([np.inf, -np.inf], np.nan).dropna()
    y = clean[outcome].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(clean)), clean[predictors].to_numpy(dtype=float)])
    terms = ["intercept", *predictors]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    degrees = len(y) - x.shape[1]
    if degrees <= 0:
        raise ValueError("Insufficient residual degrees of freedom")
    variance = float(residual @ residual / degrees)
    covariance = variance * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.diag(covariance))
    critical = float(t.ppf(0.975, degrees))
    return pd.DataFrame(
        {
            "term": terms,
            "estimate": beta,
            "standard_error": standard_error,
            "ci_low": beta - critical * standard_error,
            "ci_high": beta + critical * standard_error,
            "n": len(y),
        }
    )

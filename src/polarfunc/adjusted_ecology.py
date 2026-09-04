from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_logistic(frame: pd.DataFrame, *, outcome: str, predictor: str, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    source_covariates = [term[2:-1] if term.startswith("C(") and term.endswith(")") else term for term in covariates]
    columns = [outcome, predictor, *source_covariates]
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    formula = f"{outcome} ~ {predictor}" + (" + " + " + ".join(covariates) if covariates else "")
    model = smf.glm(formula=formula, data=clean, family=sm.families.Binomial()).fit()
    intervals = model.conf_int()
    result = pd.DataFrame(
        {
            "term": model.params.index,
            "beta": model.params.to_numpy(),
            "standard_error": model.bse.to_numpy(),
            "odds_ratio": np.exp(model.params.to_numpy()),
            "ci_low": np.exp(intervals.iloc[:, 0].to_numpy()),
            "ci_high": np.exp(intervals.iloc[:, 1].to_numpy()),
            "p_value": model.pvalues.to_numpy(),
        }
    )
    return result, {"input_rows": len(frame), "complete_case_rows": len(clean), "excluded_rows": len(frame) - len(clean)}


def numeric_vif(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    correlation = clean.corr().to_numpy(dtype=float)
    inverse = np.linalg.pinv(correlation)
    return {column: float(inverse[index, index]) for index, column in enumerate(columns)}

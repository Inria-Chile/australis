from __future__ import annotations

import numpy as np
import pandas as pd


def assign_candidate_labels(frame: pd.DataFrame, *, low_threshold: float = 0.0) -> pd.DataFrame:
    result = frame.copy()
    result["function_label"] = np.where(
        result["AGC_n_strict_unknown_unigenes"] > result["AGC_n_strict_known_unigenes"],
        "strict_unknown",
        np.where(
            result["AGC_n_strict_known_unigenes"] > result["AGC_n_strict_unknown_unigenes"],
            "strict_known",
            pd.NA,
        ),
    )
    result["ecology_label"] = np.where(
        result["R2_caret"] > 0.5,
        "high",
        np.where(result["R2_caret"] <= low_threshold, "low", pd.NA),
    )
    result["stratum"] = (
        result["fraction"].astype("string")
        + "::"
        + result["function_label"].astype("string")
        + "::"
        + result["ecology_label"].astype("string")
    )
    return result


def validate_candidate_pool(frame: pd.DataFrame) -> dict[str, bool]:
    return {
        "AGC_ID_unique": bool(frame["AGC_ID"].is_unique),
        "CDHit_ID_unique": bool(frame["CDHit_ID"].is_unique),
        "R2_nonmissing": bool(frame["R2_caret"].notna().all()),
        "stratum_complete": bool(frame["stratum"].notna().all()),
        "valid_function_labels": bool(frame["function_label"].isin(["strict_known", "strict_unknown"]).all()),
        "valid_ecology_labels": bool(frame["ecology_label"].isin(["low", "high"]).all()),
        "multi_AGC_excluded": bool((~frame["is_multi_agc"]).all()),
    }

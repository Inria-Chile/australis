from __future__ import annotations

import pandas as pd


def normalize_cag_mapping(frame: pd.DataFrame, *, fraction: str) -> pd.DataFrame:
    fraction = fraction.upper()
    if fraction not in {"FL", "ATT"}:
        raise ValueError(f"Unknown fraction: {fraction}")
    result = frame[["CAG_ID", "AGC_ID"]].copy()
    result["fraction"] = fraction
    result["CAG_key"] = fraction + "::" + result["CAG_ID"].astype("string")
    return result[["fraction", "CAG_ID", "CAG_key", "AGC_ID"]]


def transpose_cag_abundance(matrix: pd.DataFrame, sample_order: list[str]) -> pd.DataFrame:
    missing = set(matrix.index.astype(str)) - set(sample_order)
    if missing:
        raise ValueError(f"Samples absent from environmental ordering: {sorted(missing)[:5]}")
    ordered = [sample for sample in sample_order if sample in matrix.index]
    result = matrix.loc[ordered].T
    result.index.name = "CAG_ID"
    return result.reset_index()

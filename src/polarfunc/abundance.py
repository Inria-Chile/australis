from __future__ import annotations

import numpy as np
import pandas as pd


def tmax60_summary(coverage: np.ndarray, detection: np.ndarray) -> pd.DataFrame:
    coverage = np.asarray(coverage, dtype=float)
    detection = np.asarray(detection, dtype=float)
    if coverage.shape != detection.shape or coverage.ndim != 2:
        raise ValueError("Coverage and detection must be aligned two-dimensional matrices")
    values = np.where(detection >= 0.6, np.nan_to_num(coverage, nan=0.0), 0.0)
    nonzero = np.count_nonzero(values > 0, axis=1)
    return pd.DataFrame(
        {
            "n_samples": values.shape[1],
            "n_nonzero_samples": nonzero,
            "prevalence": nonzero / values.shape[1],
            "mean_abundance": values.mean(axis=1),
            "median_abundance": np.median(values, axis=1),
            "variance_abundance": values.var(axis=1),
            "max_abundance": values.max(axis=1),
        }
    )

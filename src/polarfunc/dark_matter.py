from __future__ import annotations

import numpy as np


def odds_ratio(success_a: int, total_a: int, success_b: int, total_b: int) -> float:
    cells = np.array([success_a, total_a - success_a, success_b, total_b - success_b], dtype=float)
    if np.any(cells == 0):
        cells += 0.5
    return float((cells[0] * cells[3]) / (cells[1] * cells[2]))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int = 500,
    max_observations: int = 100_000,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if len(clean) > max_observations:
        clean = rng.choice(clean, size=max_observations, replace=False)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        estimates[index] = clean[rng.integers(0, len(clean), size=len(clean))].mean()
    return tuple(map(float, np.quantile(estimates, [0.025, 0.975])))

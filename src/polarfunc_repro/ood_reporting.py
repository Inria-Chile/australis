from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _bootstrap_mean_interval(
    values: np.ndarray, *, repeats: int, rng: np.random.Generator
) -> tuple[float, float]:
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _summarize_values(
    values: list[float], *, repeats: int, rng: np.random.Generator
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    low, high = _bootstrap_mean_interval(array, repeats=repeats, rng=rng)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "bootstrap_mean_ci95_low": low,
        "bootstrap_mean_ci95_high": high,
    }


def summarize_ood_runs(
    summary_paths: list[Path], *, bootstrap_repeats: int = 10_000, seed: int = 42
) -> dict[str, object]:
    if len(summary_paths) < 2:
        raise ValueError("at least two OOD summaries are required")
    if bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100")

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    if any(run.get("status") != "PASS" for run in runs):
        raise ValueError("all OOD runs must have PASS status")

    detector_names = set(runs[0]["detectors"])
    reference_counts = runs[0]["function_partition_counts"]
    for run in runs[1:]:
        if set(run["detectors"]) != detector_names:
            raise ValueError("detector sets differ across OOD runs")
        if run["function_partition_counts"] != reference_counts:
            raise ValueError("function partition cardinalities differ across OOD runs")

    rng = np.random.default_rng(seed)
    classification = {
        metric: _summarize_values(
            [float(run["classification"][metric]) for run in runs],
            repeats=bootstrap_repeats,
            rng=rng,
        )
        for metric in ("accuracy", "macro_f1")
    }
    detector_metrics = (
        "auroc",
        "aupr_ood",
        "fpr_at_95_tpr",
        "test_balanced_accuracy_at_threshold",
        "test_id_false_reject_rate",
        "test_ood_false_accept_rate",
    )
    detectors = {
        detector: {
            metric: _summarize_values(
                [float(run["detectors"][detector][metric]) for run in runs],
                repeats=bootstrap_repeats,
                rng=rng,
            )
            for metric in detector_metrics
        }
        for detector in sorted(detector_names)
    }
    return {
        "status": "PASS",
        "task": "multi-seed functional OOD summary",
        "n_runs": len(runs),
        "seeds": [int(run["config"]["seed"]) for run in runs],
        "function_partition_counts": reference_counts,
        "classification": classification,
        "detectors": detectors,
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
        "source_summaries": [str(path) for path in summary_paths],
    }

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax


def _as_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1:
        values = np.column_stack([-values, values])
    if values.ndim != 2:
        raise ValueError("logits must be a two-dimensional array")
    return values


def msp_score(logits: np.ndarray) -> np.ndarray:
    """Maximum-softmax-probability OOD score; larger means more OOD."""

    return 1.0 - softmax(_as_logits(logits), axis=1).max(axis=1)


def maxlogit_score(logits: np.ndarray) -> np.ndarray:
    """Negative maximum logit, following the high-score-means-more-OOD convention."""

    return -_as_logits(logits).max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative log-sum-exp energy score; larger means more OOD."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = _as_logits(logits)
    return -temperature * logsumexp(values / temperature, axis=1)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float64).eps)


@dataclass(frozen=True)
class DistanceState:
    classes: np.ndarray
    means: np.ndarray
    global_mean: np.ndarray
    inverse_variance: np.ndarray
    normalized_means: np.ndarray


def fit_distance_state(
    x_train: np.ndarray, y_train: np.ndarray, *, variance_floor: float = 1e-4
) -> DistanceState:
    values = np.asarray(x_train, dtype=np.float64)
    labels = np.asarray(y_train)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("training features and labels have incompatible shapes")
    classes = np.unique(labels)
    means = np.vstack([values[labels == label].mean(axis=0) for label in classes])
    residuals = np.vstack([values[labels == label] - means[i] for i, label in enumerate(classes)])
    variance = np.maximum(residuals.var(axis=0, ddof=1), variance_floor)
    return DistanceState(
        classes=classes,
        means=means,
        global_mean=values.mean(axis=0),
        inverse_variance=1.0 / variance,
        normalized_means=_normalize_rows(means),
    )


def mahalanobis_score(x: np.ndarray, state: DistanceState, *, chunk_size: int = 2048) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    output = []
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        delta = chunk[:, None, :] - state.means[None, :, :]
        distances = np.einsum("ncd,d,ncd->nc", delta, state.inverse_variance, delta)
        output.append(distances.min(axis=1))
    return np.concatenate(output) if output else np.empty(0, dtype=float)


def relative_mahalanobis_score(
    x: np.ndarray, state: DistanceState, *, chunk_size: int = 2048
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    class_distance = mahalanobis_score(values, state, chunk_size=chunk_size)
    global_delta = values - state.global_mean
    global_distance = np.einsum("nd,d,nd->n", global_delta, state.inverse_variance, global_delta)
    return class_distance - global_distance


def prototype_cosine_score(x: np.ndarray, state: DistanceState) -> np.ndarray:
    values = _normalize_rows(np.asarray(x, dtype=np.float64))
    similarities = values @ state.normalized_means.T
    return 1.0 - similarities.max(axis=1)


def exact_cosine_knn_score(
    x: np.ndarray, reference: np.ndarray, *, k: int = 10, chunk_size: int = 1024
) -> np.ndarray:
    """Exact cosine kNN baseline with bounded query-by-reference memory."""

    if k < 1 or k > len(reference):
        raise ValueError("k must be between one and the reference size")
    queries = _normalize_rows(np.asarray(x, dtype=np.float64))
    database = _normalize_rows(np.asarray(reference, dtype=np.float64))
    output = []
    for start in range(0, len(queries), chunk_size):
        similarities = queries[start : start + chunk_size] @ database.T
        nearest = np.partition(similarities, similarities.shape[1] - k, axis=1)[:, -k:]
        output.append(1.0 - nearest.mean(axis=1))
    return np.concatenate(output) if output else np.empty(0, dtype=float)

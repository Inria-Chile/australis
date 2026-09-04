from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def shard_for(identifier: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16) % num_shards


def mean_pool(hidden: np.ndarray, residue_mask: np.ndarray) -> np.ndarray:
    if hidden.ndim != 3 or residue_mask.shape != hidden.shape[:2]:
        raise ValueError("Pooling shapes are incompatible")
    counts = residue_mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Cannot pool a sequence without residue tokens")
    return (hidden * residue_mask[..., None]).sum(axis=1) / counts[:, None]


def validate_embeddings(identifiers: list[str], embeddings: np.ndarray, expected_dimension: int | None = None) -> None:
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Embedding IDs are not unique")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(identifiers):
        raise ValueError("Embedding matrix shape does not match IDs")
    if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
        raise ValueError("Unexpected embedding dimension")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values")


def require_cuda(available: bool) -> None:
    if not available:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")


def completed_shard_is_valid(output_dir: Path, shard_id: int) -> bool:
    stem = f"shard_{shard_id:05d}"
    done = output_dir / f"{stem}.DONE"
    archive = output_dir / f"{stem}.npz"
    metadata = output_dir / f"{stem}.json"
    if not (done.exists() and archive.exists() and metadata.exists()):
        return False
    try:
        data = np.load(archive, allow_pickle=False)
        identifiers = data["ids"].astype(str).tolist()
        embeddings = data["embeddings"]
        meta = json.loads(metadata.read_text())
        validate_embeddings(identifiers, embeddings, int(meta["embedding_dimension"]))
        return len(identifiers) == int(meta["sequences"])
    except Exception:
        return False

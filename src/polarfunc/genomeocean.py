from __future__ import annotations

import os
from pathlib import Path

import numpy as np


MATCH_TERMS = ("genomeocean", "genome_ocean", "genome-ocean")


def bounded_asset_search(roots: list[Path], max_depth: int = 4) -> list[Path]:
    matches = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for directory, names, files in os.walk(root):
            current = Path(directory)
            depth = len(current.parts) - base_depth
            if depth >= max_depth:
                names[:] = []
            for candidate in [current, *[current / name for name in names], *[current / name for name in files]]:
                if any(term in candidate.name.lower() for term in MATCH_TERMS):
                    matches.append(candidate)
    return sorted(set(matches))


def pool_hidden(hidden: np.ndarray, residue_mask: np.ndarray, method: str) -> np.ndarray:
    if hidden.ndim != 3 or residue_mask.shape != hidden.shape[:2]:
        raise ValueError("Incompatible hidden-state and mask shapes")
    counts = residue_mask.sum(1)
    if np.any(counts == 0):
        raise ValueError("Cannot pool an empty residue sequence")
    if method == "mean":
        return (hidden * residue_mask[..., None]).sum(1) / counts[:, None]
    if method == "last-valid-token":
        positions = np.where(residue_mask, np.arange(residue_mask.shape[1]), -1).max(axis=1)
        return hidden[np.arange(len(hidden)), positions]
    raise ValueError(f"Unsupported pooling method: {method}")

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def index_npz_shards(
    paths: Iterable[Path],
    *,
    root: Path,
    representation: str,
    model_revision: str,
    expected_dimension: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    dimensions: set[int] = set()
    dtypes: set[str] = set()
    shard_count = 0
    for path in sorted(item.resolve() for item in paths):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"ids", "embeddings"}:
                raise ValueError(f"unexpected NPZ keys in {path}")
            ids = archive["ids"].astype(str)
            embeddings = np.asarray(archive["embeddings"])
        if embeddings.ndim != 2 or len(ids) != len(embeddings):
            raise ValueError(f"invalid embedding shape in {path}")
        dimension = int(embeddings.shape[1])
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError(
                f"embedding dimension mismatch in {path}: {dimension} != {expected_dimension}"
            )
        relative = path.relative_to(root).as_posix()
        for row_index, gene_id in enumerate(ids):
            if gene_id in seen:
                raise ValueError(f"duplicate embedding ID: {gene_id}")
            seen.add(gene_id)
            records.append(
                {
                    "gene_id": gene_id,
                    "representation": representation,
                    "model_revision": model_revision,
                    "shard": relative,
                    "row_index": row_index,
                    "dimension": dimension,
                    "dtype": str(embeddings.dtype),
                }
            )
        dimensions.add(dimension)
        dtypes.add(str(embeddings.dtype))
        shard_count += 1
    if not records:
        raise ValueError("no embeddings found")
    index = pd.DataFrame.from_records(records)
    summary = {
        "status": "PASS",
        "rows": len(index),
        "shards": shard_count,
        "dimensions": sorted(dimensions),
        "dtypes": sorted(dtypes),
        "unique_ids": int(index["gene_id"].nunique()),
        "representation": representation,
        "model_revision": model_revision,
    }
    return index, summary


def write_embedding_index(index: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(destination, index=False, compression="zstd")

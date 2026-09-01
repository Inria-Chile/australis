from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_embedding_manifest(
    manifest: Path,
    *,
    id_column: str,
    shard_path_column: str,
    reference_manifest: Path | None = None,
    reference_id_column: str | None = None,
    expected_rows: int | None = None,
    expected_dimension: int | None = None,
    expected_model: str | None = None,
    checksums: bool = False,
) -> dict[str, Any]:
    columns = [
        id_column,
        "shard_id",
        "row_index",
        "dimension",
        "dtype",
        "model_name",
        shard_path_column,
    ]
    table = pq.read_table(manifest, columns=columns)
    frame = table.to_pandas()
    ids = frame[id_column].astype(str)
    errors: list[str] = []

    if ids.duplicated().any():
        errors.append(f"duplicate {id_column} values")
    if expected_rows is not None and len(frame) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(frame)}")

    dimensions = sorted(int(value) for value in frame["dimension"].unique())
    dtypes = sorted(str(value) for value in frame["dtype"].unique())
    models = sorted(str(value) for value in frame["model_name"].unique())
    if expected_dimension is not None and dimensions != [expected_dimension]:
        errors.append(f"unexpected dimensions: {dimensions}")
    if expected_model is not None and models != [expected_model]:
        errors.append(f"unexpected models: {models}")

    reference_rows = None
    if reference_manifest is not None:
        reference_column = reference_id_column or id_column
        reference = pq.read_table(reference_manifest, columns=[reference_column])
        reference_ids = set(reference.column(reference_column).to_pylist())
        manifest_ids = set(ids)
        reference_rows = len(reference_ids)
        if manifest_ids != reference_ids:
            errors.append(
                "manifest/reference ID mismatch: "
                f"missing={len(reference_ids - manifest_ids)}, "
                f"extra={len(manifest_ids - reference_ids)}"
            )

    shard_records = []
    total_vectors = 0
    for shard_path_text, group in frame.groupby(shard_path_column, sort=True):
        shard_path = Path(shard_path_text)
        if not shard_path.is_file():
            errors.append(f"missing shard: {shard_path}")
            continue
        ordered = group.sort_values("row_index")
        row_indices = ordered["row_index"].to_numpy(dtype=np.int64)
        with np.load(shard_path, allow_pickle=False) as payload:
            if not {"ids", "embeddings"}.issubset(payload.files):
                errors.append(f"missing ids/embeddings arrays: {shard_path}")
                continue
            shard_ids = payload["ids"]
            embeddings = payload["embeddings"]
            if embeddings.ndim != 2:
                errors.append(f"embedding array is not 2D: {shard_path}")
                continue
            if not np.array_equal(row_indices, np.arange(len(shard_ids))):
                errors.append(f"row index coverage is not contiguous: {shard_path}")
            if len(ordered) != len(shard_ids) or len(shard_ids) != len(embeddings):
                errors.append(f"row count mismatch: {shard_path}")
            elif not np.array_equal(ordered[id_column].astype(str).to_numpy(), shard_ids.astype(str)):
                errors.append(f"ID order mismatch: {shard_path}")
            if embeddings.shape[1] not in dimensions:
                errors.append(f"dimension mismatch: {shard_path}")
            if str(embeddings.dtype) not in dtypes:
                errors.append(f"dtype mismatch: {shard_path}")
            if not np.isfinite(embeddings).all():
                errors.append(f"non-finite embedding values: {shard_path}")
            total_vectors += len(embeddings)
            shard_records.append(
                {
                    "path": str(shard_path),
                    "rows": len(embeddings),
                    "dimension": embeddings.shape[1],
                    "dtype": str(embeddings.dtype),
                    "size_bytes": shard_path.stat().st_size,
                    "sha256": _sha256(shard_path) if checksums else None,
                }
            )

    if total_vectors != len(frame):
        errors.append(f"manifest/shard total mismatch: manifest={len(frame)}, shards={total_vectors}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(manifest),
        "rows": len(frame),
        "unique_ids": ids.nunique(),
        "reference_rows": reference_rows,
        "shards": len(shard_records),
        "vectors": total_vectors,
        "dimensions": dimensions,
        "dtypes": dtypes,
        "models": models,
        "checksums_computed": checksums,
        "errors": errors,
        "shard_records": shard_records,
    }

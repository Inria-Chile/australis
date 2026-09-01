from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

MISSING_ANNOTATION_TOKENS = frozenset({"", "-", "NA", "nan"})


def iter_annotation_chunks(
    path: Path,
    *,
    usecols: list[str] | None = None,
    chunksize: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunksize,
        dtype="string[pyarrow]",
        keep_default_na=False,
        na_filter=False,
    )


def representative_rows(
    frame: pd.DataFrame, *, gene_col: str = "GENE_ID", unigene_col: str = "CDHit_ID"
) -> pd.DataFrame:
    return frame.loc[frame[gene_col] == frame[unigene_col]].copy()


def _has_value(series: pd.Series, missing_tokens: set[str]) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip()
    return ~normalized.isin(missing_tokens)


def derive_function_flags(
    frame: pd.DataFrame,
    *,
    functional_columns: Iterable[str],
    agc_category_col: str = "AGC_Cat",
    agc_singleton_category_col: str = "AGC_Singl_Cat",
    missing_tokens: Iterable[str] = MISSING_ANNOTATION_TOKENS,
) -> pd.DataFrame:
    result = frame.copy()
    tokens = set(missing_tokens)
    annotated = pd.Series(False, index=result.index)
    for column in functional_columns:
        annotated |= _has_value(result[column], tokens)
    category = result[agc_category_col].astype("string")
    effective = category.mask(
        category.eq("SINGL"), result[agc_singleton_category_col].astype("string")
    )
    result["has_functional_annotation"] = annotated
    result["broad_unknown"] = ~annotated
    result["effective_AGNOSTOS_category"] = effective
    result["agnostos_unknown"] = effective.isin(["GU", "EU"])
    result["strict_unknown"] = result["broad_unknown"] & result["agnostos_unknown"]
    result["strict_known"] = annotated & effective.eq("K")
    result["is_KWP"] = effective.eq("KWP")
    result["is_DISC"] = effective.eq("DISC")
    return result


def build_unigene_master(
    input_path: Path,
    output_dir: Path,
    *,
    functional_columns: list[str],
    chunksize: int = 1_000_000,
    max_chunks: int | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Create one Parquet row per CD-HIT representative without loading the ORF table."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    input_rows = 0
    representative_count = 0
    parts: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(iter_annotation_chunks(input_path, chunksize=chunksize)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        input_rows += len(chunk)
        representatives = derive_function_flags(
            representative_rows(chunk), functional_columns=functional_columns
        )
        representatives["canonical_AGC_ID"] = representatives["AGC_ID"]
        part_path = output_dir / f"part-{chunk_index:05d}.parquet"
        representatives.to_parquet(part_path, index=False, compression=compression)
        representative_count += len(representatives)
        parts.append(
            {
                "path": str(part_path),
                "rows": len(representatives),
                "size_bytes": part_path.stat().st_size,
            }
        )
    return {
        "input_rows": input_rows,
        "representative_rows": representative_count,
        "parts": parts,
        "elapsed_seconds": time.monotonic() - started,
    }

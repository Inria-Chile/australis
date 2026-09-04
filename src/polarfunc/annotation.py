from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import time

import pandas as pd


def iter_annotation_chunks(
    path: Path,
    *,
    usecols: list[str] | None = None,
    chunksize: int = 5_000_000,
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


def representative_rows(frame: pd.DataFrame, *, gene_col: str, cdhit_col: str) -> pd.DataFrame:
    return frame.loc[frame[gene_col] == frame[cdhit_col]].copy()


def _has_value(series: pd.Series, missing_tokens: set[str]) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip()
    return ~normalized.isin(missing_tokens)


def derive_function_flags(
    frame: pd.DataFrame,
    *,
    functional_columns: Iterable[str],
    agc_category_col: str,
    agc_singleton_category_col: str | None = None,
    missing_tokens: Iterable[str] = ("", "-", "NA", "nan"),
) -> pd.DataFrame:
    result = frame.copy()
    tokens = set(missing_tokens)
    columns = list(functional_columns)
    annotated = pd.Series(False, index=result.index)
    for column in columns:
        annotated |= _has_value(result[column], tokens)
    result["broad_unknown"] = ~annotated
    category = result[agc_category_col].astype("string")
    effective = category.copy()
    if agc_singleton_category_col:
        singleton = category.eq("SINGL")
        effective = effective.mask(singleton, result[agc_singleton_category_col].astype("string"))
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
    chunksize: int = 5_000_000,
    max_chunks: int | None = None,
    compression: str = "zstd",
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    input_rows = 0
    representative_count = 0
    parts: list[dict[str, object]] = []
    for chunk_index, chunk in enumerate(iter_annotation_chunks(input_path, chunksize=chunksize)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        input_rows += len(chunk)
        representatives = representative_rows(chunk, gene_col="GENE_ID", cdhit_col="CDHit_ID")
        representatives = derive_function_flags(
            representatives,
            functional_columns=functional_columns,
            agc_category_col="AGC_Cat",
            agc_singleton_category_col="AGC_Singl_Cat",
        )
        representatives["canonical_AGC_ID"] = representatives["AGC_ID"]
        part_path = output_dir / f"part-{chunk_index:05d}.parquet"
        representatives.to_parquet(part_path, index=False, compression=compression)
        representative_count += len(representatives)
        parts.append({"path": str(part_path), "rows": len(representatives), "size_bytes": part_path.stat().st_size})
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "input_rows": input_rows,
        "representative_rows": representative_count,
        "parts": parts,
        "chunksize": chunksize,
        "max_chunks": max_chunks,
        "elapsed_seconds": time.monotonic() - started,
    }


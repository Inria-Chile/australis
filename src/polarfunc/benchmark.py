from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


VALID_FRACTIONS = {"FL", "ATT"}
VALID_FUNCTION_LABELS = {"strict_known", "strict_unknown"}
VALID_ECOLOGY_LABELS = {"high", "low"}


def benchmark_join_query(paths: dict[str, object]) -> str:
    return f"""
      SELECT g.*, c.AGC_ID, c.R2_caret, q.length_nt, q.GC_fraction, q.sequence_sha256, q.protein_length,
             c.AGC_n_unigenes, ln(1 + c.AGC_n_unigenes) AS log1p_AGC_n_unigenes,
             CASE WHEN g.fraction='FL' THEN a.prevalence_FL ELSE a.prevalence_ATT END AS prevalence,
             CASE WHEN g.fraction='FL' THEN a.mean_abundance_FL ELSE a.mean_abundance_ATT END AS mean_abundance,
             ln(1 + CASE WHEN g.fraction='FL' THEN a.mean_abundance_FL ELSE a.mean_abundance_ATT END) AS log1p_mean_abundance,
             coalesce(nullif(trim(t.Domain), ''), 'Unknown') AS Domain,
             coalesce(nullif(trim(t.Phylum), ''), 'Unknown') AS Phylum,
             coalesce(t.Domain_conflict_fraction, 0) > 0 AS domain_conflict
      FROM read_parquet('{paths['groups']}') g
      JOIN read_parquet('{paths['qc']}') q ON q.CDHit_ID = g.CDHit_ID
      JOIN read_parquet('{paths['candidates']}') c ON c.CDHit_ID = g.CDHit_ID
      LEFT JOIN read_parquet('{paths['abundance']}') a ON a.AGC_ID = c.AGC_ID
      LEFT JOIN read_parquet('{paths['taxonomy']}') t ON t.AGC_ID = c.AGC_ID
      WHERE q.protein_valid
    """


def stable_hash(value: str, seed: int = 20260830) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def add_matching_bins(frame: pd.DataFrame, *, quantiles: int = 2, top_phyla: int = 10) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    numeric = ["length_nt", "GC_fraction", "protein_length", "log1p_AGC_n_unigenes", "prevalence", "log1p_mean_abundance"]
    bin_columns = []
    for column in numeric:
        if result[column].isna().any():
            result[column] = result[column].fillna(result[column].median())
        output = f"{column}_bin"
        result[output] = pd.qcut(result[column], quantiles, labels=False, duplicates="drop").fillna(0).astype("int8")
        bin_columns.append(output)
    phylum = result.Phylum.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    keep = set(phylum.value_counts().head(top_phyla).index)
    result["Phylum_bin"] = phylum.where(phylum.isin(keep), "Other")
    return result, [*bin_columns, "Phylum_bin"]


def coarsened_exact_match(frame: pd.DataFrame, bin_columns: list[str], *, seed: int = 20260830) -> pd.DataFrame:
    base = ["fraction", "ecology_label", *bin_columns]
    result = frame.copy()
    result["selection_hash"] = result.CDHit_ID.astype(str).map(lambda value: stable_hash(value, seed))
    counts = result.groupby(base + ["function_label"], dropna=False).size().unstack(fill_value=0)
    for label in VALID_FUNCTION_LABELS:
        if label not in counts:
            counts[label] = 0
    quotas = counts[list(sorted(VALID_FUNCTION_LABELS))].min(axis=1).rename("cell_quota").reset_index()
    result = result.merge(quotas, on=base, how="left", validate="many_to_one")
    result = result.sort_values(base + ["function_label", "selection_hash"])
    result["cell_rank"] = result.groupby(base + ["function_label"], dropna=False).cumcount()
    return result.loc[result.cell_rank < result.cell_quota].copy()


def assign_split_groups(frame: pd.DataFrame, *, seed: int = 20260830) -> pd.DataFrame:
    groups = frame.groupby("split_group", sort=False).agg(group_size=("CDHit_ID", "size")).reset_index()
    groups["assignment_hash"] = groups.split_group.map(lambda value: stable_hash(str(value), seed))
    bucket = groups.assignment_hash.str[:8].map(lambda value: int(value, 16) / 0xFFFFFFFF)
    groups["split"] = np.select([bucket < 0.70, bucket < 0.85], ["train", "validation"], default="test")
    return groups


def select_nested_groups(groups: pd.DataFrame, targets: list[int]) -> dict[int, set[str]]:
    ordered = groups.sort_values("assignment_hash")
    selected: set[str] = set()
    selected_count = 0
    result: dict[int, set[str]] = {}
    for target in sorted(targets):
        for row in ordered.itertuples(index=False):
            if row.split_group in selected:
                continue
            if selected_count + int(row.group_size) <= target:
                selected.add(row.split_group)
                selected_count += int(row.group_size)
                if selected_count == target:
                    break
        if selected_count != target:
            raise ValueError(f"Cannot select exactly {target} rows without splitting groups; reached {selected_count}")
        result[target] = set(selected)
    return result


def assert_no_split_leakage(frame: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "CDHit_ID_unique": bool(frame.CDHit_ID.is_unique),
        "AGC_ID_unique": bool(frame.AGC_ID.is_unique),
        "sequence_sha256_unique": bool(frame.sequence_sha256.is_unique),
        "split_group_one_partition": bool(frame.groupby("split_group").split.nunique().le(1).all()),
        "target_labels_nonmissing": not bool(frame[["function_label", "ecology_label", "fraction"]].isna().any().any()),
        "fraction_labels_valid": set(frame.fraction).issubset(VALID_FRACTIONS),
        "function_labels_valid": set(frame.function_label).issubset(VALID_FUNCTION_LABELS),
        "ecology_labels_valid": set(frame.ecology_label).issubset(VALID_ECOLOGY_LABELS),
    }
    return checks

from __future__ import annotations

import pandas as pd


def build_cdhit_agc_pairs(
    annotation: pd.DataFrame,
    *,
    gene_col: str = "GENE_ID",
    cdhit_col: str = "CDHit_ID",
    agc_col: str = "AGC_ID",
) -> pd.DataFrame:
    """Collapse ORF rows to one row per observed unigene/AGC pair."""
    frame = annotation[[gene_col, cdhit_col, agc_col]].copy()
    frame["has_representative"] = frame[gene_col].eq(frame[cdhit_col])
    pairs = (
        frame.groupby([cdhit_col, agc_col], dropna=False, sort=True)
        .agg(n_ORFs=(gene_col, "size"), has_representative=("has_representative", "any"))
        .reset_index()
    )
    pairs[agc_col] = pairs[agc_col].fillna("")
    return pairs


def summarize_cdhit_agc_cardinality(
    pairs: pd.DataFrame,
    *,
    cdhit_col: str = "CDHit_ID",
    agc_col: str = "AGC_ID",
    example_limit: int = 20,
) -> dict[str, object]:
    valid = pairs.loc[pairs[agc_col].fillna("").str.strip().ne("")]
    counts = valid.groupby(cdhit_col, sort=False)[agc_col].nunique()
    all_unigenes = pd.Index(pairs[cdhit_col].drop_duplicates())
    counts = counts.reindex(all_unigenes, fill_value=0)
    multi_ids = counts[counts.gt(1)].index[:example_limit]
    examples = pairs.loc[pairs[cdhit_col].isin(multi_ids)].to_dict("records")
    return {
        "unigenes": int(len(counts)),
        "unique_mapping": int(counts.eq(1).sum()),
        "multi_mapping": int(counts.gt(1).sum()),
        "missing_agc": int(counts.eq(0).sum()),
        "max_agcs_per_unigene": int(counts.max()) if len(counts) else 0,
        "multi_mapping_examples": examples,
    }


def select_canonical_agc(
    pairs: pd.DataFrame,
    *,
    cdhit_col: str = "CDHit_ID",
    agc_col: str = "AGC_ID",
) -> pd.DataFrame:
    """Select one AGC per unigene with an auditable deterministic policy."""
    selections: list[dict[str, object]] = []
    for cdhit_id, group in pairs.groupby(cdhit_col, sort=False):
        candidates = group.loc[group[agc_col].fillna("").str.strip().ne("")].copy()
        if candidates.empty:
            selections.append(
                {cdhit_col: cdhit_id, "canonical_AGC_ID": "", "selection_reason": "missing_agc"}
            )
            continue
        representatives = candidates.loc[candidates["has_representative"]]
        if not representatives.empty:
            ranked = representatives.sort_values(["n_ORFs", agc_col], ascending=[False, True])
            reason = "representative_row" if len(representatives) == 1 else "representative_tiebreak"
        else:
            ranked = candidates.sort_values(["n_ORFs", agc_col], ascending=[False, True])
            top_support = int(ranked.iloc[0]["n_ORFs"])
            reason = (
                "largest_orf_support"
                if int(candidates["n_ORFs"].eq(top_support).sum()) == 1
                else "lexical_tiebreak"
            )
        selections.append(
            {
                cdhit_col: cdhit_id,
                "canonical_AGC_ID": ranked.iloc[0][agc_col],
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(selections)


def _stable_mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values.loc[values.str.strip().ne("")]
    if values.empty:
        return ""
    counts = values.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def build_agc_stats(
    pairs: pd.DataFrame,
    unigenes: pd.DataFrame,
    *,
    cdhit_col: str = "CDHit_ID",
    agc_col: str = "AGC_ID",
) -> pd.DataFrame:
    """Build AGC statistics using canonical unigene assignments."""
    valid_pairs = pairs.loc[pairs[agc_col].fillna("").str.strip().ne("")]
    orf_counts = valid_pairs.groupby(agc_col, sort=True)["n_ORFs"].sum().rename("AGC_n_ORFs")
    canonical = select_canonical_agc(pairs, cdhit_col=cdhit_col, agc_col=agc_col)
    assigned = canonical.merge(unigenes, on=cdhit_col, how="left", validate="one_to_one")
    assigned = assigned.loc[assigned["canonical_AGC_ID"].fillna("").str.strip().ne("")]
    grouped = assigned.groupby("canonical_AGC_ID", sort=True)
    stats = grouped.size().rename("AGC_n_unigenes").to_frame()
    for source, target in (
        ("strict_known", "AGC_n_strict_known_unigenes"),
        ("strict_unknown", "AGC_n_strict_unknown_unigenes"),
    ):
        if source in assigned:
            stats[target] = grouped[source].sum().astype("int64")
    if "effective_AGNOSTOS_category" in assigned:
        stats["effective_AGNOSTOS_category"] = grouped["effective_AGNOSTOS_category"].agg(_stable_mode)
    stats.index.name = agc_col
    stats = stats.join(orf_counts, how="outer").fillna(
        {
            "AGC_n_unigenes": 0,
            "AGC_n_strict_known_unigenes": 0,
            "AGC_n_strict_unknown_unigenes": 0,
        }
    )
    stats["AGC_n_unigenes"] = stats["AGC_n_unigenes"].astype("int64")
    stats["AGC_n_ORFs"] = stats["AGC_n_ORFs"].astype("int64")
    stats["AGC_unigene_ORF_ratio"] = stats["AGC_n_unigenes"] / stats["AGC_n_ORFs"]
    if "AGC_n_strict_unknown_unigenes" in stats:
        denominator = stats["AGC_n_unigenes"].replace(0, pd.NA)
        stats["AGC_fraction_strict_unknown"] = stats["AGC_n_strict_unknown_unigenes"] / denominator
    return stats.reset_index()

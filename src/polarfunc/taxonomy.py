from __future__ import annotations

import pandas as pd


RANKS = ("Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species")


def contig_id_from_gene_id(gene_id: pd.Series) -> pd.Series:
    """Extract the ACE contig key while preserving missing/malformed IDs."""
    return gene_id.astype("string").str.extract(r"^ACE_(G[^_]+_[^_]+)_[^_]+$", expand=False)


def parse_gtdb_lineage(lineage: pd.Series) -> pd.DataFrame:
    cleaned = lineage.astype("string").fillna("").str.strip()
    parts = cleaned.str.split(r"\s*;\s*", regex=True)
    parsed = pd.DataFrame(index=lineage.index)
    for offset, rank in enumerate(RANKS, start=1):
        parsed[f"GTDB_{rank}"] = parts.str.get(offset).fillna("")
    invalid = cleaned.isin(["", "unclassified", "root"])
    parsed.loc[invalid, :] = ""
    return parsed


def domain_conflict(annotation_domain: pd.Series, gtdb_domain: pd.Series) -> pd.Series:
    annotation = annotation_domain.astype("string").fillna("").str.strip().str.lower()
    gtdb = gtdb_domain.astype("string").fillna("").str.strip().str.lower()
    compatible = annotation.eq(gtdb)
    compatible |= annotation.eq("prokaryote") & gtdb.isin(["bacteria", "archaea"])
    compatible |= annotation.eq("eukaryote") & gtdb.isin(["eukaryota", "eukarya"])
    return annotation.ne("") & gtdb.ne("") & ~compatible


def modal_taxonomy(frame: pd.DataFrame, *, group_col: str, ranks: tuple[str, ...] = RANKS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, subset in frame.groupby(group_col, sort=False, observed=True):
        row: dict[str, object] = {group_col: group, "n_unigenes": len(subset)}
        for rank in ranks:
            values = subset[rank].astype("string").fillna("").str.strip()
            values = values.loc[values.ne("")]
            row[rank] = pd.NA
            row[f"{rank}_coverage"] = len(values) / len(subset)
            row[f"{rank}_purity"] = pd.NA
            if not values.empty:
                counts = values.value_counts()
                maximum = counts.max()
                row[rank] = sorted(counts.loc[counts.eq(maximum)].index)[0]
                row[f"{rank}_purity"] = maximum / len(values)
        rows.append(row)
    return pd.DataFrame(rows)

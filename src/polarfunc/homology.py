from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def stable_cluster_group(representative: str) -> str:
    return "MMSEQ_" + hashlib.sha256(representative.encode("utf-8")).hexdigest()[:16]


def parse_mmseqs_clusters(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", names=["mmseqs_representative", "CDHit_ID"], dtype=str)
    if frame.empty:
        raise ValueError("MMseqs cluster table is empty")
    if frame.CDHit_ID.duplicated().any():
        duplicate = frame.loc[frame.CDHit_ID.duplicated(), "CDHit_ID"].iloc[0]
        raise ValueError(f"Protein belongs to multiple MMseqs clusters: {duplicate}")
    frame["split_group"] = frame.mmseqs_representative.map(stable_cluster_group)
    sizes = frame.groupby("mmseqs_representative").size().rename("cluster_size")
    return frame.join(sizes, on="mmseqs_representative")


def cluster_mixing_summary(mapping: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, object]:
    joined = mapping.merge(metadata, on="CDHit_ID", how="left", validate="one_to_one")
    if joined.function_label.isna().any():
        raise ValueError("Missing functional label for an MMseqs member")
    cluster = joined.groupby("mmseqs_representative", sort=False).agg(
        cluster_size=("CDHit_ID", "size"),
        function_labels=("function_label", "nunique"),
        ecology_labels=("ecology_label", "nunique"),
        fractions=("fraction", "nunique"),
    )
    return {
        "clusters": int(len(cluster)),
        "proteins": int(len(joined)),
        "singletons": int(cluster.cluster_size.eq(1).sum()),
        "singleton_fraction": float(cluster.cluster_size.eq(1).mean()),
        "largest_cluster": int(cluster.cluster_size.max()),
        "function_label_mixed_clusters": int(cluster.function_labels.gt(1).sum()),
        "ecology_label_mixed_clusters": int(cluster.ecology_labels.gt(1).sum()),
        "fraction_mixed_clusters": int(cluster.fractions.gt(1).sum()),
    }

from __future__ import annotations

import pandas as pd


def summarize_multi_agc(pairs: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    required_pairs = {"CDHit_ID", "AGC_ID"}
    required_canonical = {"CDHit_ID", "canonical_AGC_ID"}
    if not required_pairs.issubset(pairs.columns):
        raise ValueError(f"Pair columns missing: {sorted(required_pairs - set(pairs.columns))}")
    if not required_canonical.issubset(canonical.columns):
        raise ValueError(
            f"Canonical columns missing: {sorted(required_canonical - set(canonical.columns))}"
        )
    unique_pairs = pairs.loc[:, ["CDHit_ID", "AGC_ID"]].drop_duplicates()
    counts = (
        unique_pairs.groupby("CDHit_ID", sort=False)["AGC_ID"]
        .nunique()
        .rename("n_AGCs")
        .astype("int32")
        .reset_index()
    )
    result = counts.merge(
        canonical.loc[:, ["CDHit_ID", "canonical_AGC_ID"]],
        on="CDHit_ID",
        how="left",
        validate="one_to_one",
    )
    membership = unique_pairs.rename(columns={"AGC_ID": "canonical_AGC_ID"}).assign(
        canonical_is_associated=True
    )
    result = result.merge(
        membership,
        on=["CDHit_ID", "canonical_AGC_ID"],
        how="left",
        validate="one_to_one",
    )
    if result["canonical_is_associated"].isna().any():
        raise ValueError("Canonical AGC is not associated with every unigene")
    result["is_multi_agc"] = result["n_AGCs"] > 1
    return result.drop(columns="canonical_is_associated")


def star_edges(agc_ids: list[str]) -> list[tuple[str, str]]:
    unique = sorted(set(agc_ids))
    if len(unique) < 2:
        return []
    anchor = unique[0]
    return [(anchor, other) for other in unique[1:]]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_components(edges: list[tuple[str, str]]) -> pd.DataFrame:
    identifiers = sorted({identifier for edge in edges for identifier in edge})
    index = {identifier: position for position, identifier in enumerate(identifiers)}
    union_find = UnionFind(len(identifiers))
    for left, right in edges:
        union_find.union(index[left], index[right])
    roots = [union_find.find(position) for position in range(len(identifiers))]
    members: dict[int, list[int]] = {}
    for position, root in enumerate(roots):
        members.setdefault(root, []).append(position)
    ordered = sorted(members.values(), key=lambda positions: identifiers[min(positions)])
    records = []
    for component_number, positions in enumerate(ordered):
        component_id = f"MAC{component_number:08d}"
        size = len(positions)
        records.extend(
            {
                "AGC_ID": identifiers[position],
                "component_id": component_id,
                "component_size": size,
                "is_cross_agc_component": size > 1,
            }
            for position in positions
        )
    return pd.DataFrame.from_records(records)

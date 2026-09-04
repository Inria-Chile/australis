from __future__ import annotations

import numpy as np
import pandas as pd


def build_water_mass_specificity(
    abundance: pd.DataFrame,
    grouping: pd.DataFrame,
    *,
    fraction: str,
) -> pd.DataFrame:
    environment = grouping.set_index("ACE_seq_name")
    sample_columns = [column for column in abundance.columns if column != "CAG_ID"]
    missing = set(sample_columns) - set(environment.index)
    if missing:
        raise ValueError(f"Abundance samples absent from environment: {sorted(missing)[:5]}")
    water_masses = sorted(environment.loc[sample_columns, "Water_mass_simplified"].dropna().unique())
    means = pd.DataFrame(index=abundance.index)
    for water_mass in water_masses:
        samples = [
            sample for sample in sample_columns if environment.at[sample, "Water_mass_simplified"] == water_mass
        ]
        means[water_mass] = abundance[samples].mean(axis=1)
    total = means.sum(axis=1)
    probabilities = means.div(total.replace(0, np.nan), axis=0)
    entropy = -(probabilities * np.log(probabilities.where(probabilities.gt(0)))).sum(axis=1)
    specificity = 1.0 - entropy / np.log(len(water_masses))
    result = pd.DataFrame(
        {
            "CAG_ID": abundance["CAG_ID"].astype("string"),
            "fraction": fraction.upper(),
            "water_mass_count": len(water_masses),
            "water_mass_specificity": specificity.mask(total.eq(0)),
            "dominant_water_mass": means.idxmax(axis=1).mask(total.eq(0)),
            "dominant_water_mass_share": probabilities.max(axis=1),
            "sum_water_mass_mean_abundance": total,
        }
    )
    return result

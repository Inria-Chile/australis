#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.cag_ecology import aggregate_cag_composition, fit_ols
from polarfunc.provenance import sha256_file, utc_now, write_json


def group_effect(known: np.ndarray, unknown: np.ndarray, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    known = known[np.isfinite(known)]
    unknown = unknown[np.isfinite(unknown)]
    known_sample = rng.choice(known, min(len(known), 200_000), replace=False)
    unknown_sample = rng.choice(unknown, min(len(unknown), 200_000), replace=False)
    test = mannwhitneyu(unknown_sample, known_sample, alternative="two-sided")
    delta = 2 * float(test.statistic) / (len(unknown_sample) * len(known_sample)) - 1
    return {
        "N_K": len(known), "N_GU_EU": len(unknown), "sample_N_K": len(known_sample),
        "sample_N_GU_EU": len(unknown_sample), "mann_whitney_U": float(test.statistic),
        "p_value": float(test.pvalue), "cliffs_delta_GU_EU_minus_K": delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CAG-level water-mass specificity analysis.")
    parser.add_argument("--water-mass-dir", type=Path, default=ROOT / "artifacts/ecology/water_mass")
    args = parser.parse_args()
    started_at = utc_now()
    output = ROOT / "artifacts/results/cag_macro_watermass.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite validated output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    models = []
    report: dict[str, object] = {"scope": "env-AGC-derived CAG universe", "fractions": {}}

    for index, fraction in enumerate(["FL", "ATT"]):
        source = args.water_mass_dir / f"agc_water_mass_specificity_{fraction}.parquet"
        agcs = pd.read_parquet(source)
        cags = aggregate_cag_composition(agcs)
        cags["log1p_CAG_size"] = np.log1p(cags["CAG_size"])
        model = fit_ols(
            cags,
            outcome="water_mass_specificity",
            predictors=["unknown_dominated_fraction", "log1p_CAG_size"],
        )
        model.insert(0, "fraction", fraction)
        models.append(model)
        k = cags.loc[cags.dominant_AGNOSTOS_category.eq("K"), "water_mass_specificity"].to_numpy(float)
        gu_eu = cags.loc[cags.dominant_AGNOSTOS_category.isin(["GU", "EU"]), "water_mass_specificity"].to_numpy(float)
        effect = group_effect(k, gu_eu, seed=2026 + index)
        category_fraction_columns = [column for column in cags if column.startswith("fraction_")]
        checks = {key: bool(value) for key, value in {
            "one_row_per_composite_cag": not cags.duplicated(["fraction", "CAG_ID"]).any(),
            "specificity_bounded": cags.water_mass_specificity.dropna().between(0, 1).all(),
            "composition_fraction_sum_le_one": cags[category_fraction_columns].sum(axis=1).le(1.0 + 1e-9).all(),
            "model_coefficients_finite": np.isfinite(model[["estimate", "ci_low", "ci_high"]].to_numpy()).all(),
        }.items()}
        if not all(checks.values()):
            raise RuntimeError(f"W2-06 {fraction} validation failure: {checks}")
        report["fractions"][fraction] = {
            "CAGs": len(cags),
            "AGCs": int(cags.CAG_size.sum()),
            "specificity_mean": float(cags.water_mass_specificity.mean()),
            "specificity_median": float(cags.water_mass_specificity.median()),
            "unknown_fraction_coefficient": model.loc[model.term.eq("unknown_dominated_fraction")].to_dict(orient="records")[0],
            "K_vs_GU_EU": effect,
            "validation": checks,
        }
        frames.append(cags)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    model_table = pd.concat(models, ignore_index=True)
    model_path = ROOT / "artifacts/results/cag_macro_watermass_models.tsv"
    model_table.to_csv(model_path, sep="\t", index=False)
    report["validation"] = {
        "passed": all(all(values["validation"].values()) for values in report["fractions"].values()),
        "checks": {f"{fraction}_{name}": value for fraction, values in report["fractions"].items() for name, value in values["validation"].items()},
    }
    report_json = ROOT / "reports/wave2/cag_macro_watermass.json"
    report_md = ROOT / "reports/wave2/cag_macro_watermass.md"
    write_json(report_json, report)
    report_md.write_text(
        "# CAG-macro water-mass specificity\n\n"
        "Unit of inference: one `(fraction, CAG_ID)` in the env-AGC-derived CAG universe.\n\n"
        + "\n".join(
            f"- {fraction}: {values['CAGs']:,} CAGs; unknown-fraction beta="
            f"{values['unknown_fraction_coefficient']['estimate']:.4f} "
            f"[{values['unknown_fraction_coefficient']['ci_low']:.4f}, {values['unknown_fraction_coefficient']['ci_high']:.4f}]; "
            f"GU/EU-vs-K Cliff delta={values['K_vs_GU_EU']['cliffs_delta_GU_EU_minus_K']:.4f}."
            for fraction, values in report["fractions"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = ROOT / "figures/inach_cpu/cag_macro_watermass.pdf"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, fraction in zip(axes, ["FL", "ATT"], strict=True):
        subset = combined.loc[combined.fraction.eq(fraction)]
        axis.hexbin(subset.unknown_dominated_fraction, subset.water_mass_specificity, gridsize=35, mincnt=1, cmap="viridis")
        axis.set(title=fraction, xlabel="Unknown-dominated AGC fraction", ylabel="Water-mass specificity")
    fig.savefig(figure_path)
    plt.close(fig)

    inputs = [args.water_mass_dir / f"agc_water_mass_specificity_{fraction}.parquet" for fraction in ["FL", "ATT"]]
    outputs = [output, model_path, report_json, report_md, figure_path]
    manifest = {
        "task_id": "W2-06", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": 1,
        "inputs": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in inputs],
        "parameters": {"unit": "composite CAG key", "scope": report["scope"]},
        "outputs": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "validation": report["validation"], "notes": ["CAG universe is downstream of env-AGC selection."],
    }
    write_json(ROOT / "manifests/wave2/W2-06/manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

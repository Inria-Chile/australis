#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.adjusted_ecology import fit_logistic, numeric_vif
from polarfunc.provenance import sha256_file, utc_now, write_json


def quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def analysis_frame(connection: duckdb.DuckDBPyConnection, fraction: str) -> pd.DataFrame:
    ecology = ROOT / f"artifacts/ecology/agc_ecology_{fraction}.parquet"
    abundance = ROOT / f"artifacts/wave2/agc_abundance_covariates_{fraction}.parquet"
    taxonomy = ROOT / "artifacts/wave2/agc_taxonomy.parquet"
    return connection.execute(
        f"""
        SELECT e.AGC_ID, e.R2_caret, e.env_AGC_paper, e.strong_env_AGC,
               e.effective_AGNOSTOS_category, e.AGC_n_unigenes,
               e.AGC_n_strict_known_unigenes, e.AGC_n_strict_unknown_unigenes,
               a.prevalence, a.mean_abundance, t.Domain, t.Phylum
        FROM read_parquet({quote(ecology)}) e
        LEFT JOIN read_parquet({quote(abundance)}) a USING (AGC_ID)
        LEFT JOIN read_parquet({quote(taxonomy)}) t USING (AGC_ID)
        """
    ).fetchdf()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run confounder-adjusted known-versus-unknown ACE ecology models.")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="128GB")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    started_at = utc_now()
    suffix = "_probe" if args.max_rows is not None else ""
    output = ROOT / f"artifacts/results/adjusted_ecology{suffix}.parquet"
    report_json = ROOT / f"reports/wave2/adjusted_ecology{suffix}.json"
    report_md = ROOT / f"reports/wave2/adjusted_ecology{suffix}.md"
    figure = ROOT / f"figures/inach_cpu/adjusted_ecology_forestplot{suffix}.pdf"
    for path in (output, report_json, report_md, figure):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={quote(args.memory_limit)}")
    rows = []
    report: dict[str, object] = {"language": "associational, not causal", "fractions": {}, "probe": args.max_rows is not None}
    comparisons = [("K", "GU"), ("K", "EU"), ("K", "KWP"), ("GU", "EU")]
    base_covariates = ["log1p_AGC_n_unigenes", "prevalence", "log1p_mean_abundance", "C(Domain)"]
    for fraction in ("FL", "ATT"):
        frame = analysis_frame(connection, fraction)
        if args.max_rows is not None:
            frame = frame.head(args.max_rows).copy()
        counts = {"joined_rows": len(frame), "missing_R2": int(frame.R2_caret.isna().sum()), "missing_abundance": int(frame.prevalence.isna().sum()), "missing_Domain": int(frame.Domain.isna().sum())}
        frame = frame.loc[frame.R2_caret.notna()].copy()
        frame["log1p_AGC_n_unigenes"] = np.log1p(pd.to_numeric(frame.AGC_n_unigenes, errors="coerce"))
        frame["log1p_mean_abundance"] = np.log1p(pd.to_numeric(frame.mean_abundance, errors="coerce"))
        frame["Domain"] = frame.Domain.astype("string").replace("", pd.NA).astype("object")
        frame["unknown_dominated"] = np.where(
            frame.AGC_n_strict_unknown_unigenes > frame.AGC_n_strict_known_unigenes, 1,
            np.where(frame.AGC_n_strict_known_unigenes > frame.AGC_n_strict_unknown_unigenes, 0, np.nan),
        )
        primary = frame.loc[frame.unknown_dominated.notna()].copy()
        fraction_report: dict[str, object] = {"counts": counts, "models": {}, "numeric_vif": numeric_vif(primary, ["log1p_AGC_n_unigenes", "prevalence", "log1p_mean_abundance"])}
        for outcome in ("strong_env_AGC", "env_AGC_paper"):
            model, model_counts = fit_logistic(primary, outcome=outcome, predictor="unknown_dominated", covariates=base_covariates)
            model.insert(0, "comparison", "strict_unknown_vs_strict_known")
            model.insert(0, "outcome", outcome)
            model.insert(0, "fraction", fraction)
            rows.append(model)
            effect = model.loc[model.term.eq("unknown_dominated")].iloc[0]
            fraction_report["models"][outcome] = {"counts": model_counts, "odds_ratio": float(effect.odds_ratio), "ci_low": float(effect.ci_low), "ci_high": float(effect.ci_high)}
        for reference, comparison in comparisons:
            subset = frame.loc[frame.effective_AGNOSTOS_category.isin([reference, comparison])].copy()
            subset["category_contrast"] = subset.effective_AGNOSTOS_category.eq(comparison).astype(int)
            model, model_counts = fit_logistic(subset, outcome="strong_env_AGC", predictor="category_contrast", covariates=base_covariates)
            label = f"{comparison}_vs_{reference}"
            model.insert(0, "comparison", label)
            model.insert(0, "outcome", "strong_env_AGC")
            model.insert(0, "fraction", fraction)
            rows.append(model)
            effect = model.loc[model.term.eq("category_contrast")].iloc[0]
            fraction_report["models"][label] = {"counts": model_counts, "odds_ratio": float(effect.odds_ratio), "ci_low": float(effect.ci_low), "ci_high": float(effect.ci_high)}
        report["fractions"][fraction] = fraction_report
    results = pd.concat(rows, ignore_index=True)
    results.to_parquet(output, index=False, compression="zstd")
    effects = results.loc[results.term.isin(["unknown_dominated", "category_contrast"])].copy()
    validation_checks = {
        "both_fractions": set(effects.fraction) == {"FL", "ATT"},
        "odds_ratios_finite_positive": bool(np.isfinite(effects[["odds_ratio", "ci_low", "ci_high"]]).all().all() and effects.odds_ratio.gt(0).all()),
        "primary_models_present": len(effects.loc[effects.comparison.eq("strict_unknown_vs_strict_known")]) == 4,
        "no_fraction_concatenation": not effects.duplicated(["fraction", "outcome", "comparison"]).any(),
    }
    report["validation"] = {"passed": all(validation_checks.values()), "checks": validation_checks}
    write_json(report_json, report)
    report_md.write_text(
        "# Adjusted known-versus-unknown ecology\n\n"
        "Separate associational logistic models for FL and ATT; no causal interpretation. Covariates: log1p AGC size, prevalence, log1p mean TMAX60 coverage, and Domain.\n\n"
        + "\n".join(
            f"- {fraction} {outcome}: strict-unknown vs strict-known OR={values['models'][outcome]['odds_ratio']:.3f} "
            f"[{values['models'][outcome]['ci_low']:.3f}, {values['models'][outcome]['ci_high']:.3f}]"
            for fraction, values in report["fractions"].items() for outcome in ("strong_env_AGC", "env_AGC_paper")
        ) + "\n",
        encoding="utf-8",
    )
    primary_effects = effects.loc[effects.comparison.eq("strict_unknown_vs_strict_known")].copy()
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = (primary_effects.fraction + " / " + primary_effects.outcome).tolist()
    positions = np.arange(len(primary_effects))
    axis.errorbar(primary_effects.odds_ratio, positions, xerr=[primary_effects.odds_ratio - primary_effects.ci_low, primary_effects.ci_high - primary_effects.odds_ratio], fmt="o", color="#167a73", capsize=3)
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.set(xscale="log", yticks=positions, yticklabels=labels, xlabel="Adjusted odds ratio", title="Strict-unknown vs strict-known AGCs")
    fig.savefig(figure)
    plt.close(fig)
    if not report["validation"]["passed"]:
        raise RuntimeError(f"W2-05 validation failure: {validation_checks}")
    outputs = [output, report_json, report_md, figure]
    manifest = {
        "task_id": "W2-05-PROBE" if args.max_rows is not None else "W2-05", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": args.threads,
        "inputs": [], "parameters": {"primary_covariates": base_covariates, "max_rows": args.max_rows},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "validation": report["validation"],
    }
    write_json(ROOT / "manifests/wave2" / manifest["task_id"] / "manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

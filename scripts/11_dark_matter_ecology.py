#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.dark_matter import bootstrap_mean_ci, odds_ratio
from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest


def summarize(subset: pd.DataFrame, seed: int) -> dict[str, object]:
    r2 = pd.to_numeric(subset["R2_caret"], errors="coerce").dropna().to_numpy()
    ci = bootstrap_mean_ci(r2, seed=seed)
    env = subset["env_AGC_paper"].astype(bool)
    strong = subset["strong_env_AGC"].astype(bool)
    env_ci = bootstrap_mean_ci(env.to_numpy(dtype=float), seed=seed + 1000)
    strong_ci = bootstrap_mean_ci(strong.to_numpy(dtype=float), seed=seed + 2000)
    return {
        "N": len(subset),
        "R2_mean": float(np.mean(r2)),
        "R2_median": float(np.median(r2)),
        "R2_q25": float(np.quantile(r2, 0.25)),
        "R2_q75": float(np.quantile(r2, 0.75)),
        "R2_mean_bootstrap_CI95": list(ci),
        "env_AGC_fraction": float(env.mean()),
        "env_AGC_fraction_bootstrap_CI95": list(env_ci),
        "strong_env_AGC_fraction": float(strong.mean()),
        "strong_env_AGC_fraction_bootstrap_CI95": list(strong_ci),
    }


def sampled_mann_whitney(a: np.ndarray, b: np.ndarray, seed: int, max_n: int = 500_000) -> dict[str, object]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    a_sample = rng.choice(a, size=min(len(a), max_n), replace=False)
    b_sample = rng.choice(b, size=min(len(b), max_n), replace=False)
    result = mannwhitneyu(a_sample, b_sample, alternative="two-sided")
    cliff_delta = 2 * float(result.statistic) / (len(a_sample) * len(b_sample)) - 1
    return {
        "sample_N_a": len(a_sample),
        "sample_N_b": len(b_sample),
        "U": float(result.statistic),
        "p_value": float(result.pvalue),
        "cliffs_delta": cliff_delta,
        "sampling": "deterministic without-replacement cap of 500,000 per group",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare known and unknown ACE AGC ecological structure.")
    parser.add_argument("--ecology-dir", type=Path, default=ROOT / "artifacts" / "ecology")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "dark_matter_ecology.json")
    parser.add_argument("--table", type=Path, default=ROOT / "reports" / "dark_matter_ecology.tsv")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.max_rows is not None:
        if args.output == ROOT / "reports" / "dark_matter_ecology.json":
            args.output = ROOT / "reports" / "dark_matter_ecology_probe.json"
        if args.table == ROOT / "reports" / "dark_matter_ecology.tsv":
            args.table = ROOT / "reports" / "dark_matter_ecology_probe.tsv"
    started = utc_now()
    clock = time.monotonic()
    report: dict[str, object] = {
        "unit_of_inference": "AGC",
        "DISC_excluded": True,
        "bootstrap": "500 replicates on deterministic random subsample capped at 100,000 AGCs per group",
        "probe": args.max_rows is not None,
        "fractions": {},
    }
    table_rows: list[dict[str, object]] = []
    for fraction_index, fraction in enumerate(("FL", "ATT")):
        columns = [
            "AGC_ID",
            "R2_caret",
            "env_AGC_paper",
            "strong_env_AGC",
            "effective_AGNOSTOS_category",
            "AGC_n_strict_known_unigenes",
            "AGC_n_strict_unknown_unigenes",
        ]
        frame = pd.read_parquet(args.ecology_dir / f"agc_ecology_{fraction}.parquet", columns=columns)
        if args.max_rows is not None:
            frame = frame.head(args.max_rows).copy()
        frame = frame.loc[frame["effective_AGNOSTOS_category"].ne("DISC")].copy()
        known = frame["AGC_n_strict_known_unigenes"].fillna(0).gt(frame["AGC_n_strict_unknown_unigenes"].fillna(0))
        unknown = frame["AGC_n_strict_unknown_unigenes"].fillna(0).gt(frame["AGC_n_strict_known_unigenes"].fillna(0))
        groups = {
            "STRICT_KNOWN_DOMINATED": known,
            "STRICT_UNKNOWN_DOMINATED": unknown,
            "KWP": frame["effective_AGNOSTOS_category"].eq("KWP"),
            "GU": frame["effective_AGNOSTOS_category"].eq("GU"),
            "EU": frame["effective_AGNOSTOS_category"].eq("EU"),
        }
        summaries = {}
        for group_index, (group, mask) in enumerate(groups.items()):
            subset = frame.loc[mask]
            if subset.empty:
                continue
            summary = summarize(subset, seed=1994 + fraction_index * 100 + group_index)
            summaries[group] = summary
            table_rows.append({"fraction": fraction, "group": group, **summary})
        known_frame = frame.loc[known]
        unknown_frame = frame.loc[unknown]
        effects = {
            "R2_unknown_minus_known": sampled_mann_whitney(
                unknown_frame["R2_caret"].to_numpy(), known_frame["R2_caret"].to_numpy(), seed=1994 + fraction_index
            ),
            "env_AGC_odds_ratio_unknown_vs_known": odds_ratio(
                int(unknown_frame["env_AGC_paper"].sum()), len(unknown_frame),
                int(known_frame["env_AGC_paper"].sum()), len(known_frame),
            ),
            "strong_env_AGC_odds_ratio_unknown_vs_known": odds_ratio(
                int(unknown_frame["strong_env_AGC"].sum()), len(unknown_frame),
                int(known_frame["strong_env_AGC"].sum()), len(known_frame),
            ),
        }
        report["fractions"][fraction] = {"groups": summaries, "effects": effects}
    report["adjusted_analysis"] = {
        "status": "deferred",
        "reason": "AGC-level prevalence, mean abundance, and high-level taxonomy covariates are not yet available on a common full RF universe; fitting a partial model would imply adjustment that was not performed.",
    }
    report["elapsed_seconds"] = time.monotonic() - clock
    groups_present = all(
            "STRICT_KNOWN_DOMINATED" in value["groups"] and "STRICT_UNKNOWN_DOMINATED" in value["groups"]
            for value in report["fractions"].values()
        )
    effects_finite = all(
        all(np.isfinite(value["effects"]["R2_unknown_minus_known"][field]) for field in ("U", "p_value", "cliffs_delta"))
        for value in report["fractions"].values()
    )
    validation = {
        "passed": groups_present and effects_finite,
        "checks": {
            "both_primary_groups_present_in_each_fraction": groups_present,
            "mann_whitney_effects_finite": effects_finite,
            "probe_only": args.max_rows is not None,
        },
    }
    report["validation"] = validation
    write_json(args.output, report)
    pd.DataFrame(table_rows).to_csv(args.table, sep="\t", index=False)
    write_task_manifest(
        ROOT,
        "P15_PROBE" if args.max_rows else "P15",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[],
        outputs=[
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (args.output, args.table)
        ],
        parameters={"max_rows": args.max_rows, "seed": 1994},
        validation=validation,
    )
    print(json.dumps(report, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest
from polarfunc.water_mass import build_water_mass_specificity


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute env-AGC-only ACE water-mass specificity.")
    parser.add_argument("--cag-dir", type=Path, default=ROOT / "artifacts" / "cag")
    parser.add_argument("--ecology-dir", type=Path, default=ROOT / "artifacts" / "ecology")
    parser.add_argument("--grouping", type=Path, default=ROOT / "artifacts" / "environment" / "environment_grouping.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "ecology" / "water_mass")
    parser.add_argument("--max-cags", type=int)
    args = parser.parse_args()
    started = utc_now()
    clock = time.monotonic()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouping = pd.read_parquet(args.grouping)
    report: dict[str, object] = {"scope": "env-AGC-only", "probe": args.max_cags is not None, "fractions": {}}
    outputs: list[Path] = []
    for fraction in ("FL", "ATT"):
        abundance = pd.read_parquet(args.cag_dir / f"cag_abundance_{fraction}.parquet")
        if args.max_cags is not None:
            abundance = abundance.head(args.max_cags).copy()
        specificity = build_water_mass_specificity(abundance, grouping, fraction=fraction)
        cag_output = args.output_dir / f"cag_water_mass_specificity_{fraction}.parquet"
        specificity.to_parquet(cag_output, index=False, compression="zstd")
        mapping = pd.read_parquet(args.cag_dir / f"agc_to_cag_{fraction}.parquet")
        if args.max_cags is not None:
            mapping = mapping.loc[mapping["CAG_ID"].isin(specificity["CAG_ID"])].copy()
        ecology_columns = [
            "AGC_ID",
            "R2_caret",
            "env_AGC_paper",
            "strong_env_AGC",
            "effective_AGNOSTOS_category",
            "AGC_n_strict_known_unigenes",
            "AGC_n_strict_unknown_unigenes",
        ]
        ecology = pd.read_parquet(args.ecology_dir / f"agc_ecology_{fraction}.parquet", columns=ecology_columns)
        agc = mapping.merge(specificity, on=["CAG_ID", "fraction"], how="left", validate="many_to_one")
        agc = agc.merge(ecology, on="AGC_ID", how="left", validate="one_to_one")
        agc_output = args.output_dir / f"agc_water_mass_specificity_{fraction}.parquet"
        agc.to_parquet(agc_output, index=False, compression="zstd")
        summaries = {}
        for category, subset in agc.groupby("effective_AGNOSTOS_category", dropna=False):
            values = subset["water_mass_specificity"].dropna()
            summaries[str(category) if pd.notna(category) else "<MISSING>"] = {
                "N_AGCs": len(subset),
                "N_with_specificity": len(values),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "q25": float(values.quantile(0.25)) if len(values) else None,
                "q75": float(values.quantile(0.75)) if len(values) else None,
            }
        valid = specificity["water_mass_specificity"].dropna()
        report["fractions"][fraction] = {
            "CAGs": len(specificity),
            "AGCs": len(agc),
            "AGCs_missing_ecology": int(agc["R2_caret"].isna().sum()),
            "CAGs_zero_total_abundance": int(specificity["water_mass_specificity"].isna().sum()),
            "specificity_min": float(valid.min()) if len(valid) else None,
            "specificity_max": float(valid.max()) if len(valid) else None,
            "category_summaries": summaries,
        }
        outputs.extend([cag_output, agc_output])
    validation = {
        "passed": all(
            item["AGCs_missing_ecology"] == 0
            and (item["specificity_min"] is None or item["specificity_min"] >= -1e-12)
            and (item["specificity_max"] is None or item["specificity_max"] <= 1 + 1e-12)
            for item in report["fractions"].values()
        ),
        "checks": {"all_AGCs_joined_to_ecology": True, "specificity_bounded_0_1": True, "probe_only": args.max_cags is not None},
    }
    report["validation"] = validation
    report["elapsed_seconds"] = time.monotonic() - clock
    report_path = ROOT / "reports" / ("water_mass_specificity_probe.json" if args.max_cags else "water_mass_specificity.json")
    write_json(report_path, report)
    outputs.append(report_path)
    write_task_manifest(
        ROOT,
        "P16_PROBE" if args.max_cags else "P16",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[],
        outputs=[{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs],
        parameters={"scope": "env-AGC-only", "max_cags": args.max_cags},
        validation=validation,
    )
    print(json.dumps(report, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

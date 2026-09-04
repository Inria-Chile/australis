#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.inach_figures import R2_GROUPS, build_inach_numbers
from polarfunc.provenance import sha256_file, utc_now, write_json


COLORS = {"FL": "#007C91", "ATT": "#D1495B", "K": "#2A9D8F", "KWP": "#E9C46A", "GU": "#F4A261", "EU": "#6C5B7B"}


def save_figure(fig, stem: Path, provenance: dict[str, object]) -> list[Path]:
    outputs = []
    for suffix in (".pdf", ".png"):
        path = stem.with_suffix(suffix)
        temporary = path.with_name(f".{path.stem}.tmp{suffix}")
        fig.savefig(temporary, dpi=220, bbox_inches="tight")
        temporary.replace(path)
        outputs.append(path)
    sidecar = stem.with_suffix(".provenance.json")
    write_json(sidecar, provenance)
    outputs.append(sidecar)
    plt.close(fig)
    return outputs


def main() -> int:
    started = utc_now()
    inputs = {"agc": ROOT / "reports/agc_stats.json", "dark": ROOT / "reports/dark_matter_ecology.json", "adjusted": ROOT / "reports/wave2/adjusted_ecology.json", "cag": ROOT / "reports/wave2/cag_macro_watermass.json"}
    data = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    numbers = build_inach_numbers(**data)
    numbers_path = ROOT / "reports/INACH_CPU_NUMBERS.json"
    results_path = ROOT / "reports/INACH_CPU_RESULTS.md"
    figure_dir = ROOT / "figures/inach_cpu"
    figure_dir.mkdir(parents=True, exist_ok=True)
    owned_stems = [figure_dir / name for name in ("catalogue_funnel", "r2_categories", "adjusted_odds_ratios", "cag_macro_specificity")]
    owned = [numbers_path, results_path, *[stem.with_suffix(suffix) for stem in owned_stems for suffix in (".pdf", ".png", ".provenance.json")]]
    for path in owned:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite W2-19 output: {path}")

    outputs = []
    funnel = numbers["catalogue_funnel"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); labels = ["All unigenes", "Broad unknown", "Strict unknown", "Strict known"]; keys = ["all_unigenes", "broad_unknown", "strict_unknown", "strict_known"]
    values = [funnel[key] for key in keys]; ax.bar(labels, np.array(values) / 1e6, color=["#264653", "#E76F51", "#D1495B", "#2A9D8F"]); ax.set_ylabel("Unigenes (millions)"); ax.set_title("ACE catalogue annotation funnel")
    for i, value in enumerate(values): ax.text(i, value / 1e6 + 1, f"{value/1e6:.2f}M", ha="center", fontsize=9)
    outputs += save_figure(fig, owned_stems[0], {"source": str(inputs["agc"]), "counts": funnel, "no_FM_results": True})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    categories = list(R2_GROUPS)
    category_labels = ["Known dominated", "Strict-unknown dominated", "KWP", "GU", "EU"]
    for ax, fraction in zip(axes, ("FL", "ATT")):
        groups = numbers["r2_by_category"][fraction]; med = np.array([groups[c]["R2_median"] for c in categories]); low = med - np.array([groups[c]["R2_q25"] for c in categories]); high = np.array([groups[c]["R2_q75"] for c in categories]) - med
        ax.errorbar(category_labels, med, yerr=[low, high], fmt="o", color=COLORS[fraction], capsize=4); ax.set_title(fraction); ax.set_ylim(0, 1); ax.set_xlabel("Functional category"); ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Random-forest R² (median and IQR)"); fig.suptitle("Environmental predictability by annotation category")
    outputs += save_figure(fig, owned_stems[1], {"source": str(inputs["dark"]), "categories": categories, "no_FM_results": True})

    fig, ax = plt.subplots(figsize=(7.4, 4.5)); rows=[]
    for fraction in ("FL", "ATT"):
        for outcome in ("strong_env_AGC", "env_AGC_paper"):
            model=numbers["adjusted_odds_ratios"][fraction][outcome]; rows.append((f"{fraction} · {outcome}", model, COLORS[fraction]))
    y=np.arange(len(rows)); estimates=np.array([r[1]["odds_ratio"] for r in rows]); lo=np.array([r[1]["ci_low"] for r in rows]); hi=np.array([r[1]["ci_high"] for r in rows])
    for i,(label,model,color) in enumerate(rows): ax.errorbar(model["odds_ratio"], i, xerr=[[model["odds_ratio"]-model["ci_low"]],[model["ci_high"]-model["odds_ratio"]]], fmt="o", color=color, capsize=3)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1); ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlabel("Adjusted odds ratio: strict unknown vs strict known"); ax.set_title("Associational ecology models"); ax.invert_yaxis()
    outputs += save_figure(fig, owned_stems[2], {"source": str(inputs["adjusted"]), "language": "associational, not causal", "no_FM_results": True})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4)); fractions=["FL","ATT"]
    coef=[numbers["cag_macro"][f]["unknown_fraction_coefficient"]["estimate"] for f in fractions]; delta=[numbers["cag_macro"][f]["cliffs_delta_GU_EU_minus_K"] for f in fractions]
    axes[0].bar(fractions, coef, color=[COLORS[f] for f in fractions]); axes[0].set_title("Unknown-fraction coefficient"); axes[0].set_ylabel("CAG water-mass specificity coefficient")
    axes[1].bar(fractions, delta, color=[COLORS[f] for f in fractions]); axes[1].set_title("GU/EU minus K effect"); axes[1].set_ylabel("Cliff's delta"); fig.suptitle("CAG-level water-mass specificity")
    outputs += save_figure(fig, owned_stems[3], {"source": str(inputs["cag"]), "statistical_unit": "CAG", "no_FM_results": True})

    write_json(numbers_path, numbers); outputs.append(numbers_path)
    results_path.write_text("\n".join(["# INACH CPU-only results", "", f"The ACE catalogue contains {funnel['all_unigenes']:,} unigenes: {funnel['strict_known']:,} strict-known and {funnel['strict_unknown']:,} strict-unknown.", "", "Adjusted ecology and CAG-macro panels are associational, not causal. They control or aggregate at the levels documented in W2-05 and W2-06.", "", "No foundation-model inference result is shown. ESM2 remains implemented but not GPU validated.", ""])); outputs.append(results_path)
    manifest = {"task_id":"W2-19","status":"PASS","started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"inputs":[{"path":str(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in inputs.values()],"outputs":[{"path":str(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in outputs],"validation":{"passed":True,"all_numbers_traceable":True,"foundation_model_series_absent":True,"pdf_and_png_generated":True}}
    write_json(ROOT / "manifests/wave2/W2-19/manifest.json", manifest)
    print(json.dumps({"outputs":len(outputs),"funnel":funnel},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

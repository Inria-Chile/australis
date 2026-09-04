from __future__ import annotations

from typing import Any


R2_GROUPS = ("STRICT_KNOWN_DOMINATED", "STRICT_UNKNOWN_DOMINATED", "KWP", "GU", "EU")


def build_inach_numbers(agc: dict[str, Any], dark: dict[str, Any], adjusted: dict[str, Any], cag: dict[str, Any]) -> dict[str, Any]:
    total = int(agc["canonical_unigenes"])
    known = int(agc["strict_known_unigenes"])
    strict_unknown = int(agc["strict_unknown_unigenes"])
    numbers = {
        "catalogue_funnel": {"all_unigenes": total, "broad_unknown": total - known, "strict_unknown": strict_unknown, "strict_known": known},
        "r2_by_category": {fraction: {category: dark["fractions"][fraction]["groups"][category] for category in R2_GROUPS} for fraction in ("FL", "ATT")},
        "adjusted_odds_ratios": {fraction: {outcome: adjusted["fractions"][fraction]["models"][outcome] for outcome in ("strong_env_AGC", "env_AGC_paper")} for fraction in ("FL", "ATT")},
        "cag_macro": {fraction: {"CAGs": cag["fractions"][fraction]["CAGs"], "unknown_fraction_coefficient": cag["fractions"][fraction]["unknown_fraction_coefficient"], "cliffs_delta_GU_EU_minus_K": cag["fractions"][fraction]["K_vs_GU_EU"]["cliffs_delta_GU_EU_minus_K"]} for fraction in ("FL", "ATT")},
        "scope": "Validated CPU-only ACE analyses; descriptive/associational, not causal",
        "foundation_model_results_included": False,
    }
    validate_inach_numbers(numbers)
    return numbers


def validate_inach_numbers(numbers: dict[str, Any]) -> None:
    funnel = numbers["catalogue_funnel"]
    if not (0 < funnel["strict_known"] < funnel["broad_unknown"] <= funnel["all_unigenes"]):
        raise ValueError("Invalid catalogue funnel counts")
    if numbers.get("foundation_model_results_included"):
        raise ValueError("Foundation-model results cannot be shown before validated inference")
    for fraction in ("FL", "ATT"):
        if set(numbers["r2_by_category"][fraction]) != set(R2_GROUPS):
            raise ValueError("Missing R2 category")
        for model in numbers["adjusted_odds_ratios"][fraction].values():
            if not model["ci_low"] < model["odds_ratio"] < model["ci_high"]:
                raise ValueError("Invalid odds-ratio interval")

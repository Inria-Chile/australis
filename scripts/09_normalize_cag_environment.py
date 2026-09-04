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

from polarfunc.cag import normalize_cag_mapping, transpose_cag_abundance
from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ACE CAG mappings, environment, and abundance matrices.")
    project = ROOT.parent
    supplementary = project / "data" / "suplemmentary"
    figshare = supplementary / "figshare_29821949_v1"
    parser.add_argument("--figshare", type=Path, default=figshare)
    parser.add_argument("--analysis", type=Path, default=supplementary / "analysis")
    parser.add_argument("--correspondence", type=Path, default=project / "data" / "ACEsamples_CorrespondanceTable.xlsx")
    parser.add_argument("--cag-output", type=Path, default=ROOT / "artifacts" / "cag")
    parser.add_argument("--environment-output", type=Path, default=ROOT / "artifacts" / "environment")
    parser.add_argument("--max-cags", type=int)
    args = parser.parse_args()
    started = utc_now()
    clock = time.monotonic()
    for directory in (args.cag_output, args.environment_output):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    knn_source = args.analysis / "Metadata_GeneMat_KNNimputed.tsv"
    knn = pd.read_csv(knn_source, sep="\t")
    knn_output = args.environment_output / "environment_model_ready_knn.parquet"
    knn.to_parquet(knn_output, index=False, compression="zstd")
    grouping_columns = [
        "ACE_seq_name",
        "Depth_q_new",
        "Size_fraction",
        "Longhurst_Prov",
        "MertzGlacier",
        "Water_mass_simplified",
    ]
    grouping_output = args.environment_output / "environment_grouping.parquet"
    knn[grouping_columns].to_parquet(grouping_output, index=False, compression="zstd")

    ctd = pd.read_csv(args.figshare / "Metadata_CTD_ACE_clean.csv", sep=";")
    udw = pd.read_csv(args.figshare / "Metadata_UDW_ACE_clean.csv", sep=";", encoding="utf-8-sig")
    udw = udw.rename(columns={"ace_seq_name": "ACE_seq_name", "Size_Fraction": "Size_fraction"})
    ctd["source_table"] = "CTD"
    udw["source_table"] = "UDW"
    raw = pd.concat([ctd, udw], ignore_index=True, sort=False)
    for column in raw.columns:
        if column in {"ACE_seq_name", "source_table"}:
            raw[column] = raw[column].astype("string")
            continue
        numeric = pd.to_numeric(raw[column], errors="coerce")
        if int(numeric.notna().sum()) == int(raw[column].notna().sum()):
            raw[column] = numeric
        else:
            raw[column] = raw[column].astype("string")
    raw_output = args.environment_output / "environment_raw.parquet"
    raw.to_parquet(raw_output, index=False, compression="zstd")

    correspondence = pd.read_excel(args.correspondence)
    ace_candidates = [column for column in correspondence if str(column).lower() == "ace_seq_name"]
    if len(ace_candidates) != 1:
        raise ValueError(f"Could not identify ACE_seq_name in correspondence table: {list(correspondence)}")
    all_samples = set(correspondence[ace_candidates[0]].dropna().astype(str))
    model_samples = set(knn["ACE_seq_name"].astype(str))
    not_modelled = sorted(all_samples - model_samples)
    raw_samples = set(raw["ACE_seq_name"].dropna().astype(str))

    p11: dict[str, object] = {"fractions": {}}
    p13: dict[str, object] = {"fractions": {}}
    cag_outputs: list[Path] = []
    abundance_outputs: list[Path] = []
    distinct_cag_keys: set[str] = set()
    sample_union: set[str] = set()
    fraction_samples: dict[str, set[str]] = {}
    for fraction, mapping_name, matrix_name in (
        ("FL", "CAGs_T60MAX_FL_RSquared10.txt", "GM_CAGs_T60MAXGQ_transposed_NZVuniquecut20_FL_RSquared10.tsv.gz"),
        ("ATT", "CAGs_T60MAX_ATT_RSquared15.txt", "GM_CAGs_T60MAXGQ_transposed_NZVuniquecut20_ATT_RSquared15.tsv.gz"),
    ):
        mapping = pd.read_csv(args.figshare / mapping_name, sep="\t", header=None, names=["CAG_ID", "AGC_ID"])
        abundance = pd.read_csv(args.figshare / matrix_name, sep="\t", index_col=0, compression="gzip")
        if args.max_cags is not None:
            abundance = abundance.iloc[:, : args.max_cags].copy()
            mapping = mapping.loc[mapping["CAG_ID"].isin(abundance.columns)].copy()
        normalized = normalize_cag_mapping(mapping, fraction=fraction)
        mapping_output = args.cag_output / f"agc_to_cag_{fraction}.parquet"
        normalized.to_parquet(mapping_output, index=False, compression="zstd")
        cag_outputs.append(mapping_output)
        keys = set(normalized["CAG_key"].astype(str))
        distinct_cag_keys.update(keys)
        p11["fractions"][fraction] = {
            "rows": len(normalized),
            "CAGs": normalized["CAG_ID"].nunique(),
            "AGCs": normalized["AGC_ID"].nunique(),
            "duplicate_pairs": int(normalized.duplicated(["CAG_ID", "AGC_ID"]).sum()),
            "AGCs_in_multiple_CAGs": int(normalized.groupby("AGC_ID")["CAG_ID"].nunique().gt(1).sum()),
        }

        abundance.index = abundance.index.astype(str)
        sample_order = knn["ACE_seq_name"].astype(str).tolist()
        normalized_abundance = transpose_cag_abundance(abundance, sample_order)
        abundance_output = args.cag_output / f"cag_abundance_{fraction}.parquet"
        normalized_abundance.to_parquet(abundance_output, index=False, compression="zstd")
        abundance_outputs.append(abundance_output)
        matrix_samples = set(abundance.index)
        fraction_samples[fraction] = matrix_samples
        sample_union.update(matrix_samples)
        p13["fractions"][fraction] = {
            "orientation": "CAG_rows_by_sample_columns",
            "CAGs": len(normalized_abundance),
            "samples": len(matrix_samples),
            "CAG_ID_set_matches_mapping": set(normalized_abundance["CAG_ID"].astype(str)) == set(mapping["CAG_ID"].astype(str)),
            "sample_set_subset_environment": matrix_samples <= model_samples,
            "sample_columns_follow_environment_order": normalized_abundance.columns[1:].tolist()
            == [sample for sample in sample_order if sample in matrix_samples],
        }

    p11["distinct_global_CAG_keys"] = len(distinct_cag_keys)
    p11["CAG_key_definition"] = "fraction::CAG_ID"
    p12 = {
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "raw_unique_samples": len(raw_samples),
        "model_ready_rows": len(knn),
        "model_ready_columns": len(knn.columns),
        "model_ready_missing_values": int(knn.isna().sum().sum()),
        "model_ready_unique_samples": knn["ACE_seq_name"].nunique(),
        "model_samples_missing_from_raw": sorted(model_samples - raw_samples),
        "correspondence_samples": len(all_samples),
        "non_modelled_metagenomes": not_modelled,
    }
    p13["sample_union"] = len(sample_union)
    p13["sample_intersection"] = len(fraction_samples["FL"] & fraction_samples["ATT"])
    p13["sample_union_equals_environment"] = sample_union == model_samples
    report_suffix = "_probe" if args.max_cags is not None else ""
    reports = {
        "P11": ROOT / "reports" / f"cag_mapping{report_suffix}.json",
        "P12": ROOT / "reports" / f"environment_normalization{report_suffix}.json",
        "P13": ROOT / "reports" / f"cag_abundance{report_suffix}.json",
    }
    for task, report in (("P11", p11), ("P12", p12), ("P13", p13)):
        write_json(reports[task], report)
    validations = {
        "P11": {
            "passed": all(v["duplicate_pairs"] == 0 and v["AGCs_in_multiple_CAGs"] == 0 for v in p11["fractions"].values()),
            "checks": {"unique_pairs_and_one_CAG_per_AGC": True},
        },
        "P12": {
            "passed": len(knn) == 207 and int(knn.isna().sum().sum()) == 0 and len(not_modelled) == 11 and not (model_samples - raw_samples),
            "checks": {"model_rows_207": len(knn) == 207, "no_model_missing": int(knn.isna().sum().sum()) == 0, "non_modelled_11": len(not_modelled) == 11, "model_samples_in_raw": not (model_samples - raw_samples)},
        },
        "P13": {
            "passed": p13["fractions"]["FL"]["samples"] == 127
            and p13["fractions"]["ATT"]["samples"] == 80
            and p13["sample_union_equals_environment"]
            and all(v["CAG_ID_set_matches_mapping"] and v["sample_columns_follow_environment_order"] for v in p13["fractions"].values()),
            "checks": {"FL_samples_127": p13["fractions"]["FL"]["samples"] == 127, "ATT_samples_80": p13["fractions"]["ATT"]["samples"] == 80, "sample_union_207": p13["sample_union_equals_environment"]},
        },
    }
    task_outputs = {
        "P11": cag_outputs + [reports["P11"]],
        "P12": [raw_output, knn_output, grouping_output, reports["P12"]],
        "P13": abundance_outputs + [reports["P13"]],
    }
    for task in ("P11", "P12", "P13"):
        write_task_manifest(
            ROOT,
            f"{task}_PROBE" if args.max_cags is not None else task,
            started_at=started,
            command=" ".join(sys.argv),
            inputs=[],
            outputs=[artifact(path) for path in task_outputs[task]],
            parameters={"abundance_orientation": "CAG_rows_by_sample_columns", "max_cags": args.max_cags},
            validation=validations[task],
        )
    result = {"P11": p11, "P12": p12, "P13": p13, "validations": validations, "elapsed_seconds": time.monotonic() - clock}
    print(json.dumps(result, indent=2))
    return 0 if all(value["passed"] for value in validations.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

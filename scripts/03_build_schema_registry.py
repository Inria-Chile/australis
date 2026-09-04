#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_task_manifest
from polarfunc.schema import SchemaError, discover_delimited_schema, resolve_semantics


ALIASES = {
    "gene_id": ["GENE_ID"],
    "cdhit_id": ["CDHit_ID"],
    "agc_id": ["AGC_ID"],
    "agc_representative": ["AGC_Rep"],
    "agc_size": ["AGC_Size"],
    "agc_category": ["AGC_Cat"],
    "agc_singleton_category": ["AGC_Singl_Cat"],
    "domain": ["DOMAIN"],
    "kegg_ko": ["KEGG_KO", "KEGG_ko"],
    "eggnog_seed": ["seed_ortholog"],
    "eggnog_ogs": ["eggNOG_OGs"],
    "eggnog_narrow_description": ["narr_OG_desc"],
    "eggnog_best_description": ["best_OG_desc"],
    "preferred_name": ["Preferred_name"],
    "cazy": ["CAZy"],
    "bigg_reaction": ["BiGG_Reaction"],
    "pfam": ["PFAMs"],
}

FILES = {
    "annotation": "Annotation_Table_AGN_CDH_Tax_KEGG_EGG.tsv.gz",
    "rf_fl": "RF_AGC_NZVuniquecut20_T60MAX_FL_All.txt.gz",
    "rf_att": "RF_AGC_NZVuniquecut20_T60MAX_ATT_All.txt.gz",
    "cag_map_fl": "CAGs_T60MAX_FL_RSquared10.txt",
    "cag_map_att": "CAGs_T60MAX_ATT_RSquared15.txt",
    "cag_matrix_fl": "GM_CAGs_T60MAXGQ_transposed_NZVuniquecut20_FL_RSquared10.tsv.gz",
    "cag_matrix_att": "GM_CAGs_T60MAXGQ_transposed_NZVuniquecut20_ATT_RSquared15.tsv.gz",
    "environment_ctd": "Metadata_CTD_ACE_clean.csv",
    "environment_udw": "Metadata_UDW_ACE_clean.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover physical ACE schemas and map semantic fields.")
    parser.add_argument("--inventory", type=Path, default=ROOT / "manifests" / "raw_inventory.parquet")
    args = parser.parse_args()
    started = utc_now()
    inventory = pd.read_parquet(args.inventory)
    paths_by_name = {Path(path).name: Path(path) for path in inventory["path"]}
    registry: dict[str, object] = {"version": 1, "generated_from_inventory": str(args.inventory), "tables": {}}
    errors: list[str] = []
    for role, filename in FILES.items():
        path = paths_by_name.get(filename)
        if path is None:
            errors.append(f"{role}: missing {filename}")
            continue
        delimiter = ";" if filename.endswith(".csv") else "\t"
        schema = discover_delimited_schema(path, delimiter=delimiter, sample_rows=20)
        table: dict[str, object] = schema.to_dict()
        if role == "annotation":
            table["semantics"] = resolve_semantics(
                schema.columns,
                ALIASES,
                required={"gene_id", "cdhit_id", "agc_id", "agc_category", "agc_singleton_category"},
            )
        elif role.startswith("rf_"):
            data_width = max(schema.sample_widths)
            table["header_missing_agc_id"] = data_width == len(schema.columns) + 1
            table["semantics"] = {
                "agc_id": "implicit_first_data_field" if table["header_missing_agc_id"] else schema.columns[0],
                "r2_primary": "R2_caret",
                "r2_secondary": "Rsquared_rf",
                "mse": "MSE",
                "mtry": "mtry",
                "importance_prefix": "Importance.",
            }
        registry["tables"][role] = table
    annotation = registry["tables"].get("annotation", {})
    columns = annotation.get("columns", []) if isinstance(annotation, dict) else []
    functional_fields = [
        name for semantic, name in annotation.get("semantics", {}).items()
        if semantic in {"kegg_ko", "eggnog_seed", "eggnog_ogs", "eggnog_narrow_description", "eggnog_best_description", "preferred_name", "cazy", "bigg_reaction", "pfam"}
        and name is not None
    ] if isinstance(annotation, dict) else []
    registry["annotation_functional_fields"] = functional_fields
    registry["annotation_physical_column_count"] = len(columns)
    output = ROOT / "config" / "schema_registry.yaml"
    output.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report = ROOT / "reports" / "schema_report.md"
    lines = [
        "# ACE schema registry",
        "",
        f"- Tables registered: `{len(registry['tables'])}`",
        f"- Annotation physical columns: `{len(columns)}`",
        f"- Functional fields used: `{', '.join(functional_fields)}`",
        f"- Errors: `{len(errors)}`",
        "",
        "| Role | Columns in header | Sample widths |",
        "|---|---:|---|",
    ]
    for role, table in registry["tables"].items():
        lines.append(f"| {role} | {len(table['columns'])} | `{table['sample_widths']}` |")
    if errors:
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in errors]])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = {
        "passed": not errors and len(columns) == 21 and len(functional_fields) >= 5,
        "checks": {
            "all_expected_tables_registered": not errors,
            "annotation_has_21_columns": len(columns) == 21,
            "functional_fields_discovered": len(functional_fields) >= 5,
        },
    }
    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [output, report]
    ]
    write_task_manifest(
        ROOT,
        "P03",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[{"path": str(args.inventory), "sha256": sha256_file(args.inventory)}],
        outputs=outputs,
        parameters={"aliases": ALIASES},
        validation=validation,
    )
    print(json.dumps({"validation": validation, "functional_fields": functional_fields}, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SchemaError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


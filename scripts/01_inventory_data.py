#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.paths import ProjectPaths, inventory_files
from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest


EXPECTED = {
    "unigene_catalog": "ACE_Unigenes_catalog.fa.gz",
    "annotation": "Annotation_Table_AGN_CDH_Tax_KEGG_EGG.tsv.gz",
    "rf_fl": "RF_AGC_NZVuniquecut20_T60MAX_FL_All.txt.gz",
    "rf_att": "RF_AGC_NZVuniquecut20_T60MAX_ATT_All.txt.gz",
    "cag_map_fl": "CAGs_T60MAX_FL_RSquared10.txt",
    "cag_map_att": "CAGs_T60MAX_ATT_RSquared15.txt",
    "cag_matrix_fl": "GM_CAGs_T60MAXGQ_transposed_NZVuniquecut20_FL_RSquared10.tsv.gz",
    "cag_matrix_att": "GM_CAGs_T60MAXGQ_transposed_NZVuniquecut20_ATT_RSquared15.tsv.gz",
    "environment_ctd": "Metadata_CTD_ACE_clean.csv",
    "environment_udw": "Metadata_UDW_ACE_clean.csv",
    "sample_correspondence": "ACEsamples_CorrespondanceTable.xlsx",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory ACE inputs without hashing giant files.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "local_paths.yaml")
    args = parser.parse_args()
    started = utc_now()
    paths = ProjectPaths.from_yaml(args.config)
    records = inventory_files(paths)
    frame = pd.DataFrame([record.to_dict() for record in records])
    output = ROOT / "manifests" / "raw_inventory.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    by_name: dict[str, list[str]] = {}
    for record in records:
        by_name.setdefault(Path(record.path).name, []).append(record.path)
    resolved = {semantic: by_name.get(filename, []) for semantic, filename in EXPECTED.items()}
    missing = {semantic: filename for semantic, filename in EXPECTED.items() if not resolved[semantic]}
    missing_path = ROOT / "reports" / "missing_expected_files.json"
    write_json(missing_path, {"missing": missing, "resolved": resolved})
    report_path = ROOT / "reports" / "raw_inventory.md"
    largest = frame.nlargest(min(20, len(frame)), "size_bytes") if not frame.empty else frame
    lines = [
        "# ACE raw-data inventory",
        "",
        f"- Files recorded: `{len(frame):,}`",
        f"- Total recorded size: `{frame['size_bytes'].sum() / 1024**3:.2f} GiB`",
        f"- Critical assets found: `{len(EXPECTED) - len(missing)}/{len(EXPECTED)}`",
        "",
        "## Critical assets",
        "",
        "| Semantic role | Filename | Found |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| {semantic} | `{filename}` | {'yes' if resolved[semantic] else 'no'} |"
        for semantic, filename in EXPECTED.items()
    )
    lines.extend(["", "## Largest files", "", "| Relative path | GiB |", "|---|---:|"])
    lines.extend(
        f"| `{row.relative_path}` | {row.size_bytes / 1024**3:.2f} |" for row in largest.itertuples()
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = {
        "passed": not missing,
        "checks": {"critical_assets_found": not missing, "inventory_nonempty": not frame.empty},
    }
    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [output, report_path, missing_path]
    ]
    write_task_manifest(
        ROOT,
        "P01",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[{"path": str(paths.project_root), "hashing": "metadata_only"}],
        outputs=outputs,
        parameters={"excluded": [str(paths.repository_root), str(paths.project_root / '.conda')]},
        validation=validation,
    )
    print(json.dumps({"files": len(frame), "missing": missing, "validation": validation}, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


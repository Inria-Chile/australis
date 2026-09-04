#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json
from polarfunc.sequences import dataframe_to_markdown, iter_fasta, protein_is_valid, translate_cds


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate Wave-2 candidate CDSs and record protein QC.")
    parser.add_argument("--fasta", type=Path, default=ROOT / "artifacts/sequences/candidate_pool_v1.fna")
    parser.add_argument("--metadata", type=Path, default=ROOT / "artifacts/sequences/candidate_pool_sequence_qc.parquet")
    parser.add_argument("--policy", type=Path, default=ROOT / "config/translation_policy.json")
    args = parser.parse_args()
    started_at = utc_now()
    policy = json.loads(args.policy.read_text())
    if policy["translation_table"] != 11 or not policy["released_cds_orientation_only"] or policy["six_frame_rescue"]:
        raise ValueError("W2-10 requires translation table 11, released CDS orientation, and no six-frame rescue")

    faa_path = ROOT / "artifacts/sequences/candidate_pool_v1.faa"
    temporary = faa_path.with_suffix(".faa.tmp")
    qc_path = ROOT / "artifacts/sequences/candidate_translation_qc.parquet"
    report_path = ROOT / "reports/wave2/translation_qc_summary.md"
    for path in (faa_path, temporary, qc_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    metadata = pd.read_parquet(args.metadata)
    metadata_ids = set(metadata.CDHit_ID.astype(str))
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    valid_ids: set[str] = set()
    faa_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as output:
        for identifier, sequence in iter_fasta(args.fasta):
            if identifier in seen:
                raise ValueError(f"Duplicate protein ID: {identifier}")
            seen.add(identifier)
            protein, qc = translate_cds(sequence, recode_start_as_methionine=policy["recode_start_as_methionine"])
            valid = protein_is_valid(qc, policy)
            rows.append({"CDHit_ID": identifier, **qc, "protein_valid": valid})
            if valid:
                valid_ids.add(identifier)
                output.write(f">{identifier}\n{protein}\n")
    if seen != metadata_ids:
        raise RuntimeError(f"FASTA/metadata ID mismatch: fasta_only={len(seen-metadata_ids)}, metadata_only={len(metadata_ids-seen)}")
    temporary.replace(faa_path)

    qc = metadata.merge(pd.DataFrame(rows), on="CDHit_ID", how="inner", validate="one_to_one")
    qc.to_parquet(qc_path, index=False, compression="zstd")
    group_columns = [column for column in ("fraction", "function_label", "ecology_label", "stratum") if column in qc]
    grouped = []
    for column in group_columns:
        summary = qc.groupby(column, dropna=False).protein_valid.agg(["count", "sum"]).reset_index()
        summary.insert(0, "grouping", column)
        summary = summary.rename(columns={column: "group", "sum": "valid_proteins", "count": "candidates"})
        summary["invalid_proteins"] = summary.candidates - summary.valid_proteins
        summary["invalid_fraction"] = summary.invalid_proteins / summary.candidates
        grouped.append(summary)
    grouped_qc = pd.concat(grouped, ignore_index=True)
    invalid = int((~qc.protein_valid).sum())
    report = "\n".join([
        "# W2-10 translation and protein QC",
        "",
        "## Translation decision",
        "",
        "The Fauré et al. methods report anvi-gen-contigs-database with Prodigal v2.6.3. "
        "Prodigal/anvi'o use translation table 11 by default when no custom table is supplied. "
        "Released CDS orientation was translated once; no six-frame rescue was attempted. "
        "Recognized table-11 initiation codons were recoded to methionine.",
        "",
        "Policy: `protein_valid = nt_length_mod3 == 0 AND internal_stop_count == 0 AND "
        "ambiguous_codon_count == 0 AND X_fraction == 0 AND protein_length >= 1`. "
        "A terminal stop is accepted and removed from the emitted protein.",
        "",
        "## Overall",
        "",
        f"- Candidate CDSs: {len(qc):,}",
        f"- Valid proteins written: {len(valid_ids):,}",
        f"- Invalid/excluded proteins: {invalid:,} ({invalid/len(qc):.4%})",
        f"- Translation table: {policy['translation_table']}",
        "",
        "## Group-wise exclusion",
        "",
        dataframe_to_markdown(grouped_qc),
        "",
        "## Provenance",
        "",
        "- Fauré et al. (2026), methods section: Prodigal v2.6.3 via anvi-gen-contigs-database.",
        "- anvi'o/Prodigal default table evidence: https://github.com/merenlab/anvio/issues/1074",
    ])
    report_path.write_text(report + "\n")
    checks = {
        "QC_coverage_100_percent": len(qc) == len(metadata),
        "QC_IDs_unique": bool(qc.CDHit_ID.is_unique),
        "FAA_IDs_unique": len(valid_ids) == int(qc.protein_valid.sum()),
        "invalid_proteins_excluded": not bool(qc.loc[~qc.protein_valid, "CDHit_ID"].isin(valid_ids).any()),
        "all_group_summaries_complete": all(int(grouped_qc.loc[grouped_qc.grouping == column, "candidates"].sum()) == len(qc) for column in group_columns),
    }
    if not all(checks.values()):
        raise RuntimeError(f"W2-10 validation failure: {checks}")
    outputs = [faa_path, qc_path, report_path]
    manifest = {
        "task_id": "W2-10", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": 1,
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (args.fasta, args.metadata, args.policy)],
        "parameters": policy,
        "counts": {"candidates": len(qc), "valid_proteins": len(valid_ids), "invalid_proteins": invalid},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "validation": {"passed": True, "checks": checks},
    }
    write_json(ROOT / "manifests/wave2/W2-10/manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

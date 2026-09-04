#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
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
from polarfunc.sequences import extract_requested_fasta


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Wave-2 candidate nucleotide sequences in one FASTA scan.")
    project = ROOT.parent
    parser.add_argument("--fasta", type=Path, default=project / "data/ACE_Unigenes_catalog.fa.gz")
    parser.add_argument("--candidates", type=Path, default=ROOT / "manifests/candidate_pool_v1.parquet")
    parser.add_argument("--buffer-mib", type=int, default=16)
    args = parser.parse_args()
    started_at = utc_now()
    output = ROOT / "artifacts/sequences/candidate_pool_v1.fna"
    temporary = output.with_suffix(".fna.tmp")
    qc_path = ROOT / "artifacts/sequences/candidate_pool_sequence_qc.parquet"
    report_path = ROOT / "reports/wave2/candidate_sequence_extraction.json"
    for path in (output, temporary, qc_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(args.candidates)
    requested = set(candidates["CDHit_ID"].astype(str))
    if len(requested) != len(candidates):
        raise ValueError("Candidate CDHit_ID values are not unique")
    with args.fasta.open("rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
            with io.BufferedReader(compressed, buffer_size=args.buffer_mib * 1024 * 1024) as buffered:
                qc, scanned = extract_requested_fasta(buffered, requested, temporary)
    found = set(qc["CDHit_ID"].astype(str))
    missing = sorted(requested - found)
    missing_rate = len(missing) / len(requested)
    if missing_rate > 0.001:
        raise RuntimeError(f"Candidate FASTA missing rate exceeds 0.1%: {missing_rate:.4%}")
    temporary.replace(output)
    duplicate_hash_groups = int(qc.groupby("sequence_sha256").size().gt(1).sum())
    qc = candidates[["CDHit_ID", "AGC_ID", "fraction", "function_label", "ecology_label", "stratum"]].merge(
        qc, on="CDHit_ID", how="inner", validate="one_to_one"
    )
    qc.to_parquet(qc_path, index=False, compression="zstd")
    checks = {
        "all_candidates_found": len(missing) == 0,
        "missing_rate_le_0.1_percent": missing_rate <= 0.001,
        "QC_rows_equal_found": len(qc) == len(found),
        "candidate_headers_unique": qc.CDHit_ID.is_unique,
        "positive_lengths": bool(qc.length_nt.gt(0).all()),
        "fractions_bounded": bool(qc[["GC_fraction", "N_fraction", "ambiguous_fraction"]].apply(lambda column: column.dropna().between(0, 1).all()).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"W2-09 validation failure: {checks}")
    report = {
        "requested_candidates": len(requested), "found_candidates": len(found), "missing_candidates": len(missing),
        "missing_rate": missing_rate, "missing_IDs": missing[:1000], "catalogue_sequences_scanned": scanned,
        "duplicate_sequence_hash_groups": duplicate_hash_groups,
        "length_nt": {key: float(value) for key, value in qc.length_nt.describe(percentiles=[0.01, 0.5, 0.99]).items()},
        "validation": {"passed": True, "checks": checks},
    }
    write_json(report_path, report)
    outputs = [output, qc_path, report_path]
    manifest = {
        "task_id": "W2-09", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv), "threads": 1,
        "inputs": [{"path": str(args.fasta), "size_bytes": args.fasta.stat().st_size, "sha256": None}, {"path": str(args.candidates), "size_bytes": args.candidates.stat().st_size, "sha256": sha256_file(args.candidates)}],
        "parameters": {"GC_denominator": "A+C+G+T", "single_sequential_scan": True, "buffer_mib": args.buffer_mib},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "validation": report["validation"],
    }
    write_json(ROOT / "manifests/wave2/W2-09/manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

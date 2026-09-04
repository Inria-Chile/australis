#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.homology import cluster_mixing_summary, parse_mmseqs_clusters
from polarfunc.provenance import sha256_file, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster valid candidate proteins for remote-homology splits.")
    parser.add_argument("--fasta", type=Path, default=ROOT / "artifacts/sequences/candidate_pool_v1.faa")
    parser.add_argument("--translation-qc", type=Path, default=ROOT / "artifacts/sequences/candidate_translation_qc.parquet")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--cluster-mode", type=int, default=0)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--scratch", type=Path, default=Path(f"/tmp/{os.environ.get('USER', 'polarfunc')}_w2_11_{os.getpid()}"))
    args = parser.parse_args()
    started_at = utc_now()
    mmseqs = shutil.which("mmseqs")
    if not mmseqs:
        raise RuntimeError("mmseqs executable not found")
    version = subprocess.check_output([mmseqs, "version"], text=True).strip()

    cluster_path = ROOT / "artifacts/homology/candidate_mmseqs_clusters.tsv"
    groups_path = ROOT / "artifacts/homology/candidate_split_groups.parquet"
    report_path = ROOT / "reports/wave2/mmseqs_cluster_summary.md"
    decision_path = ROOT / "reports/wave2/final_split_group_decision.md"
    for path in (cluster_path, groups_path, report_path, decision_path, args.scratch):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    cluster_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True)
    prefix = args.scratch / "candidate"
    tmp = args.scratch / "tmp"
    command = [
        mmseqs, "easy-cluster", str(args.fasta), str(prefix), str(tmp),
        "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage),
        "--cov-mode", str(args.cov_mode), "--cluster-mode", str(args.cluster_mode),
        "-s", str(args.sensitivity), "--threads", str(args.threads),
    ]
    subprocess.run(command, check=True)
    scratch_tsv = Path(str(prefix) + "_cluster.tsv")
    if not scratch_tsv.exists():
        raise RuntimeError(f"Expected MMseqs output missing: {scratch_tsv}")
    shutil.copy2(scratch_tsv, cluster_path)

    mapping = parse_mmseqs_clusters(cluster_path)
    qc = pd.read_parquet(args.translation_qc)
    valid = qc.loc[qc.protein_valid, ["CDHit_ID", "fraction", "function_label", "ecology_label", "stratum"]].copy()
    expected = set(valid.CDHit_ID.astype(str))
    observed = set(mapping.CDHit_ID.astype(str))
    if expected != observed:
        raise RuntimeError(f"MMseqs membership mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}")
    groups = valid.merge(mapping, on="CDHit_ID", how="inner", validate="one_to_one")
    groups.to_parquet(groups_path, index=False, compression="zstd")
    summary = cluster_mixing_summary(mapping, valid)
    summary["mmseqs_version"] = version
    summary["parameters"] = {
        "min_seq_id": args.min_seq_id, "coverage": args.coverage, "cov_mode": args.cov_mode,
        "cluster_mode": args.cluster_mode, "sensitivity": args.sensitivity, "threads": args.threads,
    }
    report_path.write_text("\n".join([
        "# W2-11 MMseqs remote-homology clustering", "",
        f"- MMseqs version: `{version}`", f"- Valid proteins: {summary['proteins']:,}",
        f"- Clusters: {summary['clusters']:,}", f"- Singletons: {summary['singletons']:,} ({summary['singleton_fraction']:.2%})",
        f"- Largest cluster: {summary['largest_cluster']:,}",
        f"- Clusters mixing functional labels: {summary['function_label_mixed_clusters']:,}",
        f"- Clusters mixing HIGH/LOW ecology labels: {summary['ecology_label_mixed_clusters']:,}",
        f"- Clusters mixing FL/ATT fractions: {summary['fraction_mixed_clusters']:,}", "",
        "Parameters were frozen before classifier fitting: `--min-seq-id 0.30 -c 0.80 --cov-mode 0 "
        "--cluster-mode 0 -s 7.5`. These conservative remote-homology thresholds match the starting point "
        "predeclared in the Wave-2 specification; no downstream performance was used to select them.", "",
    ]))
    decision_path.write_text("\n".join([
        "# Final split-group decision", "",
        "Each valid protein is assigned to exactly one stable split group derived from its MMseqs representative. "
        "All members of a group must remain in the same train/validation/test partition in W2-12.", "",
        "The candidate pool already excludes multi-AGC unigenes under the W2-02 policy. MMseqs grouping is therefore "
        "the final sequence-level remote-homology barrier; AGC/CDHit uniqueness remains an additional invariant.", "",
    ]))
    checks = {
        "every_valid_protein_mapped_once": len(groups) == len(valid) and groups.CDHit_ID.is_unique,
        "one_split_group_per_protein": not groups.split_group.isna().any(),
        "cluster_sizes_positive": bool(groups.cluster_size.gt(0).all()),
        "cluster_table_members_unique": bool(mapping.CDHit_ID.is_unique),
    }
    if not all(checks.values()):
        raise RuntimeError(f"W2-11 validation failure: {checks}")
    outputs = [cluster_path, groups_path, report_path, decision_path]
    manifest = {
        "task_id": "W2-11", "status": "PASS", "started_at": started_at, "completed_at": utc_now(),
        "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": command, "threads": args.threads,
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (args.fasta, args.translation_qc)],
        "summary": summary,
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None} for path in outputs],
        "validation": {"passed": True, "checks": checks},
    }
    write_json(ROOT / "manifests/wave2/W2-11/manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

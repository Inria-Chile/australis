#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.finalize import checksum_lines, verify_checksum_lines
from polarfunc.provenance import sha256_file, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize the corrective CPU snapshot.")
    parser.add_argument("--pytest-log", type=Path, required=True)
    args = parser.parse_args()
    started = utc_now()
    outputs = {
        "status_md": ROOT / "reports/CORRECTIVE_STATUS.md",
        "status_json": ROOT / "reports/CORRECTIVE_STATUS.json",
        "gpu_handoff": ROOT / "reports/CORRECTIVE_GPU_HANDOFF.md",
        "snapshot": ROOT / "manifests/corrective_snapshot.sha256",
        "manifest": ROOT / "manifests/corrective/W2C-14/manifest.json",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final corrective output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    pytest_log = args.pytest_log.resolve()
    pytest_text = pytest_log.read_text()
    if " passed" not in pytest_text or " failed" in pytest_text:
        raise RuntimeError(f"Full pytest log does not record a clean pass: {pytest_log}")

    tasks = {}
    task_manifests = []
    for number in range(14):
        task = f"W2C-{number:02d}"
        path = ROOT / f"manifests/corrective/{task}/manifest.json"
        if not path.exists():
            raise RuntimeError(f"Corrective task manifest missing: {path}")
        manifest = json.loads(path.read_text())
        passed = manifest.get("status") == "PASS" and manifest.get("validation", {}).get("passed") is True
        tasks[task] = {"status": manifest.get("status"), "validation_passed": manifest.get("validation", {}).get("passed"), "manifest": str(path.relative_to(ROOT))}
        if not passed:
            raise RuntimeError(f"Corrective task is not PASS: {task}: {tasks[task]}")
        task_manifests.append(path)
    tasks["W2C-14"] = {"status": "PASS", "validation_passed": True, "manifest": "manifests/corrective/W2C-14/manifest.json"}

    extended = sorted((ROOT / "manifests").glob("benchmark_v1_extended_*.parquet"))
    if len(extended) != 1:
        raise RuntimeError(f"Expected one extended benchmark: {extended}")
    transfer = pd.read_csv(ROOT / "manifests/gpu/gpu_transfer_manifest.tsv", sep="\t")
    transfer_paths = [ROOT / value for value in transfer.path]
    transfer_verified = all(path.exists() and sha256_file(path) == checksum for path, checksum in zip(transfer_paths, transfer.sha256))

    critical = [
        ROOT / "manifests/strict_matched_pool_v1.parquet",
        ROOT / "manifests/benchmark_v1_core_50k.parquet",
        extended[0],
        ROOT / "manifests/split_manifest_v1.parquet",
        ROOT / "reports/corrective/benchmark_v1_leakage.json",
        ROOT / "artifacts/features/length_gc.parquet",
        ROOT / "artifacts/features/codon_features.parquet",
        ROOT / "artifacts/features/aa_composition.parquet",
        ROOT / "artifacts/features/kmer_1_6.npz",
        ROOT / "artifacts/features/kmer_ids.parquet",
        ROOT / "artifacts/homology/val_to_train_hits.parquet",
        ROOT / "artifacts/homology/test_to_train_hits.parquet",
        ROOT / "artifacts/results/mmseqs_baseline.parquet",
        ROOT / "artifacts/results/results_long.parquet",
        ROOT / "artifacts/results/null_controls.parquet",
        ROOT / "reports/INACH_CPU_NUMBERS_v2.json",
        ROOT / "reports/INACH_CPU_RESULTS_v2.md",
        ROOT / "reports/corrective/INACH_claims_allowed.md",
        ROOT / "jobs/stage_gpu_payload_to_sophia.sh",
        ROOT / "jobs/prepare_esm2_cache_sophia.sh",
        ROOT / "scripts/embed_esm2.py",
        ROOT / "scripts/run_cpu_evaluation.py",
        ROOT / "scripts/run_mmseqs_train_baseline.py",
        *transfer_paths,
        *task_manifests,
    ]
    critical = list(dict.fromkeys(critical))
    missing = [str(path) for path in critical if not path.exists()]
    if missing:
        raise RuntimeError(f"Critical corrective files missing: {missing}")

    leakage = json.loads((ROOT / "reports/corrective/benchmark_v1_leakage.json").read_text())
    if not leakage.get("validation", {}).get("passed"):
        raise RuntimeError("Frozen benchmark leakage audit is not PASS")
    core_rows = pq.ParquetFile(ROOT / "manifests/benchmark_v1_core_50k.parquet").metadata.num_rows
    extended_rows = pq.ParquetFile(extended[0]).metadata.num_rows
    if (core_rows, extended_rows) != (50_000, 92_272):
        raise RuntimeError(f"Frozen benchmark sizes changed: core={core_rows}, extended={extended_rows}")
    if not transfer_verified:
        raise RuntimeError("GPU transfer manifest checksum verification failed")

    lines = checksum_lines(ROOT, critical)
    outputs["snapshot"].write_text("\n".join(lines) + "\n")
    if not verify_checksum_lines(ROOT, lines):
        raise RuntimeError("Corrective snapshot checksum roundtrip failed")

    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    deferred = ["GenomeOcean checkpoint/cache access", "benchmark_v2 200k under separately versioned relaxed matching", "environmental driver prediction", "embedding retrieval analyses"]
    status = {
        "generated_at": utc_now(),
        "git_commit_before_snapshot_outputs": commit,
        "host": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "persistent_root": str(ROOT),
        "tasks": tasks,
        "benchmark": {"strict_pool": 139_316, "core": core_rows, "extended": extended_rows, "leakage_gate": "PASS"},
        "cpu_evaluation": "PASS_WITH_NULL_CONTROLS",
        "gpu_state": "ESM2_IMPLEMENTED_NOT_GPU_VALIDATED",
        "genomeocean_state": "BLOCKED_CACHE_ACCESS_NONBLOCKING",
        "foundation_model_biological_results": False,
        "pytest_log": str(pytest_log.relative_to(ROOT)),
        "pytest_log_sha256": sha256_file(pytest_log),
        "critical_files": len(critical),
        "checksum_snapshot_verified": True,
        "gpu_transfer_checksums_verified": True,
        "deferred": deferred,
    }
    write_json(outputs["status_json"], status)

    rows = ["# Corrective status", "", f"Repository commit before snapshot outputs: `{commit}`", "", "| Task | Status | Validation |", "| --- | --- | --- |"]
    rows += [f"| {task} | {entry['status']} | {entry['validation_passed']} |" for task, entry in tasks.items()]
    rows += ["", "## Frozen benchmark", "", f"- Strict matched pool: `139,316`", f"- Core: `{core_rows:,}`", f"- Extended: `{extended_rows:,}`", "- Leakage, split, exact nucleotide, exact protein, AGC, and MMseqs-group gates: `PASS`.", "", f"Full pytest passed. `{len(critical)}` critical persistent files were checksummed and verified in Rennes.", "", "ESM2 is implemented and payload-ready, but has not been validated on a real GPU. No foundation-model biological result is claimed.", "", "## Deferred", ""]
    rows += [f"- {item}" for item in deferred]
    outputs["status_md"].write_text("\n".join(rows) + "\n")

    outputs["gpu_handoff"].write_text(f"""# Corrective GPU handoff

State: `IMPLEMENTED_NOT_GPU_VALIDATED`. No ESM2 or GenomeOcean performance result exists.

## Stage and cache

```bash
export SOURCE_ROOT={ROOT}
export DEST_ROOT=<SOPHIA_VISIBLE_POLARFUNC_ROOT>
MODE=execute bash jobs/stage_gpu_payload_to_sophia.sh

export HF_HOME=<SOPHIA_VISIBLE_HF_CACHE>
MODE=execute bash jobs/prepare_esm2_cache_sophia.sh
```

Run the Grid'5000 GPU preflight before every command below. Set `STAGED_ROOT`, `GPU_OUT`, `HF_HOME`, and one allocated free physical GPU.

## 100-sequence smoke

```bash
CUDA_VISIBLE_DEVICES=<GPU> HF_HOME="$HF_HOME" bash jobs/run_esm2_oar.sh \\
  --manifest "$STAGED_ROOT/manifests/gpu/gpu_manifest_core_50k.parquet" \\
  --fasta "$STAGED_ROOT/artifacts/gpu/core_50k.faa" --output-dir "$GPU_OUT/smoke_100" \\
  --num-shards 1 --shard-id 0 --batch-size 8 --device cuda --throughput-limit 100 \\
  --model-checkpoint facebook/esm2_t33_650M_UR50D \\
  --model-revision 08e4846e537177426273712802403f7ba8261b6c
```

## 10k throughput

Repeat with `--output-dir "$GPU_OUT/throughput_10k" --throughput-limit 10000`. Keep mixed precision disabled until both runs validate count, dimension, finite values, pooling metadata, context policy, peak VRAM, and throughput.

## Full core, two deterministic shards

```bash
CUDA_VISIBLE_DEVICES=<GPU0> HF_HOME="$HF_HOME" bash jobs/run_esm2_oar.sh --manifest "$STAGED_ROOT/manifests/gpu/gpu_manifest_core_50k.parquet" --fasta "$STAGED_ROOT/artifacts/gpu/core_50k.faa" --output-dir "$GPU_OUT/core_50k" --num-shards 2 --shard-id 0 --batch-size 8 --device cuda --resume --model-revision 08e4846e537177426273712802403f7ba8261b6c
CUDA_VISIBLE_DEVICES=<GPU1> HF_HOME="$HF_HOME" bash jobs/run_esm2_oar.sh --manifest "$STAGED_ROOT/manifests/gpu/gpu_manifest_core_50k.parquet" --fasta "$STAGED_ROOT/artifacts/gpu/core_50k.faa" --output-dir "$GPU_OUT/core_50k" --num-shards 2 --shard-id 1 --batch-size 8 --device cuda --resume --model-revision 08e4846e537177426273712802403f7ba8261b6c
```
""")

    final_outputs = [outputs["status_md"], outputs["status_json"], outputs["gpu_handoff"], outputs["snapshot"]]
    checks = {"all_W2C_00_13_pass": True, "full_pytest_pass": True, "benchmark_sizes_frozen": True, "leakage_gate_pass": True, "gpu_transfer_checksums_verified": True, "persistent_storage_configured": ROOT.is_absolute() and ROOT.is_dir(), "gpu_status_explicit": "IMPLEMENTED_NOT_GPU_VALIDATED" in outputs["gpu_handoff"].read_text(), "no_foundation_model_claim": status["foundation_model_biological_results"] is False}
    if not all(checks.values()):
        raise RuntimeError(f"W2C-14 checks failed: {checks}")
    write_json(outputs["manifest"], {"task_id": "W2C-14", "status": "PASS", "started_at": started, "completed_at": utc_now(), "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID"), "git_commit": commit, "command": " ".join(sys.argv), "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": None} for path in task_manifests], "parameters": {"foundation_models": False}, "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": None} for path in final_outputs], "validation": {"passed": True, "checks": checks}, "scientific_decision": "CORRECTIVE_CPU_SNAPSHOT_COMPLETE_GPU_HANDOFF_READY", "affected_previous_tasks": ["W2-20 corrective snapshot"], "notes": ["GenomeOcean remains blocked by cache access and nonblocking."]})
    print(json.dumps({"status": "PASS", "tasks": len(tasks), "critical_files": len(critical), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

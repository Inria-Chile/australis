#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.finalize import verify_checksum_lines
from polarfunc.provenance import sha256_file, utc_now, write_json


CRITICAL_INPUTS = (
    "reports/WAVE2_STATUS.json",
    "reports/INACH_CPU_NUMBERS.json",
    "manifests/P05/manifest.json",
    "manifests/candidate_pool_v1.parquet",
    "artifacts/sequences/candidate_translation_qc.parquet",
    "reports/wave2/benchmark_policy_blocker.md",
    "artifacts/homology/candidate_split_groups.parquet",
    "manifests/wave2_snapshot.sha256",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the W2C-00 corrective snapshot.")
    parser.add_argument("--remaining-walltime-seconds", required=True, type=int)
    parser.add_argument("--pytest-evidence", required=True, type=Path)
    args = parser.parse_args()
    started = utc_now()
    report_dir = ROOT / "reports/corrective"
    manifest_dir = ROOT / "manifests/corrective/W2C-00"
    outputs = [report_dir / "corrective_preflight.md", report_dir / "corrective_preflight.json", manifest_dir / "manifest.json"]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    if not args.pytest_evidence.is_file() or "passed" not in args.pytest_evidence.read_text():
        raise RuntimeError("Pytest evidence is absent or does not contain a passing summary")
    status = json.loads((ROOT / "reports/WAVE2_STATUS.json").read_text())
    if status["tasks"]["W2-12"]["status"] != "BLOCKED_POLICY_DECISION":
        raise RuntimeError("W2-12 is not explicitly blocked as expected")
    snapshot_lines = (ROOT / "manifests/wave2_snapshot.sha256").read_text().splitlines()
    if not verify_checksum_lines(ROOT, snapshot_lines):
        raise RuntimeError("Wave-2 checksum snapshot verification failed")
    git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "-uall"], text=True)
    if git_status:
        raise RuntimeError(f"Corrective preflight requires a clean tree before outputs: {git_status}")
    inputs = []
    for relative in CRITICAL_INPUTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    checks = {
        "pytest_passed": True,
        "wave2_snapshot_verified": True,
        "w2_12_explicitly_blocked": True,
        "git_tree_clean_before_outputs": True,
        "persistent_storage_is_configured": ROOT.is_absolute() and ROOT.is_dir(),
        "walltime_above_90_minutes": args.remaining_walltime_seconds >= 5400,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Corrective preflight failed: {checks}")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": "W2C-00",
        "status": "PASS",
        "generated_at": utc_now(),
        "host": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "git_commit": git_commit,
        "remaining_walltime_seconds": args.remaining_walltime_seconds,
        "pytest_evidence": str(args.pytest_evidence),
        "inputs": inputs,
        "validation": {"passed": True, "checks": checks},
    }
    write_json(outputs[1], payload)
    outputs[0].write_text("\n".join([
        "# Corrective preflight", "",
        f"- Host: `{payload['host']}`", f"- OAR job: `{payload['oar_job_id']}`",
        f"- Git commit: `{git_commit}`", f"- Remaining walltime: `{args.remaining_walltime_seconds}` seconds",
        f"- Critical inputs checksummed: `{len(inputs)}`", "- Wave-2 snapshot: `VERIFIED`",
        "- W2-12 state: `BLOCKED_POLICY_DECISION`", "- Initial pytest suite: `PASS`", "",
    ]), encoding="utf-8")
    manifest = {
        "task_id": "W2C-00", "status": "PASS", "started_at": started, "completed_at": utc_now(),
        "host": payload["host"], "oar_job_id": payload["oar_job_id"], "git_commit": git_commit,
        "command": " ".join(sys.argv), "inputs": inputs,
        "outputs": [{"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs[:2]],
        "validation": payload["validation"], "scientific_decision": "Proceed to functional-label semantic audit.",
        "affected_previous_tasks": [], "notes": ["No Wave-2 artifact was overwritten."],
    }
    write_json(outputs[2], manifest)
    print(json.dumps({"status": "PASS", "inputs": len(inputs), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

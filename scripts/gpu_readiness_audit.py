#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.sequences import iter_fasta


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def values_cross_splits(frame: pd.DataFrame, column: str) -> int:
    return int(frame.groupby(column, observed=True).split.nunique().gt(1).sum())


def main() -> int:
    core_path = ROOT / "manifests/benchmark_v1_core_50k.parquet"
    extended_path = ROOT / "manifests/benchmark_v1_extended_92272.parquet"
    split_path = ROOT / "manifests/split_manifest_v1.parquet"
    gpu_core_path = ROOT / "manifests/gpu/gpu_manifest_core_50k.parquet"
    gpu_extended_path = ROOT / "manifests/gpu/gpu_manifest_extended_92272.parquet"
    core_fasta = ROOT / "artifacts/gpu/core_50k.faa"
    extended_fasta = ROOT / "artifacts/gpu/extended_92272.faa"
    leakage_path = ROOT / "reports/corrective/benchmark_v1_leakage.json"
    function_audit_path = ROOT / "reports/corrective/function_label_audit.json"
    w2c09_path = ROOT / "manifests/corrective/W2C-09/manifest.json"
    out_json = ROOT / "reports/gpu/gpu_readiness_final.json"
    out_md = ROOT / "reports/gpu/gpu_readiness_final.md"
    for output in (out_json, out_md):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")

    core = pd.read_parquet(core_path)
    extended = pd.read_parquet(extended_path)
    split = pd.read_parquet(split_path)
    gpu_core = pd.read_parquet(gpu_core_path)
    gpu_extended = pd.read_parquet(gpu_extended_path)
    core_ids = set(core.CDHit_ID.astype(str))
    extended_ids = set(extended.CDHit_ID.astype(str))
    gpu_core_ids = set(gpu_core.CDHit_ID.astype(str))
    gpu_extended_ids = set(gpu_extended.CDHit_ID.astype(str))
    core_fasta_ids = [identifier for identifier, _ in iter_fasta(core_fasta)]
    extended_fasta_ids = [identifier for identifier, _ in iter_fasta(extended_fasta)]
    leakage = json.loads(leakage_path.read_text())
    function_audit = json.loads(function_audit_path.read_text())
    w2c09 = json.loads(w2c09_path.read_text())

    allowed_splits = {"train", "validation", "test"}
    allowed_functions = {"strict_known", "strict_unknown"}
    cross_split = {
        column: values_cross_splits(split, column)
        for column in ("CDHit_ID", "AGC_ID", "split_group", "sequence_sha256", "protein_sha256")
    }
    benchmark_hashes = {"core": sha256_file(core_path), "extended": sha256_file(extended_path), "split": sha256_file(split_path)}
    gpu_hashes = {"core": sha256_file(gpu_core_path), "extended": sha256_file(gpu_extended_path)}
    checks = {
        "core_rows_50000": len(core) == 50_000,
        "extended_rows_92272": len(extended) == 92_272,
        "core_nested_in_extended": core_ids < extended_ids,
        "extended_remainder_42272": len(extended_ids - core_ids) == 42_272,
        "benchmark_ids_unique": core.CDHit_ID.is_unique and extended.CDHit_ID.is_unique,
        "all_split_leakage_zero": all(value == 0 for value in cross_split.values()),
        "leakage_manifest_pass": leakage.get("validation", {}).get("passed") is True,
        "function_audit_pass": function_audit.get("status") == "PASS" and all(function_audit.get("checks", {}).values()),
        "function_labels_frozen": set(extended.function_label.dropna().unique()) == allowed_functions,
        "split_values_frozen": set(extended.split.dropna().unique()) == allowed_splits,
        "gpu_core_ids_match": gpu_core_ids == core_ids,
        "gpu_extended_ids_match": gpu_extended_ids == extended_ids,
        "gpu_core_rows_match": len(gpu_core) == len(core),
        "gpu_extended_rows_match": len(gpu_extended) == len(extended),
        "core_fasta_ids_match": len(core_fasta_ids) == len(set(core_fasta_ids)) == 50_000 and set(core_fasta_ids) == core_ids,
        "extended_fasta_ids_match": len(extended_fasta_ids) == len(set(extended_fasta_ids)) == 92_272 and set(extended_fasta_ids) == extended_ids,
        "w2c09_cpu_baselines_pass": w2c09.get("status") == "PASS" and w2c09.get("validation", {}).get("passed") is True,
    }
    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "BLOCKED",
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_version": sorted(set(extended.benchmark_version.astype(str))),
        "counts": {"core": len(core), "extended": len(extended), "remainder": len(extended_ids - core_ids)},
        "cross_split_overlap": cross_split,
        "benchmark_sha256": benchmark_hashes,
        "gpu_manifest_sha256": gpu_hashes,
        "context_audit_from_manifest": {
            "core_gt_1022": int(core.protein_length.gt(1022).sum()),
            "extended_gt_1022": int(extended.protein_length.gt(1022).sum()),
        },
        "checks": checks,
        "production_allowed": passed,
        "notes": [
            "GPU inference consumes frozen manifests and never regenerates labels or splits.",
            "Proteins above the runtime-detected ESM2 context are recorded as explicitly ineligible; the general benchmark is unchanged.",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# GPU readiness final",
        "",
        f"Status: `{'PASS' if passed else 'BLOCKED'}`.",
        f"Core `{len(core):,}`; extended `{len(extended):,}`; remainder `{len(extended_ids-core_ids):,}`.",
        f"Frozen split SHA256: `{benchmark_hashes['split']}`.",
        "",
        "| Gate | Result |",
        "| --- | --- |",
        *[f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in checks.items()],
        "",
        "Production is permitted only while these benchmark and manifest hashes remain unchanged.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

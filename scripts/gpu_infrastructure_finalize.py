#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    reports = [json.loads((ROOT / f"reports/gpu/musa{i}_preflight.json").read_text()) for i in (1, 2, 3)]
    outputs = [
        ROOT / "reports/gpu/musa_preflight_summary.md",
        ROOT / "reports/gpu/storage_topology.json",
        ROOT / "reports/gpu/model_cache_location.json",
        ROOT / "reports/gpu/esm2_cache_validation.json",
        ROOT / "reports/gpu/esm2_cache_validation.md",
    ]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
    required = [snapshot / "config.json", snapshot / "tokenizer_config.json", snapshot / "special_tokens_map.json", snapshot / "vocab.txt"]
    weight_candidates = [snapshot / "model.safetensors", snapshot / "pytorch_model.bin"]
    config = json.loads((snapshot / "config.json").read_text())
    checks = {
        "all_nodes_pass": all(report["status"] == "PASS" for report in reports),
        "all_nodes_same_git_commit": len({report["git_commit"] for report in reports}) == 1,
        "snapshot_revision_exact": snapshot.name == REVISION,
        "config_and_tokenizer_complete": all(path.exists() and path.stat().st_size > 0 for path in required),
        "model_weights_present": any(path.exists() and path.stat().st_size > 0 for path in weight_candidates),
        "esm_architecture": "Esm" in " ".join(config.get("architectures", [])),
        "shared_cache_visible_all_nodes": all(report["cache_visible"] for report in reports),
    }
    passed = all(checks.values())
    files = sorted(path for path in snapshot.iterdir() if path.is_file())
    cache_report = {
        "status": "PASS" if passed else "BLOCKED",
        "model": MODEL,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "snapshot_size_bytes": sum(path.stat().st_size for path in files),
        "files": [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
        "visible_nodes": [report["hostname"] for report in reports],
        "checks": checks,
    }
    topology = {
        "status": "PASS" if all(report["project_visible"] and report["cache_visible"] for report in reports) else "BLOCKED",
        "project_root": str(ROOT),
        "cache_root": str(snapshot.parents[4]),
        "nodes": {report["hostname"]: {"project_visible": report["project_visible"], "cache_visible": report["cache_visible"], "cache_writable": report["cache_writable"]} for report in reports},
        "payload_staging_required": False,
    }
    model_location = {"model": MODEL, "revision": REVISION, "snapshot": str(snapshot), "persistent": True, "shared_across_nodes": True}
    (ROOT / "reports/gpu/storage_topology.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n")
    (ROOT / "reports/gpu/model_cache_location.json").write_text(json.dumps(model_location, indent=2, sort_keys=True) + "\n")
    (ROOT / "reports/gpu/esm2_cache_validation.json").write_text(json.dumps(cache_report, indent=2, sort_keys=True) + "\n")
    table = ["# Musa preflight summary", "", "| Node | Job | GPUs | CUDA | BF16 | Project | Cache | Status |", "| --- | --- | ---: | --- | --- | --- | --- | --- |"]
    for report in reports:
        table.append(f"| {report['hostname']} | {report['oar_job_id']} | {len(report['gpus'])} | {report['torch_cuda']} | {report['bf16_supported']} | {report['project_visible']} | {report['cache_visible']} | {report['status']} |")
    (ROOT / "reports/gpu/musa_preflight_summary.md").write_text("\n".join(table) + "\n")
    (ROOT / "reports/gpu/esm2_cache_validation.md").write_text(
        "# ESM2 cache validation\n\n"
        f"Status: `{'PASS' if passed else 'BLOCKED'}`.\n\n"
        f"Pinned snapshot: `{snapshot}`.\n\n"
        f"Revision: `{REVISION}`.\n\n"
        "All Musa workers must use this local snapshot with offline loading.\n"
    )
    print(json.dumps({"status": cache_report["status"], "checks": checks}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

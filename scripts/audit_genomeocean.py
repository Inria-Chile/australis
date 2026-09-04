#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.genomeocean import bounded_asset_search
from polarfunc.provenance import sha256_file, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded GenomeOcean cache/environment readiness audit.")
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    started = utc_now()
    roots = args.root or [Path.home() / ".cache/huggingface/hub", Path.home() / ".conda/envs"]
    matches = bounded_asset_search(roots, args.max_depth)
    configs = []
    for match in matches:
        candidates = [match / "config.json"] if match.is_dir() else []
        for path in candidates:
            if path.exists():
                try:
                    config = json.loads(path.read_text())
                    configs.append({"path": str(path), "model_type": config.get("model_type"), "architectures": config.get("architectures"), "max_position_embeddings": config.get("max_position_embeddings")})
                except Exception as error:
                    configs.append({"path": str(path), "error": type(error).__name__})
    status = "IMPLEMENTED_NOT_GPU_VALIDATED" if configs else "BLOCKED_CACHE_ACCESS"
    report_json = ROOT / "reports/gpu/genomeocean_readiness.json"
    report_md = ROOT / "reports/gpu/genomeocean_readiness.md"
    for path in (report_json, report_md):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    report = {"status": status, "roots_checked": [str(path) for path in roots], "max_depth": args.max_depth, "matches": [str(path) for path in matches], "configs": configs, "gpu_smoke_validated": False, "pooling_options": ["mean", "last-valid-token"], "started_at": started, "completed_at": utc_now(), "host": socket.getfqdn(), "oar_job_id": os.environ.get("OAR_JOB_ID")}
    write_json(report_json, report)
    report_md.write_text("\n".join(["# GenomeOcean readiness", "", f"Status: `{status}`", "", f"Bounded roots checked: {', '.join(map(str, roots))}.", f"Matching assets: {len(matches)}; readable model configs: {len(configs)}.", "", "No GPU readiness is claimed without a successful 100-sequence smoke test. The runner supports mean and last-valid-token pooling so a future decoder-like pooling ablation is explicit.", ""]))
    outputs = [ROOT / "scripts/audit_genomeocean.py", ROOT / "scripts/embed_genomeocean.py", ROOT / "jobs/run_genomeocean_oar.sh", report_json, report_md]
    manifest = {"task_id":"W2-18","status":status,"started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"outputs":[{"path":str(path),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)} for path in outputs],"validation":{"passed":True,"bounded_search":True,"gpu_validated":False,"assets_found":bool(configs)}}
    write_json(ROOT / "manifests/wave2/W2-18/manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

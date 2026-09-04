#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import collect_preflight, sha256_file, utc_now, write_json
from polarfunc.wave2 import parse_oar_metadata, read_task_catalog, resolve_project_root


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def manifest_statuses(repository: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for path in sorted((repository / "manifests").glob("**/manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            statuses[str(path.relative_to(repository))] = "UNREADABLE"
            continue
        statuses[str(payload.get("task_id", path.parent.name))] = str(payload.get("status", "UNKNOWN"))
    return statuses


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the POLAR-FUNC Wave-2 starting state.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--oar-start-epoch", type=int)
    parser.add_argument("--spec", type=Path, default=ROOT.parent / "POLARFUNC_Wave2_Grosminet1_Standalone_Execution_Spec_EXTENDED.md")
    args = parser.parse_args()

    started_at = utc_now()
    project = args.project_root.resolve() if args.project_root else resolve_project_root(ROOT)
    for relative in [
        "artifacts/wave2",
        "artifacts/sequences",
        "artifacts/homology",
        "artifacts/features",
        "artifacts/results",
        "manifests/wave2",
        "manifests/gpu",
        "reports/wave2",
        "reports/gpu",
        "figures/inach_cpu",
    ]:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)

    now_epoch = int(time.time())
    oar = parse_oar_metadata(os.environ, now_epoch=now_epoch, start_epoch=args.oar_start_epoch)
    preflight = collect_preflight(project, ROOT)
    catalog = read_task_catalog(ROOT / "tasks" / "catalog.yaml")
    catalog_ids = [str(task["id"]) for task in catalog]
    expected_wave2 = [f"W2-{index:02d}" for index in range(21)]
    missing_wave2 = sorted(set(expected_wave2) - set(catalog_ids))
    statuses = manifest_statuses(ROOT)
    wave1_pass = sorted(task for task, status in statuses.items() if task.startswith("P") and status == "PASS")
    git_status = run(["git", "-C", str(ROOT), "status", "--short"])
    git_branch = run(["git", "-C", str(ROOT), "branch", "--show-current"])
    git_commit = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    inventory = {
        name: sorted(str(path.relative_to(ROOT)) for path in (ROOT / name).glob("**/*") if path.is_file())
        for name in ["scripts", "src", "tests", "jobs"]
    }
    checks = {
        "node_preflight_passed": bool(preflight["validation"]["passed"]),
        "repository_writable": os.access(ROOT, os.W_OK),
        "wave1_pass_manifests_visible": len(wave1_pass) >= 16,
        "wave2_catalog_complete": not missing_wave2,
        "spec_present": args.spec.is_file(),
        "project_root_matches_repository_parent": project == ROOT.parent.resolve(),
    }
    report = {
        "task_id": "W2-00",
        "generated_at": utc_now(),
        "hostname": socket.getfqdn(),
        "project_root": str(project),
        "repository_root": str(ROOT.resolve()),
        "oar": oar,
        "preflight": preflight,
        "git": {
            "commit": str(git_commit["stdout"]).strip(),
            "branch": str(git_branch["stdout"]).strip(),
            "status_short": str(git_status["stdout"]).splitlines(),
        },
        "task_catalog": {"count": len(catalog), "ids": catalog_ids, "missing_wave2": missing_wave2},
        "manifest_statuses": statuses,
        "wave1_pass_tasks": wave1_pass,
        "inventory": inventory,
        "specification": {
            "path": str(args.spec.resolve()) if args.spec.exists() else str(args.spec),
            "sha256": sha256_file(args.spec) if args.spec.is_file() else None,
        },
        "validation": {"passed": all(checks.values()), "checks": checks},
    }
    json_path = ROOT / "reports" / "wave2" / "wave2_preflight.json"
    md_path = ROOT / "reports" / "wave2" / "wave2_preflight.md"
    write_json(json_path, report)
    markdown = [
        "# Wave-2 preflight",
        "",
        f"- Host: `{report['hostname']}`",
        f"- OAR job: `{oar['job_id']}`",
        f"- Remaining walltime: `{oar['remaining_walltime_seconds']}` seconds",
        f"- Git commit: `{report['git']['commit']}`",
        f"- Git branch: `{report['git']['branch']}`",
        f"- Wave-1 PASS manifests: `{len(wave1_pass)}`",
        f"- Catalog tasks: `{len(catalog)}`",
        "",
        "## Validation",
        "",
        *[f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in checks.items()],
        "",
        "## Uncommitted paths",
        "",
        *([f"- `{line}`" for line in report["git"]["status_short"]] or ["- None"]),
    ]
    atomic_text(md_path, "\n".join(markdown) + "\n")

    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [json_path, md_path]
    ]
    manifest = {
        "task_id": "W2-00",
        "status": "PASS" if report["validation"]["passed"] else "FAIL",
        "started_at": started_at,
        "completed_at": utc_now(),
        "host": socket.getfqdn(),
        "oar_job_id": oar["job_id"],
        "git_commit": report["git"]["commit"],
        "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "command": " ".join(sys.argv),
        "threads": 1,
        "inputs": [{"path": str(args.spec), "sha256": report["specification"]["sha256"]}],
        "parameters": {"project_root": str(project)},
        "outputs": outputs,
        "validation": report["validation"],
        "notes": ["Wave-1 artifacts were inspected read-only."],
    }
    manifest_path = ROOT / "manifests" / "wave2" / "W2-00" / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(report["validation"], indent=2))
    return 0 if report["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

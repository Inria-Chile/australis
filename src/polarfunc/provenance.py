from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except OSError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}


def _memory_bytes() -> tuple[int, int]:
    values: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", maxsplit=1)
            values[key] = int(raw.strip().split()[0]) * 1024
    total = values.get("MemTotal", os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    available = values.get("MemAvailable", total)
    return total, available


def collect_preflight(
    project_root: Path,
    repository_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    env = dict(os.environ if env is None else env)
    project_root = Path(project_root).resolve()
    repository_root = Path(repository_root).resolve()
    hostname = socket.getfqdn()
    nodefile_value = env.get("OAR_NODEFILE") or env.get("OAR_FILE_NODES")
    nodefile = Path(nodefile_value) if nodefile_value else None
    allocated_nodes = []
    if nodefile and nodefile.is_file():
        allocated_nodes = sorted(set(nodefile.read_text(encoding="utf-8").splitlines()))
    oar_detected = bool(env.get("OAR_JOB_ID") and allocated_nodes)
    warnings: list[str] = []
    if not oar_detected:
        warnings.append("OAR allocation variables or readable OAR node file are missing")
    if allocated_nodes and hostname not in allocated_nodes and hostname.split(".")[0] not in {
        node.split(".")[0] for node in allocated_nodes
    }:
        warnings.append("Current hostname is not listed in OAR_NODEFILE")
    memory_total, memory_available = _memory_bytes()
    disk = shutil.disk_usage(project_root)
    writable = os.access(project_root, os.W_OK)
    if not writable:
        warnings.append("PROJECT_ROOT is not writable")
    expected_host = "grosminet-1"
    if expected_host not in hostname:
        warnings.append(f"Expected {expected_host}, observed {hostname}")
    if memory_total <= 5 * 1024**4:
        warnings.append("Detected RAM is not greater than 5 TiB")
    binaries = {name: shutil.which(name, path=env.get("PATH")) for name in ["git", "sha256sum", "rsync", "gzip", "pigz", "mmseqs"]}
    report = {
        "generated_at": utc_now(),
        "hostname": hostname,
        "user": env.get("USER") or os.environ.get("USER"),
        "platform": platform.platform(),
        "python": {"executable": sys.executable, "version": sys.version},
        "environment_prefix": env.get("CONDA_PREFIX"),
        "oar": {
            "detected": oar_detected,
            "job_id": env.get("OAR_JOB_ID"),
            "nodefile": str(nodefile) if nodefile else None,
            "allocated_nodes": allocated_nodes,
            "walltime": env.get("OAR_JOB_WALLTIME"),
        },
        "resources": {
            "logical_cpus": os.cpu_count(),
            "memory_total_bytes": memory_total,
            "memory_available_bytes": memory_available,
            "project_disk_total_bytes": disk.total,
            "project_disk_free_bytes": disk.free,
        },
        "paths": {
            "project_root": str(project_root),
            "repository_root": str(repository_root),
            "project_writable": writable,
        },
        "binaries": binaries,
        "commands": {"lscpu": _run(["lscpu"]), "lsblk": _run(["lsblk"]), "findmnt": _run(["findmnt", "-T", str(project_root)])},
        "warnings": warnings,
    }
    report["validation"] = {
        "passed": oar_detected
        and not any("not listed" in warning for warning in warnings)
        and writable
        and expected_host in hostname
        and memory_total > 5 * 1024**4,
        "checks": {
            "oar_detected": oar_detected,
            "host_expected": expected_host in hostname,
            "host_allocated": not allocated_nodes or hostname in allocated_nodes,
            "ram_gt_5_tib": memory_total > 5 * 1024**4,
            "project_writable": writable,
        },
    }
    return report


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_task_manifest(
    repository_root: Path,
    task_id: str,
    *,
    started_at: str,
    command: str,
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
    parameters: dict[str, object],
    validation: dict[str, object],
) -> Path:
    git = _run(["git", "-C", str(repository_root), "rev-parse", "HEAD"])
    manifest = {
        "task_id": task_id,
        "status": "PASS" if validation.get("passed") else "FAIL",
        "started_at": started_at,
        "completed_at": utc_now(),
        "hostname": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "git_commit": str(git["stdout"]).strip() if git["returncode"] == 0 else None,
        "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "command": command,
        "inputs": inputs,
        "parameters": parameters,
        "outputs": outputs,
        "validation": validation,
    }
    path = Path(repository_root) / "manifests" / task_id / "manifest.json"
    write_json(path, manifest)
    return path

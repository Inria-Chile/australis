from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


ALLOWED_TASK_STATUSES = frozenset(
    {
        "PENDING",
        "READY",
        "RUNNING",
        "PASS",
        "FAIL",
        "BLOCKED",
        "SKIPPED",
        "IMPLEMENTED_NOT_GPU_VALIDATED",
        "BLOCKED_CACHE_ACCESS",
    }
)


def resolve_project_root(repository_root: Path) -> Path:
    repository = Path(repository_root).resolve(strict=True)
    if repository.name != "polarfunc":
        raise ValueError(f"Expected a polarfunc repository, observed {repository}")
    return repository.parent


def _walltime_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    fields = value.split(":")
    if len(fields) != 3 or not all(field.isdigit() for field in fields):
        raise ValueError(f"Invalid OAR walltime: {value!r}")
    hours, minutes, seconds = map(int, fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid OAR walltime: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def parse_oar_metadata(
    env: Mapping[str, str],
    *,
    now_epoch: int | None = None,
    start_epoch: int | None = None,
) -> dict[str, object]:
    walltime = _walltime_seconds(env.get("OAR_JOB_WALLTIME"))
    remaining = None
    if walltime is not None and now_epoch is not None and start_epoch is not None:
        remaining = max(0, walltime - (now_epoch - start_epoch))
    return {
        "job_id": env.get("OAR_JOB_ID"),
        "nodefile": env.get("OAR_NODEFILE") or env.get("OAR_FILE_NODES"),
        "walltime": env.get("OAR_JOB_WALLTIME"),
        "walltime_seconds": walltime,
        "start_epoch": start_epoch,
        "checked_epoch": now_epoch,
        "remaining_walltime_seconds": remaining,
    }


def read_task_catalog(path: Path) -> list[dict[str, object]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Task catalog must contain a tasks list")
    ids = [str(task.get("id")) for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate task IDs: {', '.join(duplicates)}")
    for task in tasks:
        status = task.get("status")
        if status is not None and status not in ALLOWED_TASK_STATUSES:
            raise ValueError(f"Invalid task status {status!r} for {task.get('id')}")
    return tasks

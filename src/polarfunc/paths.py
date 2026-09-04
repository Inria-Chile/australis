from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class InventoryRecord:
    path: str
    relative_path: str
    size_bytes: int
    mtime_ns: int
    readable: bool
    suffixes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    repository_root: Path
    raw_data_root: Path
    supplementary_root: Path
    local_scratch_root: Path

    def __post_init__(self) -> None:
        values = {
            name: Path(getattr(self, name)).expanduser().resolve()
            for name in (
                "project_root",
                "repository_root",
                "raw_data_root",
                "supplementary_root",
                "local_scratch_root",
            )
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if not self.repository_root.is_relative_to(self.project_root):
            raise ValueError("repository_root must be inside project_root")
        if not self.raw_data_root.is_relative_to(self.project_root):
            raise ValueError("raw_data_root must be inside project_root")

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectPaths":
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls(**{key: Path(value) for key, value in values.items()})


def _is_excluded(path: Path, excluded_roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in excluded_roots)


def inventory_files(paths: ProjectPaths) -> list[InventoryRecord]:
    excluded = [
        paths.repository_root,
        paths.project_root / ".conda",
        paths.project_root / "launch_plans",
        paths.project_root / "runs",
    ]
    records: list[InventoryRecord] = []
    excluded = [root.resolve() for root in excluded]
    for directory, dirnames, filenames in os.walk(paths.project_root):
        directory_path = Path(directory).resolve()
        dirnames[:] = [
            name for name in dirnames if not _is_excluded(directory_path / name, excluded)
        ]
        for filename in filenames:
            path = directory_path / filename
            if _is_excluded(path, excluded):
                continue
            stat = path.stat()
            records.append(
                InventoryRecord(
                    path=str(path),
                    relative_path=str(path.relative_to(paths.project_root)),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    readable=os.access(path, os.R_OK),
                    suffixes="".join(path.suffixes),
                )
            )
    return sorted(records, key=lambda record: record.relative_path)

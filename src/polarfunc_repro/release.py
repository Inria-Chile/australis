from __future__ import annotations

import json
import os
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from . import __version__

REQUIRED_FILES = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "DATA_LICENSES.md",
    "uv.lock",
    ".github/workflows/ci.yml",
}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".parquet", ".pdb", ".pt", ".safetensors"}
IGNORED_PARTS = {
    ".git",
    ".venv",
    "build",
    "dist",
    "launch_plans",
    "preflight_inputs",
    "probes",
    "runs",
}
SOURCE_ARCHIVE_REQUIRED = {
    "CITATION.cff",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DATA_LICENSES.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
    "configs/config.yaml",
    "docs/REPRODUCTION.md",
}


def _archive_policy_violations(archive: Path) -> list[str]:
    if archive.suffix == ".whl" or archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            members = handle.namelist()
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getnames()
    else:
        raise ValueError(f"unsupported release archive: {archive}")
    return sorted(
        member
        for member in members
        if any(part.startswith("._") or part == "__pycache__" for part in Path(member).parts)
    )


def _source_archive_missing_files(archive: Path) -> list[str]:
    if not archive.name.endswith((".tar.gz", ".tgz")):
        return []
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getnames()
    normalized = {"/".join(Path(member).parts[1:]) for member in members}
    return sorted(SOURCE_ARCHIVE_REQUIRED - normalized)


def release_readiness(root: Path, archives: tuple[Path, ...] = ()) -> dict[str, Any]:
    root = root.resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    oversized_or_binary = []
    apple_double = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_PARTS]
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            if path.name.startswith("._"):
                apple_double.append(relative)
            if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.stat().st_size > 5_000_000:
                oversized_or_binary.append(relative)
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version_matches = metadata["project"]["version"] == __version__
    archive_violations = {}
    archive_missing_files = {}
    for archive in archives:
        violations = _archive_policy_violations(archive)
        if violations:
            archive_violations[str(archive)] = violations
        missing_archive_files = _source_archive_missing_files(archive)
        if missing_archive_files:
            archive_missing_files[str(archive)] = missing_archive_files
    checks = {
        "required_files": not missing,
        "version_matches": version_matches,
        "no_large_scientific_artifacts": not oversized_or_binary,
        "clean_release_archives": not archive_violations,
        "complete_source_archives": not archive_missing_files,
    }
    blockers = []
    if missing:
        blockers.append("missing governance or release files")
    if not version_matches:
        blockers.append("package versions disagree")
    if oversized_or_binary:
        blockers.append("large scientific artifacts are present in the publication tree")
    if archive_violations:
        blockers.append("release archives contain AppleDouble or Python cache files")
    if archive_missing_files:
        blockers.append("source archives are missing reproduction or governance files")
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_files": missing,
        "large_or_binary_files": sorted(oversized_or_binary),
        "ignored_apple_double_files": sorted(apple_double),
        "archive_policy_violations": archive_violations,
        "archive_missing_required_files": archive_missing_files,
        "publication_blockers": blockers,
    }


def write_release_readiness(result: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

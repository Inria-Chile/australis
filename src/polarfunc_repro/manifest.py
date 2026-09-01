from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    artifact_id: str
    role: str
    uri: str
    md5: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    license: str | None = None


class ArtifactManifest(BaseModel):
    schema_version: str = "1.0"
    artifacts: list[Artifact]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return MD5 for source-integrity checks published by data repositories."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_local_path(artifact: Artifact, root: Path) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme in {"http", "https"}:
        filename = Path(artifact.artifact_id).name
        if not filename or filename != artifact.artifact_id:
            raise ValueError(f"artifact ID is not a safe filename: {artifact.artifact_id}")
        return root / filename
    local = Path(artifact.uri)
    return local if local.is_absolute() else root / local


def load_manifest(path: Path) -> ArtifactManifest:
    return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _verify_artifact(artifact: Artifact, base: Path) -> dict[str, Any]:
    local = artifact_local_path(artifact, base)
    exists = local.is_file()
    size_ok = exists and (
        artifact.size_bytes is None or local.stat().st_size == artifact.size_bytes
    )
    hash_ok = exists and (artifact.sha256 is None or sha256_file(local) == artifact.sha256)
    md5_ok = exists and (artifact.md5 is None or md5_file(local) == artifact.md5)
    return {
        "artifact_id": artifact.artifact_id,
        "exists": exists,
        "size_ok": bool(size_ok),
        "hash_ok": bool(hash_ok),
        "md5_ok": bool(md5_ok),
        "path": str(local),
    }


def verify_manifest(path: Path, root: Path | None = None, *, workers: int = 1) -> dict[str, Any]:
    manifest = load_manifest(path)
    base = root or path.parent
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        records = [_verify_artifact(artifact, base) for artifact in manifest.artifacts]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(
                executor.map(lambda artifact: _verify_artifact(artifact, base), manifest.artifacts)
            )
    passed = all(r["exists"] and r["size_ok"] and r["hash_ok"] and r["md5_ok"] for r in records)
    return {"status": "PASS" if passed else "FAIL", "records": records}


def write_verification(result: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

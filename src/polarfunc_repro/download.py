from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .manifest import Artifact, ArtifactManifest, artifact_local_path, md5_file, sha256_file

ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def manifest_from_zenodo_payload(payload: dict[str, Any]) -> ArtifactManifest:
    """Convert a Zenodo record response into the repository manifest schema."""
    record_id = str(payload["id"])
    license_id = payload.get("metadata", {}).get("license", {}).get("id")
    artifacts: list[Artifact] = []
    for item in sorted(payload.get("files", []), key=lambda value: value["key"]):
        checksum = str(item.get("checksum", ""))
        algorithm, _, digest = checksum.partition(":")
        links = item.get("links", {})
        uri = links.get("content") or links.get("self")
        if not uri:
            raise ValueError(f"Zenodo file has no downloadable link: {item['key']}")
        artifacts.append(
            Artifact(
                artifact_id=str(item["key"]),
                role=f"zenodo:{record_id}",
                uri=str(uri),
                md5=digest if algorithm.lower() == "md5" else None,
                sha256=digest if algorithm.lower() == "sha256" else None,
                size_bytes=int(item["size"]),
                license=license_id,
            )
        )
    if not artifacts:
        raise ValueError("Zenodo record contains no files")
    return ArtifactManifest(artifacts=artifacts)


def fetch_zenodo_manifest(record_id: str) -> ArtifactManifest:
    request = Request(
        ZENODO_API.format(record_id=record_id), headers={"User-Agent": "polarfunc-repro/0.1"}
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return manifest_from_zenodo_payload(payload)


def validate_download(path: Path, artifact: Artifact) -> None:
    if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"size mismatch for {artifact.artifact_id}")
    if artifact.md5 is not None and md5_file(path) != artifact.md5:
        raise ValueError(f"MD5 mismatch for {artifact.artifact_id}")
    if artifact.sha256 is not None and sha256_file(path) != artifact.sha256:
        raise ValueError(f"SHA-256 mismatch for {artifact.artifact_id}")


def download_artifact(
    artifact: Artifact, destination_root: Path, chunk_size: int = 8 << 20
) -> Path:
    """Download one artifact atomically, resuming a valid HTTP partial response."""
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = artifact_local_path(artifact, destination_root)
    partial = destination.with_name(f".{destination.name}.part")
    if destination.is_file():
        validate_download(destination, artifact)
        return destination

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "polarfunc-repro/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(artifact.uri, headers=headers)
    with urlopen(request, timeout=120) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            for block in iter(lambda: response.read(chunk_size), b""):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    validate_download(partial, artifact)
    os.replace(partial, destination)
    return destination


def download_manifest(manifest: ArtifactManifest, destination_root: Path) -> list[Path]:
    return [download_artifact(artifact, destination_root) for artifact in manifest.artifacts]

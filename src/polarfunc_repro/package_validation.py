from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unsafe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts


def validate_package_index(
    index: Path,
    *,
    reference_manifest: Path,
    reference_id_column: str,
    expected_kinds: set[str],
    archive_checksums: bool = False,
) -> dict[str, Any]:
    columns = [
        "CDHit_ID",
        "bucket",
        "kind",
        "archive_path",
        "member_path",
        "original_sha256",
        "original_bytes",
        "archive_sha256",
    ]
    frame = pq.read_table(index, columns=columns).to_pandas()
    reference_ids = set(
        pq.read_table(reference_manifest, columns=[reference_id_column])
        .column(reference_id_column)
        .to_pylist()
    )
    errors: list[str] = []
    if frame.duplicated(["CDHit_ID", "kind"]).any():
        errors.append("duplicate CDHit_ID/kind pairs")
    if frame[columns].isna().any().any():
        errors.append("package index contains missing values")
    manifest_ids = set(frame["CDHit_ID"])
    if manifest_ids != reference_ids:
        errors.append(
            "package/reference ID mismatch: "
            f"missing={len(reference_ids - manifest_ids)}, "
            f"extra={len(manifest_ids - reference_ids)}"
        )
    observed_kinds = set(frame["kind"])
    if observed_kinds != expected_kinds:
        errors.append(f"unexpected kinds: {sorted(observed_kinds)}")
    per_id_kinds = frame.groupby("CDHit_ID")["kind"].agg(set)
    incomplete_ids = int((per_id_kinds != expected_kinds).sum())
    if incomplete_ids:
        errors.append(f"{incomplete_ids} IDs do not contain every expected kind")

    archives = []
    for archive_path_text, group in frame.groupby("archive_path", sort=True):
        archive_path = Path(archive_path_text)
        expected_hashes = set(group["archive_sha256"])
        if len(expected_hashes) != 1:
            errors.append(f"conflicting archive hashes: {archive_path}")
            continue
        if not archive_path.is_file():
            errors.append(f"missing archive: {archive_path}")
            continue
        expected_members = set(group["member_path"])
        with tarfile.open(archive_path, "r:") as handle:
            members = handle.getmembers()
        member_names = [member.name for member in members if member.isfile()]
        unsafe = sorted(name for name in member_names if _unsafe_member(name))
        if unsafe:
            errors.append(f"unsafe archive members: {archive_path}")
        if len(member_names) != len(set(member_names)):
            errors.append(f"duplicate archive members: {archive_path}")
        member_set = set(member_names)
        if member_set != expected_members:
            errors.append(
                f"archive membership mismatch: {archive_path}; "
                f"missing={len(expected_members - member_set)}, "
                f"extra={len(member_set - expected_members)}"
            )
        actual_hash = _sha256(archive_path) if archive_checksums else None
        expected_hash = next(iter(expected_hashes))
        if actual_hash is not None and actual_hash != expected_hash:
            errors.append(f"archive checksum mismatch: {archive_path}")
        archives.append(
            {
                "path": str(archive_path),
                "bucket": str(group["bucket"].iloc[0]),
                "members": len(member_names),
                "size_bytes": archive_path.stat().st_size,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
        )

    return {
        "status": "PASS" if not errors else "FAIL",
        "index": str(index),
        "rows": len(frame),
        "unique_ids": len(manifest_ids),
        "reference_ids": len(reference_ids),
        "expected_kinds": sorted(expected_kinds),
        "incomplete_ids": incomplete_ids,
        "archives": len(archives),
        "archive_checksums_computed": archive_checksums,
        "total_original_bytes": int(frame["original_bytes"].sum()),
        "total_archive_bytes": sum(record["size_bytes"] for record in archives),
        "errors": errors,
        "archive_records": archives,
    }

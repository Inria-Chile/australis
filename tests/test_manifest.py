import json
from pathlib import Path

from polarfunc_repro.manifest import Artifact, artifact_local_path, md5_file, verify_manifest

ROOT = Path(__file__).parents[1]


def test_example_manifest_passes():
    result = verify_manifest(ROOT / "manifests/example_external_artifacts.json")
    assert result["status"] == "PASS"


def test_parallel_manifest_verification_preserves_records():
    manifest = ROOT / "manifests/example_external_artifacts.json"
    assert verify_manifest(manifest, workers=2) == verify_manifest(manifest, workers=1)


def test_zenodo_content_url_resolves_to_artifact_id(tmp_path):
    artifact = Artifact(
        artifact_id="catalog.fa.gz",
        role="source",
        uri="https://zenodo.org/api/records/1/files/catalog.fa.gz/content",
    )
    assert artifact_local_path(artifact, tmp_path) == tmp_path / "catalog.fa.gz"


def test_modified_artifact_fails(tmp_path):
    (tmp_path / "bad.txt").write_text("changed", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "artifacts": [
            {
                "artifact_id": "bad",
                "role": "fixture",
                "uri": "bad.txt",
                "sha256": "0" * 64,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(path)["status"] == "FAIL"


def test_md5_is_checked(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("polar", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "artifacts": [
            {
                "artifact_id": "artifact",
                "role": "fixture",
                "uri": "artifact.txt",
                "md5": md5_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(path)["status"] == "PASS"

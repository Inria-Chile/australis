import zipfile
from pathlib import Path

from polarfunc_repro.release import REQUIRED_FILES, release_readiness


def test_repository_release_tree_passes():
    result = release_readiness(Path(__file__).parents[1])
    assert result["status"] == "PASS"
    assert result["publication_blockers"] == []


def test_release_archive_rejects_appledouble_files(tmp_path):
    archive = tmp_path / "package.whl"
    with zipfile.ZipFile(archive, "w") as wheel:
        wheel.writestr("polarfunc_repro/__init__.py", "")
        wheel.writestr("polarfunc_repro/._hidden.py", "")

    result = release_readiness(Path(__file__).parents[1], (archive,))

    assert result["status"] == "FAIL"
    assert result["archive_policy_violations"][str(archive)] == [
        "polarfunc_repro/._hidden.py"
    ]


def test_source_archive_requires_reproduction_metadata(tmp_path):
    archive = tmp_path / "package.tar.gz"
    import tarfile

    with tarfile.open(archive, "w:gz"):
        pass

    result = release_readiness(Path(__file__).parents[1], (archive,))

    assert result["status"] == "FAIL"
    assert "DATA_LICENSES.md" in result["archive_missing_required_files"][str(archive)]


def test_release_tree_ignores_runtime_artifacts(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    for relative in REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "polarfunc-repro"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    runtime = root / "runs" / "large_run"
    runtime.mkdir(parents=True)
    (runtime / "embedding.npy").write_bytes(b"not-publication-content")

    result = release_readiness(root)

    assert result["status"] == "PASS"
    assert result["large_or_binary_files"] == []

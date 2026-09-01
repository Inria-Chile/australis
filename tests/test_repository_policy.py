from pathlib import Path

ROOT = Path(__file__).parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".cff"}
FORBIDDEN = (
    "/home/" + "lvalenzuela",
    "/Users/" + "luisvalenzuela",
    "group_storage_" + "rennes",
)


def test_public_sources_do_not_contain_personal_absolute_paths():
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "docs", ROOT / "tests"]
    roots.extend([ROOT / "README.md", ROOT / "DATA_LICENSES.md", ROOT / "CONTRIBUTING.md"])
    offenders = []
    for root in roots:
        paths = root.rglob("*") if root.is_dir() else [root]
        for path in paths:
            if path.is_file() and path.suffix in TEXT_SUFFIXES and not path.name.startswith("._"):
                text = path.read_text(encoding="utf-8")
                if any(token in text for token in FORBIDDEN):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_checksum_manifests_do_not_hash_themselves():
    offenders = []
    for path in (ROOT / "scripts").glob("*.sh"):
        text = path.read_text(encoding="utf-8")
        if ">\"$RUN_DIR/exports/SHA256SUMS\"" in text:
            checksum_command = text.split(">\"$RUN_DIR/exports/SHA256SUMS\"")[0][-300:]
            if "! -name SHA256SUMS" not in checksum_command:
                offenders.append(path.name)
    assert offenders == []

import hashlib
import tarfile
from pathlib import Path

import pandas as pd

from polarfunc_repro.package_validation import validate_package_index


def test_validate_package_index(tmp_path: Path):
    source = tmp_path / "gene.pdb"
    source.write_text("ATOM\n", encoding="ascii")
    archive = tmp_path / "bucket_00.tar"
    member = "00/gene.pdb"
    with tarfile.open(archive, "w") as handle:
        handle.add(source, arcname=member)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    index = tmp_path / "package.parquet"
    pd.DataFrame(
        {
            "CDHit_ID": ["gene"],
            "bucket": ["00"],
            "kind": ["pdb"],
            "archive_path": [str(archive)],
            "member_path": [member],
            "original_sha256": [hashlib.sha256(source.read_bytes()).hexdigest()],
            "original_bytes": [source.stat().st_size],
            "archive_sha256": [archive_sha],
        }
    ).to_parquet(index, index=False)
    reference = tmp_path / "reference.parquet"
    pd.DataFrame({"CDHit_ID": ["gene"]}).to_parquet(reference, index=False)

    result = validate_package_index(
        index,
        reference_manifest=reference,
        reference_id_column="CDHit_ID",
        expected_kinds={"pdb"},
        archive_checksums=True,
    )

    assert result["status"] == "PASS"
    assert result["archives"] == 1
    assert result["archive_records"][0]["actual_sha256"] == archive_sha

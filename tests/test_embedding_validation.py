from pathlib import Path

import numpy as np
import pandas as pd

from polarfunc_repro.embedding_validation import validate_embedding_manifest


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    shard = tmp_path / "shard_00000.npz"
    ids = np.array(["gene-a", "gene-b"])
    np.savez(shard, ids=ids, embeddings=np.ones((2, 3), dtype=np.float32))
    manifest = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "CDHit_ID": ids,
            "shard_id": [0, 0],
            "row_index": [0, 1],
            "dimension": [3, 3],
            "dtype": ["float32", "float32"],
            "model_name": ["model-v1", "model-v1"],
            "sequence_shard_path": [str(shard), str(shard)],
        }
    ).to_parquet(manifest, index=False)
    reference = tmp_path / "reference.parquet"
    pd.DataFrame({"CDHit_ID": ids}).to_parquet(reference, index=False)
    return manifest, reference


def test_validate_embedding_manifest(tmp_path):
    manifest, reference = _fixture(tmp_path)
    result = validate_embedding_manifest(
        manifest,
        id_column="CDHit_ID",
        shard_path_column="sequence_shard_path",
        reference_manifest=reference,
        expected_rows=2,
        expected_dimension=3,
        expected_model="model-v1",
        checksums=True,
    )
    assert result["status"] == "PASS"
    assert result["vectors"] == result["unique_ids"] == 2
    assert result["shards"] == 1
    assert result["shard_records"][0]["sha256"]


def test_validate_embedding_manifest_detects_reference_mismatch(tmp_path):
    manifest, reference = _fixture(tmp_path)
    pd.DataFrame({"CDHit_ID": ["gene-a", "gene-c"]}).to_parquet(reference, index=False)
    result = validate_embedding_manifest(
        manifest,
        id_column="CDHit_ID",
        shard_path_column="sequence_shard_path",
        reference_manifest=reference,
    )
    assert result["status"] == "FAIL"
    assert any("manifest/reference ID mismatch" in error for error in result["errors"])

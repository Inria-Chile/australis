import numpy as np
import pytest

from polarfunc_repro.embedding_index import index_npz_shards


def _write(path, ids, values):
    np.savez(path, ids=np.asarray(ids), embeddings=np.asarray(values, dtype=np.float32))


def test_embedding_index_records_shards_rows_and_contract(tmp_path):
    _write(tmp_path / "shard_00000.npz", ["a", "b"], [[1, 2], [3, 4]])
    _write(tmp_path / "shard_00001.npz", ["c"], [[5, 6]])

    index, summary = index_npz_shards(
        tmp_path.glob("*.npz"),
        root=tmp_path,
        representation="esm2",
        model_revision="revision-1",
        expected_dimension=2,
    )

    assert index["gene_id"].tolist() == ["a", "b", "c"]
    assert index["row_index"].tolist() == [0, 1, 0]
    assert summary["rows"] == 3
    assert summary["dimensions"] == [2]


def test_embedding_index_rejects_duplicate_ids(tmp_path):
    _write(tmp_path / "shard_00000.npz", ["a"], [[1, 2]])
    _write(tmp_path / "shard_00001.npz", ["a"], [[3, 4]])

    with pytest.raises(ValueError, match="duplicate embedding ID"):
        index_npz_shards(
            tmp_path.glob("*.npz"),
            root=tmp_path,
            representation="esm2",
            model_revision="revision-1",
        )

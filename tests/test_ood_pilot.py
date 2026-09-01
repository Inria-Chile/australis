import numpy as np
import pandas as pd

from polarfunc_repro.ood_pilot import fuse_embeddings, load_npz_embeddings


def test_load_npz_embeddings_combines_disjoint_directories(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    np.savez(
        first / "shard_00000.npz",
        ids=np.array(["gene_a"]),
        embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
    )
    np.savez(
        second / "shard_00000.npz",
        ids=np.array(["gene_b"]),
        embeddings=np.array([[3.0, 4.0]], dtype=np.float32),
    )

    result = load_npz_embeddings([first, second])

    assert result["gene_id"].tolist() == ["gene_a", "gene_b"]
    assert np.vstack(result["embedding"]).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_fuse_embeddings_concatenates_vectors_by_gene_id():
    first = pd.DataFrame({"gene_id": ["a", "b"], "embedding": [np.array([1.0]), np.array([2.0])]})
    second = pd.DataFrame(
        {
            "gene_id": ["b", "a"],
            "embedding": [np.array([20.0, 21.0]), np.array([10.0, 11.0])],
        }
    )

    fused = fuse_embeddings(first, second)

    assert fused["gene_id"].tolist() == ["a", "b"]
    assert np.vstack(fused["embedding"]).tolist() == [[1.0, 10.0, 11.0], [2.0, 20.0, 21.0]]

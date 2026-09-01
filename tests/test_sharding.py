from polarfunc_repro.sharding import build_file_index, stable_bucket


def test_stable_bucket_is_order_independent_and_bounded():
    values = {name: stable_bucket(name, 32) for name in ["gene-c", "gene-a", "gene-b"]}
    assert values == {name: stable_bucket(name, 32) for name in reversed(values)}
    assert all(0 <= value < 32 for value in values.values())


def test_file_index_uses_relative_paths(tmp_path):
    nested = tmp_path / "pdb"
    nested.mkdir()
    first = nested / "gene_a.pdb"
    first.write_text("ATOM\n", encoding="utf-8")

    records = build_file_index(tmp_path, [first], bucket_count=8, include_sha256=True)

    assert records[0]["relative_path"] == "pdb/gene_a.pdb"
    assert records[0]["artifact_id"] == "gene_a"
    assert len(records[0]["sha256"]) == 64

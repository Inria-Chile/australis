import gzip

from polarfunc_repro.inventory import inspect_fasta, inspect_tabular


def test_tabular_inventory_counts_header_data_and_columns(tmp_path):
    path = tmp_path / "values.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("id\tvalue\nA\t1\nB\t2\nC\t3\nD\t4\n")

    result = inspect_tabular(path)

    assert result["rows"] == 4
    assert result["columns"] == 2
    assert result["header"] == ["id", "value"]
    assert len(result["preview_rows"]) == 3


def test_fasta_inventory_counts_sequences_and_residues(tmp_path):
    path = tmp_path / "genes.faa.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(">a\nMPEP\n>b\nAA\n")

    result = inspect_fasta(path)

    assert result["sequence_count"] == 2
    assert result["residue_count"] == 6
    assert result["preview_sequence"] == "MPEP"

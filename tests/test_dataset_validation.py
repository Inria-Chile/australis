import pandas as pd

from polarfunc_repro.dataset_validation import validate_parquet_dataset


def test_parquet_dataset_validation_uses_footer_metadata(tmp_path):
    pd.DataFrame({"gene_id": ["a", "b"], "value": [1, 2]}).to_parquet(
        tmp_path / "part-00000.parquet", index=False
    )

    result = validate_parquet_dataset(
        tmp_path, expected_rows=2, required_columns={"gene_id", "value"}
    )

    assert result["status"] == "PASS"
    assert result["rows"] == 2
    assert result["part_count"] == 1


def test_parquet_dataset_validation_reports_contract_failure(tmp_path):
    pd.DataFrame({"gene_id": ["a"]}).to_parquet(tmp_path / "part.parquet", index=False)

    result = validate_parquet_dataset(
        tmp_path, expected_rows=2, required_columns={"gene_id", "missing"}
    )

    assert result["status"] == "FAIL"
    assert result["missing_columns"] == ["missing"]

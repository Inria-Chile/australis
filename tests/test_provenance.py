from polarfunc_repro.provenance import capture_provenance


def test_provenance_contains_runtime_and_config(tmp_path):
    result = capture_provenance(tmp_path, "project:\n  seed: 42\n")
    assert result["generated_at"].endswith("+00:00")
    assert "python" in result
    assert "project:" in result["resolved_config"]

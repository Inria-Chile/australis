import json

from polarfunc_repro.reports import summarize_json_reports


def test_report_summary_counts_statuses(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    result = summarize_json_reports(tmp_path)
    assert result["status"] == "PASS"
    assert result["statuses"] == {"FAIL": 1, "PASS": 2}


def test_invalid_json_is_reported(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    result = summarize_json_reports(tmp_path)
    assert result["status"] == "FAIL"
    assert result["invalid_json"] == ["bad.json"]

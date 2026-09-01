import json

import pytest

from polarfunc_repro.ood_reporting import summarize_ood_runs


def _summary(seed: int, auroc: float) -> dict:
    detector = {
        "auroc": auroc,
        "aupr_ood": 0.7,
        "fpr_at_95_tpr": 0.4,
        "test_balanced_accuracy_at_threshold": 0.6,
        "test_id_false_reject_rate": 0.2,
        "test_ood_false_accept_rate": 0.3,
    }
    return {
        "status": "PASS",
        "config": {"seed": seed},
        "function_partition_counts": {"id": 7, "validation_ood": 1, "test_ood": 2},
        "classification": {"accuracy": 0.5, "macro_f1": 0.4},
        "detectors": {"energy": detector},
    }


def test_summarize_ood_runs_reports_replicate_statistics(tmp_path):
    paths = []
    for seed, auroc in [(1, 0.6), (2, 0.8)]:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_summary(seed, auroc)), encoding="utf-8")
        paths.append(path)

    report = summarize_ood_runs(paths, bootstrap_repeats=200, seed=9)

    assert report["n_runs"] == 2
    assert report["seeds"] == [1, 2]
    assert report["detectors"]["energy"]["auroc"]["mean"] == pytest.approx(0.7)
    assert report["detectors"]["energy"]["auroc"]["min"] == pytest.approx(0.6)


def test_summarize_ood_runs_rejects_incompatible_partitions(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_summary(1, 0.6)), encoding="utf-8")
    incompatible = _summary(2, 0.7)
    incompatible["function_partition_counts"]["id"] = 8
    second.write_text(json.dumps(incompatible), encoding="utf-8")

    with pytest.raises(ValueError, match="partition cardinalities"):
        summarize_ood_runs([first, second], bootstrap_repeats=200)

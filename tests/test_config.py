from pathlib import Path

from polarfunc_repro.config import absolute_paths, compose_config

ROOT = Path(__file__).parents[1]


def test_hydra_config_composes_without_hardcoded_absolute_paths(monkeypatch):
    for name in (
        "POLARFUNC_DATA_ROOT",
        "POLARFUNC_ARTIFACT_ROOT",
        "POLARFUNC_OUTPUT_ROOT",
        "POLARFUNC_MODEL_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = compose_config(ROOT / "configs")
    assert cfg.project.name == "polarfunc-repro"
    stage_ids = {stage.id for stage in cfg.pipeline.stages}
    assert len(stage_ids) == 11
    assert {"ingest", "embeddings", "structural_packaging", "reporting"} <= stage_ids
    assert absolute_paths(cfg) == []


def test_override_is_applied():
    cfg = compose_config(ROOT / "configs", overrides=["project.seed=17"])
    assert cfg.project.seed == 17

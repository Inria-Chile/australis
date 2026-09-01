from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def compose_config(
    config_dir: Path,
    config_name: str = "config",
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Compose and resolve a Hydra configuration from an explicit directory."""
    directory = config_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Hydra config directory does not exist: {directory}")
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def resolved_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)


def absolute_paths(cfg: DictConfig) -> list[str]:
    """Return configured absolute paths, excluding runtime values supplied via env vars."""
    found: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str) and Path(value).is_absolute():
            found.append(value)

    walk(OmegaConf.to_container(cfg, resolve=False))
    return found

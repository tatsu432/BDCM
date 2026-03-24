"""Hydra compose helpers (used by CLI, tests, and :func:`bdcm.config.default_experiment_config`)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from bdcm.config import ExperimentConfig, experiment_config_from_omegaconf

_CONF_DIR = Path(__file__).resolve().parent / "conf"


def compose_config(overrides: Sequence[str] | None = None) -> DictConfig:
    """
    Compose the app config from ``bdcm/conf`` (same as the CLI, without changing cwd).

    Example overrides: ``["experiment=sanity"]``, ``["scm=2", "variant=complex"]``.
    """
    overrides = list(overrides or [])
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="config", overrides=overrides)


def load_experiment_preset(name: str) -> ExperimentConfig:
    """Load ``experiment/{name}.yaml`` (e.g. ``paper``, ``sanity``, ``preview``)."""
    cfg = compose_config([f"experiment={name}"])
    return experiment_config_from_omegaconf(cfg.experiment)


def list_experiment_presets() -> tuple[str, ...]:
    """Names of YAML files under ``conf/experiment/`` (without ``.yaml``)."""
    exp_dir = _CONF_DIR / "experiment"
    if not exp_dir.is_dir():
        return ()
    return tuple(sorted(p.stem for p in exp_dir.glob("*.yaml")))


def config_summary(cfg: DictConfig) -> str:
    """Human-readable config for logging."""
    return OmegaConf.to_yaml(cfg, resolve=True)

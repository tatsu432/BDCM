"""Tests for Hydra composition and experiment presets."""

from __future__ import annotations

import pytest

from bdcm.config import experiment_config_from_omegaconf
from bdcm.hydra_config import (
    compose_config,
    list_experiment_presets,
    load_experiment_preset,
)


@pytest.mark.parametrize("preset", ["paper", "sanity", "preview"])
def test_load_experiment_preset(preset: str) -> None:
    cfg = load_experiment_preset(preset)
    assert cfg.n_obs > 0
    assert cfg.num_seeds >= 1
    assert cfg.schedule.T >= 2


def test_compose_config_overrides_scm_and_experiment() -> None:
    cfg = compose_config(["experiment=sanity", "scm=4", "variant=complex"])
    assert int(cfg.scm) == 4
    assert str(cfg.variant) == "complex"
    assert int(cfg.experiment.num_seeds) == 1
    exp = experiment_config_from_omegaconf(cfg.experiment)
    assert exp.n_obs == 32


def test_list_experiment_presets_contains_expected() -> None:
    names = list_experiment_presets()
    assert "paper" in names
    assert "sanity" in names
    assert "preview" in names


def test_paper_matches_default_experiment_config() -> None:
    from bdcm.config import default_experiment_config

    yaml_paper = load_experiment_preset("paper")
    default_fn = default_experiment_config()
    assert yaml_paper.n_obs == default_fn.n_obs
    assert yaml_paper.num_epochs == default_fn.num_epochs
    assert yaml_paper.schedule.T == default_fn.schedule.T

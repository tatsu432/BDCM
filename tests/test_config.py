"""Unit tests for :mod:`bdcm.config` (schedules, OmegaConf conversion)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from bdcm.config import (
    ExperimentConfig,
    build_diffusion_schedule,
    experiment_config_from_omegaconf,
)


def test_build_diffusion_schedule_shape_and_bounds() -> None:
    T = 20
    s = build_diffusion_schedule(T)
    assert s.T == T
    assert s.beta_t.shape == (T,)
    assert s.alpha_t.shape == (T,)
    assert np.all(s.beta_t > 0) and np.all(s.beta_t <= 0.1 + 1e-9)
    assert np.all((s.alpha_t > 0) & (s.alpha_t <= 1.0))


def test_build_diffusion_schedule_alpha_decreases() -> None:
    s = build_diffusion_schedule(50)
    # Cumulative product (1-beta) should be non-increasing in t
    assert np.all(np.diff(s.alpha_t) <= 1e-12)


def test_experiment_config_from_omegaconf_roundtrip() -> None:
    node = OmegaConf.create(
        {
            "num_seeds": 2,
            "n_obs": 100,
            "n_sample_dcm": 50,
            "num_interventions": 3,
            "lowest_intervention": -1.0,
            "highest_intervention": 1.0,
            "num_epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.001,
            "mu": 0.0,
            "sigma": 1.0,
            "schedule": {"T": 15},
            "flags": {
                "plot_nn_train": False,
                "print_each_mmd": True,
                "show_plot": False,
            },
        }
    )
    cfg = experiment_config_from_omegaconf(node)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.num_seeds == 2
    assert cfg.n_obs == 100
    assert cfg.schedule.T == 15
    assert cfg.flags.plot_nn_train is False
    assert cfg.flags.print_each_mmd is True


def test_experiment_config_from_omegaconf_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        experiment_config_from_omegaconf(OmegaConf.create([1, 2, 3]))

"""Light tests for observational sampling."""

from __future__ import annotations

import numpy as np

from bdcm.hydra_config import load_experiment_preset
from bdcm.infrastructure.sampling import sample_array_u_and_x, sample_t_and_eps_for_x


def _linear_scm(u, ind, x):
    if ind == 0:
        return u
    return x[ind - 1] + u


def test_sample_array_u_and_x_shapes() -> None:
    cfg = load_experiment_preset("sanity")
    d = 3
    u, x = sample_array_u_and_x(d, _linear_scm, cfg)
    assert u.shape == (d, cfg.n_obs)
    assert x.shape == (d, cfg.n_obs)


def test_sample_t_and_eps_for_x_shapes() -> None:
    cfg = load_experiment_preset("sanity")
    d = 4
    t, eps = sample_t_and_eps_for_x(d, cfg)
    assert t.shape == (d, cfg.n_obs)
    assert eps.shape == (d, cfg.n_obs)
    assert np.all((t >= 1) & (t <= cfg.schedule.T))

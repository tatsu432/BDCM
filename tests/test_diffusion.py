"""Tests for diffusion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from bdcm.config import build_diffusion_schedule
from bdcm.infrastructure.diffusion import create_input_1, create_input_for_NN, normalize


def test_normalize_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    z = normalize(x)
    assert abs(np.mean(z)) < 1e-10
    assert abs(np.std(z) - 1.0) < 1e-10


def test_create_input_1_matches_closed_form() -> None:
    rng = np.random.default_rng(1)
    n = 20
    alpha_t = rng.uniform(0.2, 0.99, size=n)
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    y = create_input_1(alpha_t, x, eps)
    expected = np.sqrt(alpha_t) * x + np.sqrt(1.0 - alpha_t) * eps
    np.testing.assert_allclose(y, expected, rtol=1e-12)


def test_create_input_for_NN_single_head_shape() -> None:
    """One DEC head: 1 noisy channel + len(parent) parents + time => num_input columns."""
    schedule = build_diffusion_schedule(10)
    d = 3
    n_obs = 8
    rng = np.random.default_rng(2)
    alpha_t_train = np.ones((d, n_obs)) * 0.5
    x = rng.standard_normal((d, n_obs))
    eps = rng.standard_normal((d, n_obs))
    t_for_x = rng.integers(1, schedule.T + 1, size=(d, n_obs))
    array_num = np.array([4])
    array_eps_idx = np.array([1])
    parent = [[0, 2]]
    outs = create_input_for_NN(
        array_num,
        array_eps_idx,
        alpha_t_train,
        x,
        eps,
        parent,
        t_for_x,
    )
    assert len(outs) == 1
    num_cols = 1 + len(parent[0]) + 1
    assert outs[0].shape == (n_obs, num_cols)
    assert int(array_num[0]) == num_cols

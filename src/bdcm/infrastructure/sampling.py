from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from bdcm.config import ExperimentConfig


def sample_array_u_and_x(
    d: int,
    structural_eq: Callable[..., np.ndarray],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    array_u = np.stack(
        [np.random.normal(config.mu, config.sigma, config.n_obs) for _ in range(d)]
    )
    x = np.zeros([d, config.n_obs])
    for i in range(d):
        x[i] = structural_eq(array_u[i], i, x)
    return array_u, x


def sample_t_and_eps_for_x(d: int, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    T = config.schedule.T
    t_rows = [
        np.array([random.randint(1, T) for _ in range(config.n_obs)], dtype=np.int64)
        for _ in range(d)
    ]
    eps_rows = [
        np.random.normal(config.mu, config.sigma, config.n_obs) for _ in range(d)
    ]
    return np.stack(t_rows), np.stack(eps_rows)


def true_sample(
    d: int,
    structural_eq: Callable[..., np.ndarray],
    ind_cause: int,
    ind_result: int,
    intervened_value_for_cause_node: float,
    array_u: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    x_do = np.zeros([d, config.n_obs])
    for i in range(d):
        if i == ind_cause:
            x_do[i] = np.ones(config.n_obs) * intervened_value_for_cause_node
        else:
            x_do[i] = structural_eq(array_u[i], i, x_do)
    return x_do[ind_result]

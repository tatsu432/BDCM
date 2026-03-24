from __future__ import annotations

import random
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from bdcm.infrastructure.diffusion import dec

if TYPE_CHECKING:
    from bdcm.config import DiffusionSchedule


def sample_outcome_do_cause_dcm(
    intervened_value: float,
    x: np.ndarray,
    array_net_x: Sequence[torch.nn.Module],
    schedule: DiffusionSchedule,
    n_sample_dcm: int,
) -> np.ndarray:
    out = np.zeros(n_sample_dcm)
    nets = list(array_net_x)
    for i in range(n_sample_dcm):
        _ = random.choice(x[1])
        _ = random.choice(x[5])
        x9_sampled = intervened_value
        x10_parents = np.array([x9_sampled])
        x10_sampled = dec(x10_parents, 0, nets, schedule)
        x11_parents = np.array([x9_sampled, x10_sampled])
        out[i] = dec(x11_parents, 1, nets, schedule)
    return out


def sample_outcome_do_cause_bdcm(
    intervened_value: float,
    x: np.ndarray,
    array_net_x: Sequence[torch.nn.Module],
    schedule: DiffusionSchedule,
    n_sample_dcm: int,
) -> np.ndarray:
    out = np.zeros(n_sample_dcm)
    nets = list(array_net_x)
    for i in range(n_sample_dcm):
        x2_sampled = random.choice(x[1])
        x6_sampled = random.choice(x[5])
        x9_sampled = intervened_value
        x10_parents = np.array([x9_sampled])
        x10_sampled = dec(x10_parents, 0, nets, schedule)
        x11_parents = np.array([x2_sampled, x6_sampled, x9_sampled, x10_sampled])
        out[i] = dec(x11_parents, 2, nets, schedule)
    return out

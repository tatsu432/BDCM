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
        x3_sampled = random.choice(x[2])
        x4_sampled = random.choice(x[3])
        x7_parents = np.array([x3_sampled])
        x7_sampled = dec(x7_parents, 0, nets, schedule)
        x8_parents = np.array([x4_sampled])
        x8_sampled = dec(x8_parents, 1, nets, schedule)
        x9_sampled = intervened_value
        x10_parents = np.array([x7_sampled, x9_sampled])
        out[i] = dec(x10_parents, 2, nets, schedule)
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
        x3_sampled = random.choice(x[2])
        x4_sampled = random.choice(x[3])
        x7_parents = np.array([x3_sampled])
        x7_sampled = dec(x7_parents, 0, nets, schedule)
        x8_parents = np.array([x4_sampled])
        x8_sampled = dec(x8_parents, 1, nets, schedule)
        x9_sampled = intervened_value
        x10_parents = np.array([x3_sampled, x4_sampled, x9_sampled])
        out[i] = dec(x10_parents, 3, nets, schedule)
    return out

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
    for i in range(n_sample_dcm):
        x2_sampled = intervened_value
        _ = random.choice(x[2])
        x5_parents = np.array([x2_sampled])
        out[i] = dec(x5_parents, 0, list(array_net_x), schedule)
    return out


def sample_outcome_do_cause_bdcm(
    intervened_value: float,
    x: np.ndarray,
    array_net_x: Sequence[torch.nn.Module],
    schedule: DiffusionSchedule,
    n_sample_dcm: int,
) -> np.ndarray:
    out = np.zeros(n_sample_dcm)
    for i in range(n_sample_dcm):
        x2_sampled = intervened_value
        x3_sampled = random.choice(x[2])
        x5_parents = np.array([x2_sampled, x3_sampled])
        out[i] = dec(x5_parents, 1, list(array_net_x), schedule)
    return out

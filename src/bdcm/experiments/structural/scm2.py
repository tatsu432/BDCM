"""SCM graph 2: structural equations."""

from __future__ import annotations

import numpy as np

from bdcm.infrastructure.diffusion import normalize


def structural_eq_simple(u, ind, x):
    if ind == 0:
        output = u
    elif ind == 1:
        output = x[0] ** 2 + u
    elif ind == 2:
        output = x[1] + u
    elif ind == 3:
        output = x[2] ** 3 + x[2] + u
    elif ind == 4:
        output = x[2] ** 2 + 0.1 + u
    elif ind == 5:
        output = x[1] * x[3] + x[1] * x[4] + x[3] * x[4] + u
    return normalize(output)


def structural_eq_complex(u, ind, x):
    if ind == 0:
        output = u
    elif ind == 1:
        output = (np.sqrt(abs(x[0]) * (abs(u) + 0.1))) / 2 + abs(x[0]) + u / 5
    elif ind == 2:
        output = 1 / (1 + (abs(u) + 0.1) * np.exp(-x[1]))
    elif ind == 3:
        output = (u * (abs(x[2]) + 0.3)) / 5 + u
    elif ind == 4:
        output = 1 / (np.sqrt(abs(u * x[2])) + 0.1) + u
    elif ind == 5:
        output = x[1] ** 2 * x[3] + x[1] * x[4] + x[4] * x[5] + x[1] * u
    return normalize(output)

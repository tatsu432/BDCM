"""SCM graph 4: structural equations."""

from __future__ import annotations

import numpy as np

from bdcm.infrastructure.diffusion import normalize


def structural_eq_simple(u, ind, x):
    if ind == 0:
        output = u
    elif ind == 1:
        output = u
    elif ind == 2:
        output = x[0] + u
    elif ind == 3:
        output = -(x[1] ** 3) + x[1] + u
    elif ind == 4:
        output = x[2] ** 2 + 0.1 + u
    elif ind == 5:
        output = x[3] ** 2 + x[3] + u
    elif ind == 6:
        output = -(x[2] ** 2) + x[2] + u
    elif ind == 7:
        output = 3 * x[3] + 0.1 + u
    elif ind == 8:
        output = x[4] * x[7] + x[4] + x[7] + u
    elif ind == 9:
        output = x[5] * x[6] * x[8] + x[5] * x[6] + u
    return normalize(output)


def structural_eq_complex(u, ind, x):
    if ind == 0:
        output = u
    elif ind == 1:
        output = u
    elif ind == 2:
        output = (np.sqrt(abs(x[0]) * (abs(u) + 0.1))) / 2 + abs(x[0]) + u / 5
    elif ind == 3:
        output = (u * (abs(x[1]) + 0.3)) / 5 + u
    elif ind == 4:
        output = -1 / (1 + (abs(u) + 0.1) * np.exp(-x[2]))
    elif ind == 5:
        output = (u * (abs(x[3]) + 0.3)) / 5 + u
    elif ind == 6:
        output = -(np.sqrt(abs(x[2]) * (abs(u) + 0.1))) / 2 + abs(x[2]) + u / 5
    elif ind == 7:
        output = 3 * x[3] + 0.1 + u
    elif ind == 8:
        output = x[4] ** 3 * x[7] + x[4] + x[7] + u
    elif ind == 9:
        output = x[5] ** 2 * x[6] * x[8] + x[5] * x[6] + u
    return normalize(output)

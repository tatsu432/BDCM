"""SCM graph 5: structural equations."""

from __future__ import annotations

import numpy as np

from bdcm.infrastructure.diffusion import normalize


def structural_eq_simple(u, ind, x):
    if ind == 0:
        output = u
    elif ind == 1:
        output = -x[0] + u
    elif ind == 2:
        output = x[1] + 0.1 + u
    elif ind == 3:
        output = -x[1] + 0.1 + u
    elif ind == 4:
        output = 1.3 * x[0] + u + x[0] * u
    elif ind == 5:
        output = -1.2 * (x[4] + 0.1) + x[4] + u
    elif ind == 6:
        output = -(x[5] ** 2) - x[5] + u
    elif ind == 7:
        output = 3 * x[5] + 0.1 + u
    elif ind == 8:
        output = x[0] * x[4] + (x[0] + 0.1) - x[4] ** 2 + u
    elif ind == 9:
        output = x[8] ** 2 + u
    elif ind == 10:
        output = (
            x[2] * x[3]
            + x[6] * x[7]
            - 0.1
            + x[8] * x[9]
            + x[2] * x[8]
            - x[6] * x[9]
            + x[7]
        )
    return normalize(output)


def structural_eq_complex(u, ind, x):
    if ind == 0:
        output = u
    elif ind == 1:
        output = x[0] * (u + 0.1)
    elif ind == 2:
        output = -(np.sqrt(abs(x[1]) * (abs(u) + 0.1))) / 2 + abs(x[1]) + u / 5
    elif ind == 3:
        output = x[1] + (u + 0.1) / 2 * x[1]
    elif ind == 4:
        output = -1 / (1 + (abs(u) + 0.1) * np.exp(-x[0]))
    elif ind == 5:
        output = (u * (abs(x[4]) + 0.3)) / 5 + u
    elif ind == 6:
        output = x[5] * u + abs(x[5] + 0.01) * abs(u)
    elif ind == 7:
        output = 3 * x[5] + 0.1 + u
    elif ind == 8:
        output = x[0] ** 3 * x[4] + x[0] - x[4] + u
    elif ind == 9:
        output = x[8] * u + (u + 0.1) ** 2
    elif ind == 10:
        output = (
            x[2] * (x[7] - 0.1)
            + x[8] * x[9]
            + x[2] * x[8]
            - x[6] * x[9]
            + x[2] * x[7]
            - x[3] * x[8]
            + x[8] * x[9]
        )
    return normalize(output)

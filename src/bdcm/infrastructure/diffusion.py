from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

import numpy as np
import torch

if TYPE_CHECKING:
    from bdcm.config import DiffusionSchedule


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_alpha_t_train_for_x(
    d: int,
    t_for_x: np.ndarray,
    n_obs: int,
    schedule: DiffusionSchedule,
) -> np.ndarray:
    alpha_t = schedule.alpha_t
    alpha_t_train_for_x = np.zeros((d, n_obs))
    for i in range(n_obs):
        for j in range(d):
            alpha_t_train_for_x[j, i] = alpha_t[t_for_x[j, i] - 1]
    return alpha_t_train_for_x


def create_input_1(
    alpha_t: np.ndarray, x: np.ndarray, epsilon: np.ndarray
) -> np.ndarray:
    return np.sqrt(alpha_t) * x + np.sqrt(1 - alpha_t) * epsilon


def dec(
    x_parents: np.ndarray,
    index_order: int,
    array_net_x: List[torch.nn.Module],
    schedule: DiffusionSchedule,
) -> float:
    """Reverse diffusion decoder Dec_i(Z_i, X_{pa_i})."""
    T = schedule.T
    alpha_t = schedule.alpha_t
    x_hat = np.zeros(T + 1)
    x_hat[T] = np.random.normal(0, 1)
    for i in range(1, T + 1):
        t = T + 1 - i
        array_net_x[index_order].eval()
        input_x = np.append(np.array([x_hat[t]]), x_parents)
        input_x = np.append(input_x, np.array([t]))
        input_x = np.array([input_x])
        input_x_tensor = torch.from_numpy(input_x).float()
        with torch.no_grad():
            output_x_tensor = array_net_x[index_order](input_x_tensor)
        output_x = output_x_tensor.detach().cpu().numpy()
        x_hat[t - 1] = np.sqrt(alpha_t[t - 2] / alpha_t[t - 1]) * x_hat[t] - output_x[
            0, 0
        ] * (
            np.sqrt(alpha_t[t - 2] * (1 - alpha_t[t - 1]) / alpha_t[t - 1])
            - np.sqrt(1 - alpha_t[t - 2])
        )
    return float(x_hat[0])


# Backward-compatible name used by legacy scripts
DEC = dec


def create_input_for_NN(
    array_num_input_for_nn: np.ndarray,
    array_index_for_epsilon: np.ndarray,
    alpha_t_train_for_x: np.ndarray,
    x: np.ndarray,
    epsilon_for_x: np.ndarray,
    parent: List[List[int]],
    t_for_x: np.ndarray,
) -> List[np.ndarray]:
    num_neural_net = len(array_num_input_for_nn)
    array_input_x: List[np.ndarray] = []
    for i in range(num_neural_net):
        ind = int(array_index_for_epsilon[i])
        first_input = create_input_1(
            alpha_t_train_for_x[ind], x[ind], epsilon_for_x[ind]
        )
        parent_rows = x[parent[i]]
        input_x = np.vstack((first_input, parent_rows, t_for_x[ind])).T
        array_input_x.append(input_x)
    return array_input_x

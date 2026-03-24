from __future__ import annotations

import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from bdcm.config import ExperimentConfig
from bdcm.domain.specs import SCMExperimentSpec
from bdcm.infrastructure.diffusion import (
    create_alpha_t_train_for_x,
    create_input_for_NN,
    set_seed,
)
from bdcm.infrastructure.evaluation import (
    append_mmd_plots,
    calculate_overall_mmd,
    save_array,
)
from bdcm.infrastructure.sampling import (
    sample_array_u_and_x,
    sample_t_and_eps_for_x,
    true_sample,
)
from bdcm.infrastructure.training import train_neural_nets

StructuralEq = Callable[[np.ndarray, int, np.ndarray], np.ndarray]
InterventionFn = Callable[[float, np.ndarray, list], np.ndarray]


def _configure_matplotlib() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    plt.style.use("ggplot")
    plt.ioff()


def run_scm_experiment(
    structural_eq: StructuralEq,
    simple_or_complex: str,
    spec: SCMExperimentSpec,
    sample_dcm: InterventionFn,
    sample_bdcm: InterventionFn,
    config: ExperimentConfig | None = None,
) -> None:
    """
    Single orchestration entry: seed loop, training, sampling, MMD evaluation.

    ``sample_dcm`` / ``sample_bdcm`` have signature
    ``(intervened_value, x, array_net_x) -> np.ndarray`` (schedule and n_sample
    should be bound by the caller, e.g. functools.partial).
    """
    _configure_matplotlib()
    cfg = config or ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    array_array_mmd: list[np.ndarray] = []

    for s in tqdm(range(cfg.num_seeds)):
        set_seed(s)
        array_interventions = np.random.uniform(
            cfg.lowest_intervention,
            cfg.highest_intervention,
            cfg.num_interventions,
        )
        array_u, x = sample_array_u_and_x(spec.d, structural_eq, cfg)
        t_for_x, epsilon_for_x = sample_t_and_eps_for_x(spec.d, cfg)
        alpha_t_train_for_x = create_alpha_t_train_for_x(
            spec.d, t_for_x, cfg.n_obs, cfg.schedule
        )
        array_input_x = create_input_for_NN(
            spec.array_num_input_for_nn,
            spec.array_index_for_epsilon,
            alpha_t_train_for_x,
            x,
            epsilon_for_x,
            spec.parent,
            t_for_x,
        )
        array_net_x = train_neural_nets(
            array_input_x,
            epsilon_for_x,
            spec.array_index_for_epsilon,
            spec.array_num_input_for_nn,
            spec.array_titles,
            cfg,
        )
        array_samples = save_array(
            array_interventions,
            sample_dcm,
            sample_bdcm,
            x,
            array_net_x,
        )

        def true_for_iv(iv: float, u: np.ndarray) -> np.ndarray:
            return true_sample(
                spec.d,
                structural_eq,
                spec.ind_cause,
                spec.ind_result,
                iv,
                u,
                cfg,
            )

        append_mmd_plots(
            array_interventions,
            array_samples,
            true_for_iv,
            spec.ind_cause,
            spec.ind_result,
            array_u,
            array_array_mmd,
            spec.name_of_folder,
            s,
            simple_or_complex,
            cfg,
            device,
        )

    calculate_overall_mmd(array_array_mmd, cfg.num_seeds)

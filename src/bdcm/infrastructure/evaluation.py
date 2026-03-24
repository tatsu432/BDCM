from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from bdcm.infrastructure.diffusion import normalize
from bdcm.infrastructure.mmd import mmd as mmd_tensor

if TYPE_CHECKING:
    from bdcm.config import ExperimentConfig

METHOD_TITLES = ("DCM", "BDCM")


def save_array(
    array_interventions: np.ndarray,
    sample_outcome_do_cause_DCM: Callable[..., np.ndarray],
    sample_outcome_do_cause_BDCM: Callable[..., np.ndarray],
    x: np.ndarray,
    array_net_x: Sequence[torch.nn.Module],
) -> np.ndarray:
    n = int(np.size(array_interventions))
    dcm_rows: List[np.ndarray] = []
    bdcm_rows: List[np.ndarray] = []
    for i in range(n):
        dcm_rows.append(
            sample_outcome_do_cause_DCM(float(array_interventions[i]), x, array_net_x)
        )
        bdcm_rows.append(
            sample_outcome_do_cause_BDCM(float(array_interventions[i]), x, array_net_x)
        )
    return np.array([np.vstack(dcm_rows), np.vstack(bdcm_rows)])


def append_mmd_plots(
    array_interventions: np.ndarray,
    array_array_DCM_BDCM_samples: np.ndarray,
    true_sample_fn: Callable[[float, np.ndarray], np.ndarray],
    ind_cause: int,
    ind_result: int,
    array_u: np.ndarray,
    array_array_mmd: List[np.ndarray],
    name_of_folder: str,
    seed_index: int,
    simple_or_complex: str,
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    num_interventions = int(np.size(array_interventions))
    array_mmd_dcm_bdcm = np.zeros((2, num_interventions))
    target_folder = (
        config.results_root / f"results_{name_of_folder}_{simple_or_complex}"
    )
    target_folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_interventions):
        intervened_value = float(array_interventions[i])
        figure, axis = plt.subplots(1, 2, figsize=(12, 5))
        for j in range(2):
            samples_j = array_array_DCM_BDCM_samples[j][i]
            target = true_sample_fn(intervened_value, array_u)
            axis[j].hist(normalize(samples_j), 100, density=True, label="sample")
            axis[j].hist(
                normalize(target), 100, density=True, alpha=0.5, label="target dist"
            )
            axis[j].set_xlabel(
                rf"$X_{{{ind_result + 1}}}|do(X_{{{ind_cause + 1}}} = {intervened_value:.4f})$",
                fontsize=15,
            )
            axis[j].set_ylabel("frequency", fontsize=15)
            axis[j].set_title(METHOD_TITLES[j], fontsize=15)
            axis[j].legend().set_visible(False)
            if j == 0:
                figure.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.025),
                    ncol=2,
                    fontsize=15,
                )
            mmd_value = mmd_tensor(
                torch.tensor([normalize(samples_j)], dtype=torch.float32).T,
                torch.tensor(
                    [normalize(target)[: config.n_sample_dcm]], dtype=torch.float32
                ).T,
                "rbf",
                device,
            )
            array_mmd_dcm_bdcm[j, i] = mmd_value.item()

        file_path = target_folder / f"s={seed_index}_val={intervened_value:.2f}.png"
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.savefig(file_path)
        if config.flags.show_plot:
            plt.show()
        plt.close(figure)

    if config.flags.print_each_mmd:
        for j in range(2):
            print(
                "mean of MMD for {}: {:.3}".format(
                    METHOD_TITLES[j], float(np.mean(array_mmd_dcm_bdcm[j]))
                )
            )
            print(
                "standard deviation of MMD for {}: {:.3}".format(
                    METHOD_TITLES[j], float(np.std(array_mmd_dcm_bdcm[j]))
                )
            )

    array_array_mmd.append(array_mmd_dcm_bdcm)


def calculate_overall_mmd(
    array_array_mmd: Sequence[np.ndarray],
    num_seeds: int,
) -> None:
    all_mmd_dcm: List[np.ndarray] = []
    all_mmd_bdcm: List[np.ndarray] = []
    for i in range(num_seeds):
        all_mmd_dcm.append(array_array_mmd[i][0])
        all_mmd_bdcm.append(array_array_mmd[i][1])
    stacked_dcm = np.concatenate(all_mmd_dcm)
    stacked_bdcm = np.concatenate(all_mmd_bdcm)
    for j, series in enumerate((stacked_dcm, stacked_bdcm)):
        print(
            "mean of all MMD for {}: {:.3}".format(
                METHOD_TITLES[j], float(np.mean(series))
            )
        )
        print(
            "standard deviation of all MMD for {}: {:.3}".format(
                METHOD_TITLES[j], float(np.std(series))
            )
        )

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


def _repo_root() -> Path:
    # src/bdcm/config.py -> parents[2] = repository root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiffusionSchedule:
    T: int
    beta_t: np.ndarray
    alpha_t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta_t", np.asarray(self.beta_t, dtype=np.float64))
        object.__setattr__(self, "alpha_t", np.asarray(self.alpha_t, dtype=np.float64))


def build_diffusion_schedule(T: int) -> DiffusionSchedule:
    beta_t = (0.1 - 0.0001) * (np.linspace(1, T, T) - 1) / (T - 1) + 0.0001
    alpha_t = np.zeros(T, dtype=np.float64)
    for i in range(len(beta_t)):
        if i == 0:
            alpha_t[i] = 1.0 - beta_t[i]
        else:
            alpha_t[i] = (1.0 - beta_t[i]) * alpha_t[i - 1]
    return DiffusionSchedule(T=T, beta_t=beta_t, alpha_t=alpha_t)


@dataclass(frozen=True)
class ExperimentFlags:
    plot_nn_train: bool = True
    print_each_mmd: bool = True
    show_plot: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Runnable experiment hyperparameters (replaces module-level conf)."""

    num_seeds: int = 5
    n_obs: int = 1000
    n_sample_dcm: int = 500
    num_interventions: int = 10
    lowest_intervention: float = -3.0
    highest_intervention: float = 3.0
    num_epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 0.0001
    mu: float = 0.0
    sigma: float = 1.0
    schedule: DiffusionSchedule = field(
        default_factory=lambda: build_diffusion_schedule(100)
    )
    flags: ExperimentFlags = field(default_factory=ExperimentFlags)
    results_root: Path = field(default_factory=lambda: _repo_root() / "results")


def experiment_config_from_omegaconf(experiment_node) -> ExperimentConfig:
    """
    Build :class:`ExperimentConfig` from a Hydra/OmegaConf ``experiment`` subtree
    (see ``bdcm/conf/experiment/*.yaml``).
    """
    from omegaconf import OmegaConf

    OmegaConf.resolve(experiment_node)
    d = OmegaConf.to_container(experiment_node, resolve=True)
    if not isinstance(d, dict):
        raise TypeError("experiment config must resolve to a dict")
    sched = build_diffusion_schedule(int(d["schedule"]["T"]))
    fl = d["flags"]
    flags = ExperimentFlags(
        plot_nn_train=bool(fl["plot_nn_train"]),
        print_each_mmd=bool(fl["print_each_mmd"]),
        show_plot=bool(fl["show_plot"]),
    )
    return ExperimentConfig(
        num_seeds=int(d["num_seeds"]),
        n_obs=int(d["n_obs"]),
        n_sample_dcm=int(d["n_sample_dcm"]),
        num_interventions=int(d["num_interventions"]),
        lowest_intervention=float(d["lowest_intervention"]),
        highest_intervention=float(d["highest_intervention"]),
        num_epochs=int(d["num_epochs"]),
        batch_size=int(d["batch_size"]),
        learning_rate=float(d["learning_rate"]),
        mu=float(d["mu"]),
        sigma=float(d["sigma"]),
        schedule=sched,
        flags=flags,
        results_root=_repo_root() / "results",
    )


def default_experiment_config() -> ExperimentConfig:
    """Paper preset from :file:`bdcm/conf/experiment/paper.yaml` (single source of truth)."""
    from bdcm.hydra_config import load_experiment_preset

    return load_experiment_preset("paper")

"""BDCM experiment package: config, runner, and infrastructure."""

from bdcm.application.runner import run_scm_experiment
from bdcm.config import (
    DiffusionSchedule,
    ExperimentConfig,
    default_experiment_config,
    experiment_config_from_omegaconf,
)
from bdcm.experiments.launch import run_scm
from bdcm.hydra_config import (
    compose_config,
    config_summary,
    list_experiment_presets,
    load_experiment_preset,
)
from bdcm.domain.specs import (
    SCM1_SPEC,
    SCM2_SPEC,
    SCM3_SPEC,
    SCM4_SPEC,
    SCM5_SPEC,
    SCMExperimentSpec,
)

__all__ = [
    "ExperimentConfig",
    "DiffusionSchedule",
    "compose_config",
    "config_summary",
    "default_experiment_config",
    "experiment_config_from_omegaconf",
    "list_experiment_presets",
    "load_experiment_preset",
    "run_scm",
    "run_scm_experiment",
    "SCMExperimentSpec",
    "SCM1_SPEC",
    "SCM2_SPEC",
    "SCM3_SPEC",
    "SCM4_SPEC",
    "SCM5_SPEC",
]

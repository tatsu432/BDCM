"""Wire domain specs and intervention code into :func:`bdcm.application.runner.run_scm_experiment`."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from bdcm.application.runner import run_scm_experiment
from bdcm.config import default_experiment_config
from bdcm.domain.specs import (
    SCM1_SPEC,
    SCM2_SPEC,
    SCM3_SPEC,
    SCM4_SPEC,
    SCM5_SPEC,
)
from bdcm.interventions import scm1, scm2, scm3, scm4, scm5

if TYPE_CHECKING:
    from bdcm.config import ExperimentConfig

_SPECS = {
    1: SCM1_SPEC,
    2: SCM2_SPEC,
    3: SCM3_SPEC,
    4: SCM4_SPEC,
    5: SCM5_SPEC,
}
_INTERVENTION_MODULES = {
    1: scm1,
    2: scm2,
    3: scm3,
    4: scm4,
    5: scm5,
}


def run_scm(
    scm_id: int,
    structural_eq: Callable,
    simple_or_complex: str,
    config: ExperimentConfig | None = None,
) -> None:
    """
    Run the BDCM experiment for synthetic graph ``scm_id`` (1–5).

    ``structural_eq`` matches the signature expected by sampling (``u``, ``ind``, ``x``).
    ``simple_or_complex`` is used for result subfolder naming (e.g. ``simple`` / ``complex``).
    """
    if scm_id not in _SPECS:
        raise ValueError(f"scm_id must be in 1..5, got {scm_id!r}")
    cfg = config or default_experiment_config()
    mod = _INTERVENTION_MODULES[scm_id]
    spec = _SPECS[scm_id]
    sample_dcm = partial(
        mod.sample_outcome_do_cause_dcm,
        schedule=cfg.schedule,
        n_sample_dcm=cfg.n_sample_dcm,
    )
    sample_bdcm = partial(
        mod.sample_outcome_do_cause_bdcm,
        schedule=cfg.schedule,
        n_sample_dcm=cfg.n_sample_dcm,
    )
    run_scm_experiment(
        structural_eq,
        simple_or_complex,
        spec,
        sample_dcm,
        sample_bdcm,
        cfg,
    )

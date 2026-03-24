"""End-to-end smoke: one SCM run with the sanity preset (still trains small nets)."""

from __future__ import annotations

import pytest

from bdcm.experiments.launch import run_scm
from bdcm.experiments.structural.scm1 import structural_eq_simple
from bdcm.hydra_config import load_experiment_preset


@pytest.mark.integration
def test_scm1_sanity_end_to_end() -> None:
    cfg = load_experiment_preset("sanity")
    run_scm(
        1,
        structural_eq_simple,
        "pytest_smoke",
        cfg,
    )

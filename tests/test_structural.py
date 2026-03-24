"""Structural equation modules return finite normalized outputs."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("scm_id", [1, 2, 3, 4, 5])
def test_structural_eq_simple_ind0_finite(scm_id: int) -> None:
    mod = importlib.import_module(f"bdcm.experiments.structural.scm{scm_id}")
    rng = np.random.default_rng(scm_id)
    n = 16
    u = rng.standard_normal(n)
    x = rng.standard_normal((20, n))
    out = mod.structural_eq_simple(u, 0, x)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("scm_id", [1, 2, 3, 4, 5])
def test_structural_eq_complex_ind0_finite(scm_id: int) -> None:
    mod = importlib.import_module(f"bdcm.experiments.structural.scm{scm_id}")
    rng = np.random.default_rng(10 + scm_id)
    n = 12
    u = rng.standard_normal(n)
    x = rng.standard_normal((20, n))
    out = mod.structural_eq_complex(u, 0, x)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))

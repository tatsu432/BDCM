"""Tests for :mod:`bdcm.experiments.launch` validation."""

from __future__ import annotations

import pytest

from bdcm.experiments.launch import run_scm


def test_run_scm_rejects_invalid_id() -> None:
    def _eq(u, ind, x):
        return u

    with pytest.raises(ValueError, match="scm_id"):
        run_scm(0, _eq, "simple")
    with pytest.raises(ValueError, match="scm_id"):
        run_scm(99, _eq, "simple")

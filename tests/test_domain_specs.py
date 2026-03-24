"""Sanity checks on SCM experiment specs."""

from __future__ import annotations

import pytest

from bdcm.domain.specs import (
    SCM1_SPEC,
    SCM2_SPEC,
    SCM3_SPEC,
    SCM4_SPEC,
    SCM5_SPEC,
)


@pytest.mark.parametrize(
    "spec,d_expected",
    [
        (SCM1_SPEC, 5),
        (SCM2_SPEC, 6),
        (SCM3_SPEC, 7),
        (SCM4_SPEC, 10),
        (SCM5_SPEC, 11),
    ],
)
def test_spec_dimension_matches_intervention_indices(spec, d_expected: int) -> None:
    assert spec.d == d_expected
    assert 0 <= spec.ind_cause < spec.d
    assert 0 <= spec.ind_result < spec.d
    assert len(spec.parent) == len(spec.array_index_for_epsilon)
    assert len(spec.array_num_input_for_nn) == len(spec.parent)

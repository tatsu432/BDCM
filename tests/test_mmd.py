"""Tests for MMD kernel."""

from __future__ import annotations

import torch

from bdcm.infrastructure.mmd import mmd


def test_mmd_identical_samples_small_on_cpu() -> None:
    device = torch.device("cpu")
    x = torch.randn(40, 1, dtype=torch.float32)
    y = x.clone()
    stat = mmd(x, y, "rbf", device)
    assert stat.ndim == 0
    assert stat.item() < 1e-3


def test_mmd_shifted_distributions_larger_than_identical() -> None:
    device = torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(60, 1, dtype=torch.float32, generator=g)
    y = torch.randn(60, 1, dtype=torch.float32, generator=g) + 3.0
    same = mmd(x, x, "rbf", device)
    diff = mmd(x, y, "rbf", device)
    assert diff.item() > same.item()


def test_mmd_multiscale_finite() -> None:
    device = torch.device("cpu")
    x = torch.randn(25, 2, dtype=torch.float32)
    y = torch.randn(25, 2, dtype=torch.float32)
    stat = mmd(x, y, "multiscale", device)
    assert torch.isfinite(stat).item()

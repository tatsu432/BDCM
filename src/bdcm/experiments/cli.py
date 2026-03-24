"""Command-line entry for synthetic SCM runs (Hydra + YAML)."""

from __future__ import annotations

import importlib

import hydra
from omegaconf import DictConfig

from bdcm.config import experiment_config_from_omegaconf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    exp_cfg = experiment_config_from_omegaconf(cfg.experiment)
    scm = int(cfg.scm)
    variant = str(cfg.variant)
    if variant not in ("simple", "complex"):
        raise ValueError(f"variant must be 'simple' or 'complex', got {variant!r}")
    if scm not in (1, 2, 3, 4, 5):
        raise ValueError(f"scm must be 1..5, got {scm}")

    struct_mod = importlib.import_module(f"bdcm.experiments.structural.scm{scm}")
    structural_eq = (
        struct_mod.structural_eq_simple
        if variant == "simple"
        else struct_mod.structural_eq_complex
    )
    from bdcm.experiments.launch import run_scm

    run_scm(scm, structural_eq, variant, exp_cfg)


if __name__ == "__main__":
    main()

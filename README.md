<h1 align="center"><b>BDCM</b><br>Diffusion models for causal inference with unmeasured confounders</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python" />
  <a href="https://arxiv.org/abs/2308.03669"><img src="https://img.shields.io/badge/paper-arxiv.2308.03669-B31B1B.svg" alt="arXiv" /></a>
  <a href="https://attend.ieee.org/ssci-2023/"><img src="https://img.shields.io/badge/IEEE-SSCI%202023-00629B.svg" alt="IEEE SSCI 2023" /></a>
</p>

Code for the paper [Diffusion Model in Causal Inference with Unmeasured Confounders](https://arxiv.org/abs/2308.03669) ([IEEE SSCI 2023](https://attend.ieee.org/ssci-2023/)) by [Tatsuhiro Shimizu](https://ss1.xrea.com/tshimizu.s203.xrea.com/works/index.html). **BDCM** (Backdoor Criterion based [DCM](https://arxiv.org/abs/2302.00860)) uses the backdoor idea to choose which variables enter the diffusion decoder when some confounders are unobserved, improving counterfactual estimates versus DCM in synthetic experiments.

## Documentation

| Resource | Description |
|----------|-------------|
| This README | Install, Hydra CLI, Python API, tests |
| [`src/bdcm/conf/config.yaml`](src/bdcm/conf/config.yaml) | App defaults (`scm`, `variant`, …) |
| [`src/bdcm/conf/experiment/`](src/bdcm/conf/experiment/) | Presets: `paper`, `sanity`, `preview` |
| [`tests/`](tests/) | `pytest` suite (unit + optional integration) |

## Installation & quick start

```bash
git clone https://github.com/tatsu432/BDCM.git
cd BDCM
# uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
# or: pip install -e .
```

Run a fast preset (no plots, good for smoke / CI):

```bash
export PYTHONPATH=src   # omit if you used pip install -e .
python -m bdcm.experiments experiment=sanity scm=1 variant=simple
```

Hydra writes under `outputs/` (gitignored). Multirun: `python -m bdcm.experiments -m experiment=paper,sanity`.

## Experiment presets

| Preset | Use |
|--------|-----|
| [`paper`](src/bdcm/conf/experiment/paper.yaml) | Full hyperparameters (default CLI experiment) |
| [`sanity`](src/bdcm/conf/experiment/sanity.yaml) | Short run, no plots — CI / debugging |
| [`preview`](src/bdcm/conf/experiment/preview.yaml) | Short run with plots for a quick visual check |

Override `scm=1..5` and `variant=simple|complex` on the CLI.

## Python API

```python
from bdcm import load_experiment_preset, run_scm
from bdcm.experiments.structural import scm1

run_scm(1, scm1.structural_eq_simple, "simple", load_experiment_preset("sanity"))
```

Compose arbitrary Hydra overrides without the CLI:

```python
from bdcm import compose_config, experiment_config_from_omegaconf

cfg = compose_config(["experiment=preview", "scm=3"])
exp = experiment_config_from_omegaconf(cfg.experiment)
```

`default_experiment_config()` loads the **paper** preset from YAML (same source as the CLI default).

## CLI with Hydra

[Hydra](https://hydra.cc/) loads [`src/bdcm/conf/`](src/bdcm/conf/). Defaults: `experiment=paper`, `scm=1`, `variant=simple`.

```bash
export PYTHONPATH=src
python -m bdcm.experiments
python -m bdcm.experiments experiment=sanity scm=2 variant=complex
```

## Tests

From the repo root (`pythonpath` is set in `pyproject.toml`):

```bash
uv sync --extra dev
uv run pytest
```

- **Fast:** `pytest -m "not integration"` — schedules, presets, MMD, structural SCMs, sampling, validation.
- **Integration:** `tests/test_integration_smoke.py` (trains briefly); marked `@pytest.mark.integration`.

## Related work

- [Diffusion-based Causal Model (DCM)](https://arxiv.org/abs/2302.00860) — observed confounders setting.

## Citation

```
@article{shimizu2023diffusion,
  title={Diffusion Model in Causal Inference with Unmeasured Confounders},
  author={Shimizu, Tatsuhiro},
  journal={arXiv preprint arXiv:2308.03669},
  year={2023}
}
```

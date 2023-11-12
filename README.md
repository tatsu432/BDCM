# Dissusion Model in Causal Inference with Unmeasured Confounders

This repository is for the exeriment conducted in ["Dissusion Model in Causal Inference with Unmeasured Confounders"](https://arxiv.org/abs/2308.03669) ([IEEE SSCI 2023](https://attend.ieee.org/ssci-2023/))" by [Tatsuhiro Shimizu](https://ss1.xrea.com/tshimizu.s203.xrea.com/works/index.html).

## Abstract

We study how to extend the use of the diffusion model to answer the causal question from the observational data under the existence of unmeasured confounders. In Pearl's framework of using a Directed Acyclic Graph (DAG) to capture the causal intervention, a [Diffusion-based Causal Model (DCM)](https://arxiv.org/abs/2302.00860) was proposed incorporating the diffusion model to answer the causal questions more accurately, assuming that all of the confounders are observed. However, unmeasured confounders in practice exist, which hinders DCM from being applicable. To alleviate this limitation of DCM, we propose an extended model called Backdoor Criterion based DCM (BDCM), whose idea is rooted in the Backdoor criterion to find the variables in DAG to be included in the decoding process of the diffusion model so that we can extend DCM to the case with unmeasured confounders. Synthetic data experiment demonstrates that our proposed model captures the counterfactual distribution more precisely than DCM under the unmeasured confounders.

## Citation

```
@article{shimizu2023diffusion,
  title={Diffusion Model in Causal Inference with Unmeasured Confounders},
  author={Shimizu, Tatsuhiro},
  journal={arXiv preprint arXiv:2308.03669},
  year={2023}
}
```


## Requirements and Setup
```bash
# clone the repository
git clone https://github.com/tatsu432/BDCM
```

The versions of Python and necessary packages are specified as follows.

```
[tool.poetry.dependencies]
python = ">=3.9,<3.10"
obp = "0.5.5"
scikit-learn = "1.0.2"
pandas = "1.3.5"
scipy = "1.7.3"
numpy = "^1.22.4"
matplotlib = "^3.5.2"
seaborn = "^0.11.2"
hydra-core = "1.0.7"
```


### Section 4: Synthetic Data Experiment
```
# How does BDCM perform in comparison with DCM under the existence of unmeasured confounders?
```

#### Figure 4

<img width="400" alt="Figure4" src="https://github.com/tatsu432/BDCM/assets/80372303/0c21e1e3-c95d-4953-a43b-21e59654bfab">


```
# Exapmle 10 (Figure 4 and simple structural equations)
SCM1_simple.ipynb
```

<img width="400" alt="SCM1 simple" src="results/res_SCM1_simp.png">

```
# Exapmle 11 (Figure 4 and complex structural equations)
SCM1_complex.ipynb
```
<img width="400" alt="SCM1 comlex" src="results/res_SCM1_comp.png">

#### Figure 5

<img width="524" alt="Figure5" src="https://github.com/tatsu432/BDCM/assets/80372303/b9bcad4a-a728-4f47-b908-261885e017c1">


```
# Exapmle 12 (Figure 5 and simple structural equations)
SCM2_simple.ipynb
```
<img width="400" alt="SCM2 simple" src="results/res_SCM2_simp.png">

```
# Exapmle 13 (Figure 5 and complex structural equations)
SCM2_complex.ipynb
```
<img width="400" alt="SCM2 comlex" src="results/res_SCM2_comp.png">

#### Figure 6
<img width="400" alt="SCM3" src="https://github.com/tatsu432/BDCM/assets/80372303/38fc72cf-418e-483e-982f-f91732ffdca0">



```
# Exapmle 14 (Figure 6 and simple structural equations)
SCM3_simple.ipynb
```
<img width="400" alt="SCM3 simple" src="results/res_SCM3_simp.png">


```
# Exapmle 15 (Figure 6 and complex structural equations)
SCM3_complex.ipynb
```
<img width="400" alt="SCM3 comlex" src="results/res_SCM3_comp.png">

#### Figure 7

<img width="400" alt="SCM4" src="https://github.com/tatsu432/BDCM/assets/80372303/fee6d0c0-b817-4545-9728-3ebf47e73d96">


```
# Exapmle 16 (Figure 7 and simple structural equations)
SCM4_simple.ipynb
```
<img width="400" alt="SCM4 simple" src="results/res_SCM4_simp.png">

```
# Exapmle 17 (Figure 7 and complex structural equations)
SCM4_complex.ipynb
```
<img width="400" alt="SCM4 comlex" src="results/res_SCM4_comp.png">

#### Figure 8

<img width="400" alt="SCM5" src="https://github.com/tatsu432/BDCM/assets/80372303/b1cb20bf-cc96-46df-b79b-418136dbfcc3">

```
# Exapmle 18 (Figure 8 and simple structural equations)
SCM5_simple.ipynb
```
<img width="400" alt="SCM5 simple" src="results/res_SCM5_simp.png">

```
# Exapmle 19 (Figure 8 and complex structural equations)
SCM5_complex.ipynb
```
<img width="400" alt="SCM5 comlex" src="results/res_SCM5_comp.png">


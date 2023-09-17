## Dissusion Model in Causal Inference with Unmeasured Confounders
#### in Proceedings of the 2023 IEEE Symposium Series on Computational Intelligence
- Author: Tatsuhiro Shimizu
-  Affiliation: Waseda University Department of Political Science and Economics, Shinjuku, Tokyo, Japan,
-   ORCID: 0009-0009-9746-3346
-   E-mail: t.shimizu432@akane.waseda.jp
-   arXiv preprint: https://arxiv.org/abs/2308.03669
-   Abstract: We study how to extend the use of the diffusion model to answer the causal question from the observational data under the existence of unmeasured confounders. In Pearl's framework of using a Directed Acyclic Graph (DAG) to capture the causal intervention, a Diffusion-based Causal Model (DCM) was proposed incorporating the diffusion model to answer the causal questions more accurately, assuming that all of the confounders are observed. However, unmeasured confounders in practice exist, which hinders DCM from being applicable. To alleviate this limitation of DCM, we propose an extended model called Backdoor Criterion based DCM (BDCM), whose idea is rooted in the Backdoor criterion to find the variables in DAG to be included in the decoding process of the diffusion model so that we can extend DCM to the case with unmeasured confounders. Synthetic data experiment demonstrates that our proposed model captures the counterfactual distribution more precisely than DCM under the unmeasured confounders.


This is the implementation of the Backdoor Criterion-based Diffusion-based Causal Model (BDCM).

- See SCM1_simple_structural_equations.ipynb for Example 10.

- See SCM1_complex_structural_equations.ipynb for Example 11.

- See SCM2_simple_structural_equations.ipynb for Example 12.

- See SCM2_complex_structural_equations.ipynb for Example 13.

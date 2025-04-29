# AutoLHVs

### Discovering Local Hidden-Variable Models for Arbitrary Multipartite Entangled States and Arbitrary Measurements

[Published in PRX Quantum](https://doi.org/10.1103/PRXQuantum.6.020317)
[![arXiv](https://img.shields.io/badge/arXiv-2407.04673-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2407.04673)

Abstract:

Measurement correlations in quantum systems can exhibit non-local behavior, a fundamental aspect of quantum mechanics with applications such as device-independent quantum information processing. However, it is in general not known which states are local and which ones are not. In particular, it remains an outstanding challenge to explicitly construct local hidden-variable (LHV) models for arbitrary multipartite entangled states. To address this, we use gradient-descent algorithms from machine learning to find LHV models which reproduce the statistics of arbitrary measurements for quantum many-body states. In contrast to previous approaches, our method employs a general ansatz, enabling it to discover LHV models for all local states. Therefore, it for example provides actual estimates for the critical noise levels at which two-qubit Werner states and three-qubit GHZ and W states become local. Furthermore, we find evidence suggesting that two-spin subsystems in the ground states of translationally invariant Hamiltonians are genuinely local, while bigger subsystems are in general not. Our method now offers a quantitative tool for determining the regimes of non-locality in any given physical context, such as non-equilibrium, decoherence or disorder.

--------------------------------------------------

![image](https://github.com/Nick-von-Selzam/AutoLHVs/blob/main/Images/GitHub_image.jpeg)

--------------------------------------------------

The [Examples](Code/Examples.ipynb) notebook shows how to easily apply the algorithm and optimize local hidden-variable (LHV) models for any desired state.


We use Python (3.13.0). For the full package lists (pip and conda) see the text files in the [Requirements](Requirements) folder. Most importantly, we need numpy (2.1.3), sympy (1.13.3), jax (0.4.35) and optax (0.2.4).


The [code](Code), copyright (c) 2025 Nick von Selzam, can be used under [MIT license](LICENSE.txt).

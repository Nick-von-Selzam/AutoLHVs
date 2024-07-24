# AutoLHVs

### Discovering Local Hidden-Variable Models for Arbitrary Multipartite Entangled States and Arbitrary Measurements

[![arXiv](https://img.shields.io/badge/arXiv-2407.04673-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2407.04673)

Abstract:

Measurement correlations in quantum systems can exhibit non-local behavior, a fundamental aspect of quantum mechanics with applications such as device-independent quantum information processing. However, the explicit  construction of local hidden-variable (LHV) models remains an outstanding challenge in the general setting. To address this, we develop an approach that employs gradient-descent algorithms from machine learning to find LHV models which reproduce the statistics of arbitrary measurements for quantum many-body states. In contrast to previous approaches, our method employs a general ansatz, enabling it to discover an LHV model in all cases where the state is local. Therefore, it provides actual estimates for the critical noise levels at which two-qubit Werner states and three-qubit GHZ and W states become non-local. Furthermore, we find evidence suggesting that two-spin subsystems in the ground states of translationally invariant Hamiltonians are local, while bigger subsystems are in general not. Our method now offers a quantitative tool for determining the regimes of non-locality in any given physical context, including scenarios involving non-equilibrium and decoherence. 

--------------------------------------------------

![image](https://github.com/user-attachments/files/16366284/GitHub_teaser.pdf)

--------------------------------------------------

The [Examples](Code/Examples.ipynb) notebook shows how to easily apply the algorithm and optimize local hidden-variable (LHV) models for any desired state.

We use Python (version 3.12.1) and specifically JAX (version 0.4.24). Otherwise we need:

optax (0.1.9)

numpy (1.26.3)

scipy (1.12.0)

matplotlib (3.8.0)


For the full package lists (pip and conda) see the text files in the [Packages](Packages) folder.


The [code](Code), copyright (c) 2024 Nick von Selzam, can be used under MIT license.

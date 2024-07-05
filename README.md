# AutoLHVs

### Optimizing Local Hidden-Variable Models for Quantum Many-Body States

ArXiv link: 

Abstract:

Measurement correlations in quantum systems can exhibit non-local behavior, a fundamental
aspect of quantum mechanics with applications such as device-independent quantum information
processing. However, the understanding of non-locality remains limited, particularly in the context
of measurements with an infinite number of possible settings (e.g., all projective measurements) and
many-body systems. To address this, we developed a machine learning algorithm which optimizes
local hidden-variable (LHV) models to reproduce the measurement statistics of many-body states.
Our method efficiently produces LHV models for projective measurements of spin-1/2 systems, and
provides estimates for the critical visibilities of two-qubit Werner and noisy three-qubit GHZ and W
states. We find evidence suggesting that two-qubit subsystems in the ground states of translationally
invariant local Hamiltonians are local, while bigger subsystems are in general not. Our approach
offers a tool for studying non-locality in all kinds of situations, such as ground states, thermal states,
non-equilibrium states and states subject to different types of noise.

--------------------------------------------------

This repository includes the code, figures and datafiles for the paper above.
The "Examples.ipynb" notebook shows how to easily apply the algorithm and optimize local hidden-variable (LHV) models for any desired state.

We use Python (version 3.12.1) and specifically JAX (version 0.4.24). Otherwise we need:

optax (0.1.9)

numpy (1.26.3)

scipy (1.12.0)

matplotlib (3.8.0)


For the full package lists (pip and conda) see the text files in the "Packages" folder.


The code, (c) Nick von Selzam 2024, can be used under MIT license.

Please contact me via e-mail in case you are interested in using the figures.

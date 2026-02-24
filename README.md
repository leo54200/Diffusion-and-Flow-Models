# CS492C: Diffusion and Flow Models - Implementations & Solvers

A streamlined implementation of state-of-the-art generative frameworks, spanning from Markovian diffusion to continuous-time Flow Matching. Developed during the graduate course at **KAIST (Fall 2025)** under the direction of **Prof. Minhyuk Sung**.

---

## 🔬 Frameworks & Core Logic

### 1. Discrete Diffusion (DDPM & DDIM)
This module implements the transition from stochastic Markov chains to deterministic implicit sampling.
* **DDPM**: Implements Gaussian diffusion via a T-step forward process. The model is trained using a simplified Mean Squared Error (MSE) loss to predict the noise added to a latent state at any given timestep.
* **DDIM**: Introduces a non-Markovian formulation for deterministic sampling. This allows the repository to generate high-quality samples with significantly fewer steps (e.g., 50 vs 1000) using the same pre-trained weights

### 2. Continuous-Time Dynamics (SDE & DPM-Solver)
Reframing diffusion as continuous probability flows through Stochastic Differential Equations (SDEs).
* **Score-Based Modeling:** Maps the relationship between the noise predictor and the data score function $\nabla_{x} \log q(x)$.
* **DPM-Solver:** Implementation of dedicated high-order ODE solvers for Diffusion ODEs. This optimizes the trade-off between sampling speed (low NFE) and image fidelity.

### 3. Flow Matching & Rectified Flow
Moving beyond traditional diffusion via straight-line probability paths.
* **Rectified Flow:** Implements "straight-path" trajectories between noise and data distributions, minimizing the curvature of the generative flow.
* **Optimization:** Leverages regression-based training for more stable convergence and faster inference compared to standard SDE-based approaches.

---
## 📚 Acknowledgements

Implemented by **leo54200** as part of the **CS492C: Diffusion and Flow Models** course at KAIST.

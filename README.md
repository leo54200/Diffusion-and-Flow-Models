# CS492C: Diffusion and Flow Models - Lab Implementations

This repository contains my lab assignments for the KAIST (Fall 2025) graduate course **CS492C: Diffusion and Flow Models**, taught by Prof. Minhyuk Sung.

As a Master's student in Computer Engineering, my goal is to build a deep, practical understanding of modern generative models by implementing their core algorithms from the ground up, based on the course lectures and foundational papers.

---

## 🚀 Assignments & Core Concepts

The assignments are structured to follow the course's progression from foundational diffusion models to modern, high-speed solvers and flow-based models.

### Assignment 1: DDPM and DDIM

This assignment implements the foundational **Denoising Diffusion Probabilistic Model (DDPM)** and its faster implicit variant, **DDIM**.

#### 1. Denoising Diffusion Probabilistic Models (DDPM)

DDPMs are a class of generative models framed as a hierarchical VAE. They consist of two processes:

* **Forward Process (Predefined):** A fixed Markovian process that gradually adds Gaussian noise to an image $x_0$ over $T$ timesteps.
    * The single-step transition is $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$.
    * A key property is the "forward jump" $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\overline{\alpha}_t}x_0, (1-\overline{\alpha}_t)I)$, which allows sampling a noisy $x_t$ directly from $x_0$.

* **Reverse Process (Learned):** A neural network is trained to reverse this noising process, starting from pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoising it to generate a clean sample $x_0$.
    * The network is trained to optimize the "denoising matching term" of the ELBO.
    * Instead of predicting the mean of the reverse step, the model is reparameterized as a **noise predictor $\hat{\epsilon}_\theta(x_t, t)$**.
  * This results in the simplified loss function $L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon_t}\left[ \lVert \hat{\epsilon}_\theta(x_t, t) - \epsilon_t \rVert^2 \right]$

#### 2. Denoising Diffusion Implicit Models (DDIM)

DDIM addresses the primary drawback of DDPM: **slow sampling** (which requires $T=1000$ or more steps).

* **Key Idea:** DDIM introduces a **non-Markovian** forward process that is specifically designed to have the **exact same marginal distribution $q(x_t|x_0)$** as the DDPM.
* **Main Benefit:** Because the marginals and the ELBO objective are the same, the **exact same noise predictor $\hat{\epsilon}_\theta$ trained for DDPM can be used directly for DDIM** without retraining.
* **Fast Sampling:** The DDIM reverse process includes a parameter $\sigma_t$ (or $\eta$) that controls stochasticity.
    * When $\eta=1$, DDIM is identical to DDPM.
    * When $\eta=0$ (or $\sigma_t = 0$), the reverse process becomes **deterministic**.
    * This determinism allows for sampling using a much shorter sub-sequence of timesteps (e.g., $S=10$ or $S=50$ instead of $T=1000$), enabling **accelerated generation**.

---

### Assignment 2: SDEs & DPM-Solver

This assignment re-frames diffusion from the continuous-time perspective of **Stochastic Differential Equations (SDEs)** and implements a high-order solver.

* **Score-Based Models:** Connects the DDPM noise predictor $\hat{\epsilon}_\theta$ to the **score function** $\nabla_{x_t} \log q(x_t)$. The reverse process can be defined as an SDE that depends on this score.
* **ODE Solvers:** The deterministic DDIM ($\eta=0$) process is a discretization of a corresponding **Probability Flow ODE**.
* **DPM-Solver:** (Based on course topics) Implements a fast, high-order solver for this diffusion ODE, which can achieve high-quality samples in even fewer steps.

---

### Assignment 3: Flow Matching

This assignment explores a newer, alternative class of generative models.

* **Rectified Flow:** Instead of a complex SDE, **Flow Matching** (and related methods like Rectified Flow) learns a "straight path" or "rectified flow" to transport samples from a simple prior (noise) to the data distribution.
* **Training:** This model is trained using a simple regression objective (flow matching) which is often more stable and efficient to train than diffusion models.

---

## 📚 Acknowledgements

This code is for academic purposes and is based on the lectures and materials for **CS492C: Diffusion and Flow Models (Fall 2025)** at KAIST, taught by **Prof. Minhyuk Sung**.

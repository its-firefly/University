# Quarter-Car Suspension Optimisation (PSO + Monte Carlo)

## Overview

This project models a **passive quarter-car suspension system** and uses **Particle Swarm Optimisation (PSO)** to find spring/damper parameters that minimise the **variance of sprung-mass acceleration** — a proxy for passenger ride comfort. The optimised design is then stress-tested with a **Monte Carlo simulation** to assess robustness under real-world parameter uncertainty.

```
    +-------------------------+
    |    Sprung Mass (Ms)     |  <-- vehicle body
    +-------------------------+
          |             |
     Spring (Ks)   Damper (Cs)
          |             |
    +-------------------------+
    |   Unsprung Mass (Mu)    |  <-- wheel assembly
    +-------------------------+
          |             |
     Tyre (Kt)     Damper (Ct)
          |             |
    ~~~~~~~~ Road profile ~~~~~~
```

## Methodology

### 1. System Model
The quarter-car is represented in the frequency domain by a transfer function $H_x(s)$ relating sprung-mass acceleration to road displacement, derived from the equations of motion of the sprung/unsprung mass system.

### 2. Road Input
Road roughness is modelled as a displacement power spectral density (PSD):

$$S_d(\omega) = \frac{G_d(\Omega_0)}{2V}\left[\frac{\omega}{V\Omega_0}\right]^{-W}$$

### 3. Objective Function
The acceleration response PSD is computed as:

$$S_x(\omega) = \omega^4 |H_x(\omega)|^2 S_d(\omega)$$

and integrated over frequency (via `scipy.integrate.quad`) to obtain the **variance of sprung-mass acceleration (VSMA)** — the quantity to be minimised.

### 4. Optimisation
**PySwarms** (Particle Swarm Optimisation) searches the design space for suspension stiffness ($K_s$), tyre stiffness ($K_t$), and suspension damping ($C_s$), using **Latin Hypercube Sampling** (`pyDOE`) to generate well-distributed initial swarm positions.

### 5. Robustness — Monte Carlo Analysis
A large-sample Monte Carlo simulation re-evaluates VSMA while varying key parameters (sprung/unsprung mass, stiffness, damping, tyre damping, vehicle speed, road roughness) according to normal and triangular probability distributions, representing real-world variation (e.g. passenger load, road conditions, component wear). A Pearson correlation analysis then identifies which parameters most strongly influence ride comfort.

## Repository Structure

```
KB7043_Assignment1/
└── main.ipynb   # Model definition, PSO optimisation, and Monte Carlo analysis
```

## Requirements

```bash
pip install numpy scipy matplotlib pyswarms pyDOE
```

## Running

Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run all cells. The notebook will:
1. Define the transfer function and PSD models
2. Run PSO to find the optimal suspension parameters
3. Run the Monte Carlo robustness analysis and produce correlation plots

## Key Takeaways

- Softer suspension and tyre stiffness (within the design bounds) reduced sprung-mass acceleration variance, improving ride comfort at the cost of suspension travel.
- Sprung mass and vehicle speed were the parameters most strongly correlated with ride comfort in the sensitivity analysis, highlighting their importance in suspension tuning.

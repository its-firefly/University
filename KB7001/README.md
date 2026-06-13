# Heat Exchanger Analytical Modelling (Plain vs. Finned Tubes)

## Overview

This project develops an analytical model of a counter-flow shell-and-tube **oil-to-water heat exchanger**, evaluating its thermal performance over a range of tube lengths and comparing the results between **plain tubes** and **finned tubes**.

The model calculates key fluid and heat-transfer parameters from first principles and standard correlations, then sweeps tube length to study how heat transfer rate, outlet temperatures, and exchanger effectiveness evolve.

## Methodology

For each fluid stream (hot oil / cold water), the model computes:

- **Heat capacity rate**, $C = \dot{m} \cdot c_p$
- **Reynolds number**, $Re = \dfrac{4 \dot{m}}{\pi d \mu}$, to classify the flow regime (laminar vs. turbulent)
- **Nusselt number**:
  - Oil side: assumed laminar with constant heat flux ($Nu \approx 4.36$), refined using the **Hausen correlation** for thermally developing laminar flow
  - Water side: assumed turbulent, evaluated using the **Dittus–Boelter correlation**
- **Convective heat transfer coefficients** ($h$) derived from the Nusselt numbers
- **Overall heat transfer coefficient (U)** and exchanger **effectiveness/NTU** relations

The analysis is repeated for a **finned-tube configuration**, where additional fin surface area and fin efficiency are incorporated to quantify the improvement in heat dissipation over the plain-tube baseline.

Results are swept across a range of tube lengths (1–100 m) and exported to CSV/Excel for comparison and plotting.

## Repository Structure

```
KB7001/
├── KB7001_Assessment_analytical.ipynb         # Plain-tube heat exchanger model
├── KB7001_Assessment_analytical_finned.ipynb  # Finned-tube heat exchanger model
├── Data.csv / Data2.csv / data3.xlsx          # Generated/exported results data
```

## Requirements

```bash
pip install numpy matplotlib
```

## Running

Open either notebook in Jupyter Notebook, JupyterLab, or VS Code and run all cells. Each notebook:
1. Defines fluid properties and exchanger geometry
2. Computes Reynolds/Nusselt numbers and heat transfer coefficients
3. Sweeps tube length and computes the resulting thermal performance
4. Exports results to CSV and generates comparison plots

## Notes

- Property values (density, viscosity, specific heat, etc.) are treated as constant across the temperature range for simplicity.
- The finned-tube notebook reuses the same correlations as the plain-tube case, with additional terms for fin area and efficiency.

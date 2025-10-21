# Quantitative Macroeconomics — DSGE & Expectations

<div align="center">
<em>Autumn 2023</em>
</div>

---

## Abstract

<div align="left">

| |
|---|
| This project implements a comprehensive quantitative macroeconomic analysis combining Dynamic Stochastic General Equilibrium (DSGE) modeling with expectation formation under uncertainty. In the first part, we develop a Real Business Cycle DSGE model solved through first-order linearization around the deterministic steady state. After deriving household and firm optimality conditions and computing the steady state, we analyze impulse response functions. A positive technology shock raises the real wage proportionally with productivity gains, while a negative preference (labor-leisure) shock lowers the real wage through increased labor supply. In the second part, we study expectation formation under rare, stochastic crises modeled as a two-state Markov chain. Expectations are updated with a simple adaptive learning rule parameterized by λ, and forecasting performance is evaluated using root-mean-squared error (RMSE). Across simulations, the lowest RMSE occurs at λ = 0.25, indicating an optimal balance between responsiveness and stability in expectation updating. The work emphasizes clean numerical implementation, stochastic simulation techniques, and careful interpretation of dynamic propagation mechanisms in modern macroeconomic models. |

</div>

---

## Keywords

DSGE • Real Business Cycle • Impulse Response Functions • Markov Chains • Stochastic Simulation • Adaptive Learning • Expectation Formation • First-Order Approximation • Steady State Analysis

---

<div align="left">

### Author

Abdul Kadir Jeylani Bakari

*HEC Lausanne, Université de Lausanne*

</div>

---

## Repository Structure

```text
Quantitative Macroeconomics (2023)/
├── code/
│   ├── Q1.m           # Learning & Markov crisis simulation
│   └── Q2.mod         # DSGE model in Dynare
├── figures/
│   ├── Q1_1.pdf       # Real vs expected state (t=400-500)
│   ├── Q1_2.pdf       # RMSE by learning parameter λ
│   ├── Q2_1.pdf       # IRF: Technology shock
│   └── Q2_2.pdf       # IRF: Preference shock
├── report/
│   └── Answers.pdf    # Derivations, steady state, FOCs
└── README.md
```

---

## Technical Stack

- **MATLAB**: Numerical simulations and stochastic modeling
- **Dynare**: DSGE model solving and IRF computation

---

## Reproduction Instructions

### Part 1: Expectation Formation (MATLAB)

```matlab
% Navigate to code directory
cd code/

% Run Q1 simulation
Q1

% Outputs:
% - Simulates state process {s_t, y_t}
% - Computes RMSE for different λ values
% - Exports figures/Q1_1.pdf and figures/Q1_2.pdf
```

### Part 2: DSGE Model (Dynare)

```matlab
% Navigate to code directory
cd code/

% Run Dynare model
dynare Q2

% Outputs:
% - Solves DSGE model by first-order approximation
% - Generates IRFs for technology and preference shocks
% - Exports figures/Q2_1.pdf (tech shock) and figures/Q2_2.pdf (preference shock)
```

---

## Key Results

### Expectation Formation

- Real vs expected state dynamics visualized over t = 400–500 time segment
- RMSE minimized at **λ = 0.25**, indicating optimal learning speed
- Trade-off between responsiveness and stability in expectation updating

### DSGE Impulse Responses

**Technology Shock (Positive)**:

- Real wage increases proportionally with productivity gains
- Output and consumption rise
- Standard RBC propagation mechanism

**Preference Shock (Negative - Labor Supply)**:

- Real wage decreases due to increased labor supply
- Hours worked increase
- Consumption adjusts to income effect

---

## Documentation

Full technical derivations, steady-state calculations, and first-order conditions available in `report/Answers.pdf`.


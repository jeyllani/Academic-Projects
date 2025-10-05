# Sustainability-Aware Asset Management: Asset Allocation with a Carbon Objective

<div align="center">
<em>27 mai 2025</em>
</div>

---

## Abstract

<div align="left">
<div style="text-align: justify; max-width: 700px; margin: 0 auto; padding: 10px;">

Can deep portfolio decarbonisation be achieved without eroding financial performance? Using monthly returns and annual CO₂ data for more than 600 North-American equities (2014-2024), we extend mean-variance optimisation into a three-dimensional (μ, σ, carbon) space and evaluate five allocations: (i) an unconstrained minimum-variance portfolio; (ii) a value-weighted benchmark; (iii) a minimum-variance design capped at 50% of its own footprint; (iv) a benchmark-neutral strategy that halves the benchmark footprint subject to a tracking-error budget; and (v) a dynamic Net-Zero portfolio imposing a 10% yearly emissions decline.

Across the sample, carbon-constrained strategies cut financed emissions by 20-50% while delivering Sharpe ratios statistically indistinguishable from—sometimes superior to—the unconstrained benchmark. Their annualised tracking errors cluster around 4.9%, well above the 0.30-0.50% floor to prevent index replication. Only the Net-Zero design at the shortest look-back window (48 months) briefly peaks at 7.4% before reverting below 5% as the estimation horizon lengthens.

Early cuts of up to roughly 50% in carbon footprint appear to entail little to no performance penalty, whereas deeper reductions trade off transparently against the tracking-error "budget". The Net-Zero trajectory remains consistent with a 1.5°C carbon budget while preserving almost the entire equity risk premium, offering counter-evidence to the notion of an inevitable sustainability tax. Overall, the evidence shows that ambitious, Paris-aligned decarbonisation can be embedded in mainstream portfolios through explicit tracking-error limits, offering investors a coherent and auditable framework for aligning assets with the Paris Agreement while maintaining competitive performance.

</div>
</div>

---

## Keywords

Portfolio Optimization • Carbon Constraints • Climate Finance • Tracking Error • Minimum-Variance Portfolio • Net-Zero Trajectories • Three-Dimensional Trade-Off • Sustainable Investing • Paris Alignment

---

<div align="left">

### Authors

Abdul Kadir Jeylani Bakari • Arnaud Küffer • Jules Curti • Tenzin-Minu Zimmer • Yannick Travasa

*HEC Lausanne, Université de Lausanne*

</div>

---

## Code Overview

```mermaid
graph TD
    A[Raw Data] --> B[Data Processing]
    B --> C[Window Generation]
    C --> D[Portfolio Optimization]
    F --> E[Performance Analysis]
    F --> G[Visualization] 

    B --> B0[Harmonization]
    B --> B1[Missing Value Processing] 
    B --> B2[Company Filtering]
    B --> B3[Data Alignment]

    D --> D1[MVP Strategy]
    D --> D2[Value-Weighted]
    D --> D3[Carbon-Constrained]
    D --> D4[Tracking Error]
    D --> D5["Net Zero (-10 p.a.)"]

    D1 --> F[ Returns Computation ]
    D2 --> F
    D3 --> F
    D4 --> F
    D5 --> F

    F --> E1[Performance Metrics]
    F --> E2[Carbon Analysis]
    F --> E3[Comparative Studies]
```

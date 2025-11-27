# Reassessing Accounting Anomalies in an Investable U.S. Equity Universe

<p align="center">
  <em>26 November 2025</em>
</p>

---

## Abstract

<div align="left">
<div style="text-align: justify; max-width: 700px; margin: 0 auto; padding: 10px;">

This paper re-examines the profitability of 25 accounting-based signals in the U.S. equity market over the period 1963–2024. Using CRSP returns, Compustat annual fundamentals, and the CRSP–Compustat Merged database, we construct a point-in-time panel of U.S. common stocks and implement conservative filters (NYSE P20 size breakpoint, minimum price of $5, and historical return requirements) to approximate a liquid, investable universe. We form value-weighted and equal-weighted long–short decile portfolios, correct for delisting bias, and evaluate performance using Fama–French six-factor regressions with Newey–West standard errors and an explicit t>3 benchmark to account for multiple testing. Our results show that, once we focus on value-weighted portfolios in this liquid universe, most fundamental signals fail to deliver statistically significant alphas and appear largely absorbed by modern risk factors. A small subset of strategies remains economically meaningful. A "Cash Cushion" measure of financial flexibility yields an annualised six-factor alpha of roughly 5%, gross of transaction costs, with t ≈ 2.1 in the full sample, and somewhat larger point estimates in the post-1980 and post-2000 subperiods; we interpret this as a promising, but not definitive, candidate for a defensive premium. By contrast, several growth and quality-related signals—most notably ROIC Momentum and Intangible Power—earn large and robustly negative alphas with t>3, suggesting that the market tends to overpay for certain improvement and intangible-capital stories. We highlight that our deeper analysis of Cash Cushion and ROIC Momentum is conducted ex post, based on their full-sample performance. Overall, the evidence points to limited scope for robust, value-weighted alpha from standard accounting signals in U.S. equities, with the most compelling opportunities arising on the short side rather than from long-only fundamental tilts.

</div>
</div>

---

## Keywords

Accounting Anomalies • Fundamental Signals • Multiple Testing • Value-Weighted Portfolios • Financial Flexibility • Short-Selling • U.S. Equities • Asset Pricing • Fama-French Factors • Alpha Generation

---

<div align="left">

### Authors

Abdul Kadir Jeylani Bakari • Arnaud Küffer • Stella Marinelli • Yannick Travasa

*HEC Lausanne, Université de Lausanne*

</div>

---

## Repository Structure

```text
Reassessing Accounting Anomalies (2025)/
├── Article.pdf           # Complete research paper
├── notebooks/            # Jupyter notebooks for analysis
├── data/                 # Portfolio returns and post-processed datasets
├── requirements.txt      # Python dependencies
└── README.md
```

**Data Availability Note**: Raw CRSP/Compustat data are proprietary and not included. The repository provides portfolio returns and processed datasets necessary to replicate regression results and key analyses.

---

## Technical Stack

- **Python** — Pandas, NumPy, statistical analysis
- **Jupyter Notebooks** — Interactive analysis and visualization
- **Data Sources** — CRSP, Compustat, CRSP-Compustat Merged (via institutional access)

---

## Installation & Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Analysis

```bash
# Navigate to notebooks directory
cd notebooks/

# Launch Jupyter
jupyter notebook
```


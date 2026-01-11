
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Active-success.svg)](https://github.com/)

# Adaptive Factor Allocation via Machine Learning Regime Detection

<p align="center">
  <em>11 January 2026</em>
</p>

---

## Abstract

<div align="left">
<div style="text-align: justify; max-width: 800px; margin: 0 auto; padding: 10px;">

Equity factor payoffs are strongly time-varying, raising the practical question of whether factor exposure can be adapted across market regimes using a simple, pre-specified, implementable rule rather than discretionary timing. We study adaptive factor allocation via machine-learning-based regime classification in a long-only U.S. equity setting. Using 25 fundamental characteristics, we construct investable signal portfolios and aggregate them into two tradable sleeves: an Offensive basket (growth-oriented signals) and a Defensive basket (quality and conservative signals). Market regimes are defined from an EWMA volatility proxy computed on an investable benchmark and mapped into a binary high/low state. We train multiple supervised classifiers under a common feature set and evaluate them in a walk-forward design with annual refits at June formation dates and a fixed July–June holding horizon. The out-of-sample evaluation spans July 1980–June 2024 and enforces strict point-in-time constraints and explicit timing alignment to preclude look-ahead bias. Performance is reported gross and net of turnover-scaled transaction costs, and risk-adjusted results are assessed under alternative factor models with HAC inference and multiple-testing diagnostics. We find that several regime-switching configurations improve net performance relative to the market benchmark with an economically modest gross-to-net wedge under annual implementation. However, incremental gains relative to transparent buy-and-hold allocations (including a 50/50 Offensive/Defensive benchmark) are typically modest, benchmark-dependent, and sensitive to the risk-adjustment model. Overall, machine learning can add value for regime-based factor allocation in some configurations, but the effect is heterogeneous and implementation frictions are first-order for net performance.

</div>
</div>

---

## Keywords

Machine Learning • Regime Detection • Factor Investing • Asset Allocation • Volatility Regimes • Long-Only Portfolios • Transaction Costs • Out-of-Sample Evaluation

---

<div align="left">

### Author

**Abdul Kadir Jeylani Bakari**

*HEC Lausanne, Université de Lausanne*

---

## Repository Structure

The repository is organized to ensure full reproducibility of the research paper results.

```text
adaptive-factor-allocation/
├── Paper.pdf                   # Complete research paper
├── Paper_short.pdf             # Shorter version (10 pages + Appendix) of Paper.pdf
├── LICENSE                     # Apache 2.0 License
├── README.md                   # This file
└── project/                    # Complete Research Pipeline (Code, Data, Results)
    ├── README.md               # Technical Pipeline Documentation
    ├── environment.yml         # Conda environment configuration
    ├── requirements.txt        # Python dependencies
    ├── 01_signal_formation...  # Analysis Notebooks (Sequence 01-12)
    ├── data/                   # Processed datasets (Signals, Portfolios, ML Features)
    └── results/                # Output Tables, Figures, and LaTeX exports
```

> **Data Availability Note**: Raw CRSP and Compustat data are proprietary and **not included** in this repository. Users must have a valid WRDS license to access the source files required for the initial `raw_data_processing` step. All intermediate processed data (e.g., portfolio returns) are provided to replicate the post-processing results.

---

## ⚡ Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/jeyllani/Projects/adaptive-factor-allocation.git
cd adaptive-factor-allocation
```

### 2. Install Environment

We recommend using **Conda** to manage the scientific python stack:

```bash
cd project
conda env create -f environment.yml
conda activate afactors
```

### 3. Reproduce Results

Navigate to the `project/` directory and execute the notebooks in sequence (or use the provided reproduction scripts).

```bash
cd project
jupyter notebook
```

*Refer to `project/README.md` for detailed step-by-step technical instructions.*

---

## 🎓 Acknowledgments

Special thanks to **Ms Maria-Pia Lombardo**, **Ms Anna Smirnova**, **Prof. Simon Scheidegger**, and **Prof. Amit Goyal** (HEC Université de Lausanne).

---

## 📬 Contact

For any questions or issues regarding this codebase, please **[contact the author](mailto:abdul.jeylanibakari@unil.ch)**.

---

## 📚 Citation

If you use this repository, please credit the author.


---

## ⚖️ Legal & Financial Disclaimer

**HYPOTHETICAL PERFORMANCE RESULTS HAVE MANY INHERENT LIMITATIONS, SOME OF WHICH ARE DESCRIBED BELOW. NO REPRESENTATION IS BEING MADE THAT ANY ACCOUNT WILL OR IS LIKELY TO ACHIEVE PROFITS OR LOSSES SIMILAR TO THOSE SHOWN. IN FACT, THERE ARE FREQUENTLY SHARP DIFFERENCES BETWEEN HYPOTHETICAL PERFORMANCE RESULTS AND THE ACTUAL RESULTS SUBSEQUENTLY ACHIEVED BY ANY PARTICULAR TRADING PROGRAM.**

ONE OF THE LIMITATIONS OF HYPOTHETICAL PERFORMANCE RESULTS IS THAT THEY ARE GENERALLY PREPARED WITH THE BENEFIT OF HINDSIGHT. IN ADDITION, HYPOTHETICAL TRADING DOES NOT INVOLVE FINANCIAL RISK, AND NO HYPOTHETICAL TRADING RECORD CAN COMPLETELY ACCOUNT FOR THE IMPACT OF FINANCIAL RISK IN ACTUAL TRADING. FOR EXAMPLE, THE ABILITY TO WITHSTAND LOSSES OR TO ADHERE TO A PARTICULAR TRADING PROGRAM IN SPITE OF TRADING LOSSES ARE MATERIAL POINTS WHICH CAN ALSO ADVERSELY AFFECT ACTUAL TRADING RESULTS. THERE ARE NUMEROUS OTHER FACTORS RELATED TO THE MARKETS IN GENERAL OR TO THE IMPLEMENTATION OF ANY SPECIFIC TRADING PROGRAM WHICH CANNOT BE FULLY ACCOUNTED FOR IN THE PREPARATION OF HYPOTHETICAL PERFORMANCE RESULTS AND ALL OF WHICH CAN ADVERSELY AFFECT ACTUAL TRADING RESULTS.

**Educational Purpose**: This code is provided solely for academic context and educational purposes. It is not financial advice.
</div>


---

## ⚖️ License 
Apache 2.0 (See `LICENSE`)

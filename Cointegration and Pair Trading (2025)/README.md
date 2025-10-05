# Cointegration-Based Pair Trading in Energy Sector Equity Markets

<div align="center">
<em>14 avril 2025</em>
</div>

---

## Abstract

<div align="left">

| |
|---|
| This paper explores the theory and practice of cointegration-based pair trading in equity markets. We begin by examining the statistical foundations of cointegration, focusing on how non-stationary asset price series can form a stationary linear combination—an essential requirement for pairs trading. Through unit-root tests (Dickey-Fuller) on daily log prices for several major energy sector stocks, we find that most exhibit non-stationarity (I(1)), except for TOTALENERGIES, which appears stationary. This outcome affects subsequent cointegration analyses, as stationarity is typically assumed for individual series entering a cointegration relationship. Next, we employ Monte Carlo simulations to derive sample-specific critical values for the Dickey-Fuller statistic on both individual time series and residuals. We then identify a statistically significant cointegration relationship between CHEVRON and CONOCOPHILLIPS, evident in both log-price and raw-price regressions. We confirm that the resulting spread is indeed mean-reverting, although it displays a high degree of persistence and slow mean reversion. Building on this evidence, we design and implement a pair trading strategy. We outline its signal generation (based on the normalized spread crossing predefined thresholds), leverage management, and stop-loss rules. The empirical results illustrate how leveraging the pair's mean-reverting spread can be profitable—under controlled leverage and adequate risk management. However, they also highlight the practical challenges of prolonged drawdowns and elevated carrying costs when the spread reverts slowly. Overall, our findings underscore both the theoretical appeal of cointegration-based pair trading and the operational complexities of deploying it in real markets. By integrating robust statistical tests, realistic simulation-based critical values, and a comprehensive risk framework, we provide a practical roadmap for traders seeking to exploit persistent yet slowly mean-reverting deviations between fundamentally related assets. |

</div>

---

## Keywords

Cointegration • Pair Trading • Stationarity • Unit Root Testing (Dickey-Fuller) • Energy Sector Stocks • Mean Reversion • Monte Carlo Simulations • Non-stationary Time Series • Spread Trading • Risk Management • Leverage Constraints

---

<div align="left">

### Authors

Abdul Kadir Jeylani Bakari • Arnaud Küffer • Julien Marti • Yannick Travasa

*HEC Lausanne, Université de Lausanne*

</div>

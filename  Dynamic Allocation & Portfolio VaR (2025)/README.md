
# Dynamic Allocation & Portfolio VaR: A Two-Asset Mean-Variance Analysis

<div align="center">
<em>26 mai 2025</em>
</div>

---

## Abstract

<div align="left">
<div style="text-align: justify; max-width: 700px; margin: 0 auto; padding: 10px;">

This paper develops and empirically evaluates static and dynamic mean-variance portfolio allocations for a two-asset universe (FTSE 100 equities and UK 10-year gilts) under risk aversion parameters (λ=2) and (λ=10). First, we derive closed-form optimal static weights and document their leverage effects. Second, we estimate an AR(1)-GARCH(1,1) model on weekly returns to generate conditional volatility forecasts, which feed into a weekly re-optimization of portfolio weights subject to position limits [-1,+1.5].

Our analysis shows that dynamic allocation substantially increases compound returns and Sharpe ratios for both aggressive and conservative investors, at the cost of higher drawdowns. We then quantify the critical proportional transaction-cost thresholds (f*) (≈0.0195% for λ=2, ≈0.0682% for λ=10) above which the dynamic strategy loses its performance edge.

Finally, we compute one-day ahead 99% Value-at-Risk using three methods—historical unconditional, GARCH-based conditional, and Extreme Value Theory (block maxima)—and demonstrate that the EVT-GEV approach achieves the best empirical coverage across both static and dynamic strategies, underscoring the importance of explicit tail modeling in portfolio risk management. These results highlight the trade-off between dynamic responsiveness and implementation costs, providing practical guidelines for time-varying allocation under realistic market conditions.

</div>
</div>

---

## Keywords

Dynamic asset allocation • Mean-variance optimization • AR(1)-GARCH(1,1) • Value-at-Risk • Extreme Value Theory • Back-testing • Transaction costs • Tail risk

---

<div align="left">

### Auteurs

**Abdul Kadir Jeylani Bakari** *(Main Author)*  
Arnaud Küffer • Julien Marti • Yannick Travasa

*HEC Lausanne, Université de Lausanne*

</div>

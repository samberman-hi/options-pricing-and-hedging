# Methodology Q&A: the 10 hardest questions

Short answers to the questions an interviewer or examiner is most likely to push on. Numbers refer to the committed results: chain snapshot 2026-10-23 14:03 UTC, hedging sample 2005-01-03 to 2026-09-23.

---

**1. "Your seller made 40 bp per month with a t-stat of 12.7. Why isn't that a money machine?"**

Four reasons:
- **The premium is overstated.** I price the option at VIX, which is a variance-swap rate that embeds the put skew and typically sits 1–3 vol points above ATM implied vol. 1 vol point on a 1-month ATM option is about 11.5 bp of spot (vega ≈ 1,149 bp per unit of vol), so this bias alone could account for a large part of the 40 bp.
- **No option bid–ask is charged.** Crossing the bid–ask on the option would cost several more bp.
- **The returns are crash-shaped.** Skewness is −0.75, and the worst window lost 521 bp, about 13 months of average profit, in Oct 2008.
- **The t-stat overstates the information.** Windows overlap, so the effective sample is about 261 independent months. The non-overlapping t-test still gives t = 8.9, but it remains one asset and one regime of vol-selling history.

The economically meaningful result is the mechanism (theta at implied vol vs gamma at realised vol), not the size of the alpha.

**2. "Derive the P&L ≈ ½Γ S²(σ² − σ_r²)dt formula."**

1. Hold −V and +Δ shares, and keep V − ΔS in cash at r. Over dt: dΠ = −dV + Δ(dS + qS dt) + r(V − ΔS)dt.
2. Itô/Taylor expansion: dV = Θdt + ΔdS + ½ΓdS².
3. Substitute. The ΔdS terms cancel: dΠ = −Θdt − ½ΓdS² + qΔS dt + r(V − ΔS)dt.
4. The BSM PDE gives −Θ = (r − q)SΔ + ½σ²S²Γ − rV. Substitute it, and all the r and q terms cancel: dΠ = ½ΓS²(σ²dt − (dS/S)²).

With (dS/S)² ≈ σ_r²dt this is ½ΓS²(σ_h² − σ_r²)dt for the short. The long has the opposite sign, which is the brief's formula. In the data, regressing realised P&L on this sum gives slope 0.968 (HAC SE 0.022, p = 0.15 for slope = 1) and R² of 0.91.

**3. "Why doesn't daily hedging halve your risk relative to weekly, as √5 would suggest?"**

Because discretisation noise is only part of the hedging error. When the hedge vol σ_h differs from realised vol, the P&L contains Σ½ΓS²(σ_h² − σ_r²)Δt. That term depends on the path through Γ: it matters whether the big moves happen near the strike. Rebalancing more often does not remove it.
- **Oracle hedge (realised vol, look-ahead):** the vol term disappears, and sd(ε) scales with slope 0.49 in log h, against 0.5 in theory. Daily error is 28.8 bp against Derman–Kamal's 28.4 bp, and weekly is 2.4× daily.
- **Tradable hedge vols:** the slope is 0.27–0.29 and the weekly/daily ratio is only about 1.45.

More frequent hedging removes the noise, not the bet on vol.

**4. "Why do you have a look-ahead 'oracle' variant? Isn't that cheating?"**

It is not a strategy. It is an experiment that isolates one source of error. Its P&L is never presented as achievable, it is labelled "look-ahead" in every table and chart, and variants A and B use only information available at the sale-day close:
- **A:** VIX at t0.
- **B:** trailing 21-day realised vol ending at t0.

A unit test perturbs a future price and checks that the variant B hedge vol is unchanged.

**5. "Which volatility should you hedge with, implied or realised?"**

Each choice trades off a different risk:
- **Hedging at implied vol** makes the P&L accrue smoothly (each day's P&L is ½ΓS²(σ_i² − σ_r,t²)dt), but the total depends on the path.
- **Hedging at the true realised vol** locks in V(σ_i) − V(σ_r) regardless of the path, but the P&L swings day to day, and you don't know σ_r in advance.

In my data the trailing-realised hedge (B) was no better than the implied hedge (A): the daily error sd was 58 bp against 54 bp, because last month's realised vol is a noisy forecast of next month's. Desks usually hedge at implied vol, since that is also the vol the book is marked at.

**6. "Your surface uses BSM, but SPY options are American. Isn't that wrong?"**

It is an approximation, and I measured its size rather than assuming it away:
- **Only OTM options are used:** puts below the forward and calls above it. That is the liquid side and has the smallest early-exercise value.
- **The error is measured.** I backed out IVs from a 200-step American CRR tree and compared them with BSM:
  - **1 month:** the difference is at most 0.17 vol points.
  - **1 year:** puts struck between spot and the forward come out up to 1.3 vol points lower under the American tree. They are ITM against spot, and with r = 4.4% > q = 1% early exercise has value.
- **The forward check agrees.** Parity-implied forwards are 33–106 bp below the model forward at the longest expiries, which is the direction American put premia push C − P.

A production desk would de-Americanise every quote or fit the surface on European-style SPX options.

**7. "Why is the skew negative, and why does it flatten with maturity?"**

- **Why negative:**
  - *Demand.* Long-only investors buy OTM index puts for protection, and dealers who sell them are left short downside gamma and charge for it.
  - *Physical dynamics.* Index vol rises when prices fall (the leverage effect and spot–vol correlation), and crashes are jumps. Both fatten the left tail of the risk-neutral distribution.
- **Why it flattens:** over longer horizons returns aggregate and look more normal (the central limit effect), and a fixed 10% move is a smaller number of standard deviations when σ√T is larger. 1-month 90/110 skew is 13.7 vol points against 5.9 at one year; the ATM slope dσ/dk falls from about −0.94 at 1 month to −0.20 at 2.2 years.

**8. "How do you know your SVI fit isn't overfitting? It has 5 parameters."**

Every 5th strike in each expiry was held out, the fit was done on the remaining 80%, and the RMSE was measured on the held-out strikes:
- **Accuracy:** the median out-of-sample RMSE is 0.13 vol points, against 0.13 in-sample. A held-out error no worse than the in-sample error means the model is not fitting noise.
- **Arbitrage:** the fitted surface has 0 calendar-arbitrage violations on 1,122 grid points.
- **Caveats:** a few long-dated slices have poorly identified parameters (ρ at its bound, or a large m), and 3 slices breach Lee's wing-slope bound beyond the data. I treat the wings outside the quoted strikes as unreliable and do not use them for any reported number. The 7- and 14-day 90/110 skews involve extrapolation and are flagged on the chart.

**9. "Why Newey–West with 20 lags? And are your p-values honest?"**

Consecutive windows share up to 20 of their 21 daily returns, so their P&Ls are strongly autocorrelated and an iid t-test would greatly overstate significance. A Bartlett-kernel HAC estimator with lag 20 covers the full overlap. As a check that does not depend on the choice of lag, I also run a plain t-test on every 21st window (261 non-overlapping months).

Results I flag as weak:
- **Higher mean P&L for less frequent hedging:** +10 bp, t = 2.1, p = 0.03. It is one of many comparisons and would not survive a multiple-testing correction.
- **Descriptive only:** the Jarque–Bera p-values and the scaling-slope standard error, which come from 5 correlated points, assume independence.

**10. "How do you know the pricing code is right?"**

There are 190 pytest checks, run on every push by GitHub Actions:
- **Textbook values:** Hull's examples (call 4.76 and put 0.81; the index option 51.83; the Greeks example with Δ = 0.522, Γ = 0.066, vega 12.1, Θ = −4.31, ρ = 8.91; the 23.5% implied vol; the American put at 4.49, 4.263, 4.278 and 4.283 for 5, 30, 100 and 500 steps).
- **Calculus:** every Greek, for calls and puts with q > 0, against central finite differences.
- **Identities:** put–call parity to 10⁻¹⁰; American ≥ European; an American call with no dividend equals the European call.
- **Convergence:** CRR → BSM within 0.01 at 1,000 steps.
- **Monte Carlo:** within 4 standard errors of BSM, with standard errors shrinking like 1/√n.
- **Implied vol:** round-trips across a grid, including the Brent fallback path.
- **Hedging:** a hand-computed 2-step hedge, with dividends and costs, matches to 10⁻¹²; the decomposition components sum to the total exactly; Newey–West matches a hand-computed example.

---

## Key numbers to remember

| | |
|---|---|
| Windows / period | 5,465 overlapping 21-day windows, sale dates 2005-01-03 to 2026-09-23 (≈261 independent months) |
| Mean hedged P&L, short ATM call, daily, VIX hedge | **+40.0 bp of spot** (18.4% of a 220 bp premium), HAC t = 12.7; 81% of windows positive |
| Decomposition (daily, VIX) | theta +123.5, gamma −82.1, costs −2.4, residual +1.1 bp |
| Worst window | sold 2008-10-08: −521 bp (−80% of premium); VIX 57.5, realised 90% |
| P&L vs gamma–theta theory | slope 0.968 (SE 0.022), R² 0.91 |
| Hedging-error scaling | oracle slope 0.49 (theory 0.5); daily 28.8 bp vs Derman–Kamal 28.4 bp; tradable hedges 0.27–0.29 |
| Daily vs weekly P&L sd (VIX hedge) | 66 vs 90 bp |
| Cost sensitivity, daily | 42.5 / 40.0 / 30.4 bp at 0 / 1 / 5 bp per trade |
| Average VIX vs realised vol over the sample | 19.1% vs 15.5% |
| Surface | 9,280 → 2,688 clean OTM quotes, 23 expiries; ATM 15.5% (1M) / 17.6% (1Y); 90/110 skew 13.7 / 5.9 vol pts; SVI out-of-sample RMSE 0.13 vol pts |
| Monte Carlo | antithetic + control variate cut variance 23× (control alone 5.9×, antithetic 1.8×) |
| CRR | error O(1/n): −0.020 at 100 steps, −0.002 at 1,000; Hull American put 4.49 (5 steps) → 4.283 (500 steps) |

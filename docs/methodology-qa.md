# Methodology Q&A: the 10 hardest questions

Short answers to the questions an interviewer or examiner is most likely to push on. Numbers refer to the committed results: chain snapshot 2026-10-23 14:03 UTC, hedging sample of sale dates 2005-01-03 to 2026-09-23.

---

**1. "Your seller made 40 bp a month with t = 10. Is that a volatility risk premium?"**

No, and the write-up says so. The option is priced at the VIX, which is a variance-swap rate. It includes the put skew, so it sits above the ATM implied vol at which a 1-month ATM call actually trades.
- **Break-even:** the mean P&L is gone if that spread is 3.5 vol points (40 bp ÷ vega of about 11.5 bp per vol point).
- **Sensitivity:** re-running with assumed spreads of 0, 1, 2, 3 and 4 points gives 40, 28, 17, 5 and −6 bp. The t-stat falls to 1.4 at a 3-point spread.
- **Snapshot day:** VIX was about 19.0 and the SPY 1-month ATM vol 15.5%, a gap of about 3.5 points.
- **Option bid–ask:** not charged at all.
- **Tail:** the worst window lost 521 bp, about 13 months of average profit, in Oct 2008.

The defensible results are the mechanics: the gamma–theta attribution, the error scaling, and what rebalancing can and cannot fix. A clean premium estimate needs a history of true ATM IVs, which belongs to my separate volatility-risk-premium project.

**2. "Derive the P&L ≈ ½Γ S²(σ² − σ_r²)dt formula."**

1. Hold −V and +Δ shares, and keep V − ΔS in cash at r. Over dt: dΠ = −dV + Δ(dS + qS dt) + r(V − ΔS)dt.
2. Itô/Taylor expansion: dV = Θdt + ΔdS + ½ΓdS². Substitute it, and the ΔdS terms cancel.
3. The BSM PDE gives −Θ = (r − q)SΔ + ½σ²S²Γ − rV. Substitute it, and all the r and q terms cancel: dΠ = ½ΓS²(σ²dt − (dS/S)²).

With (dS/S)² ≈ σ_r²dt this is ½ΓS²(σ_h² − σ_r²)dt for the short position, the brief's formula with the sign flipped.

In the data this attribution accounts for 91% of the variance of window P&L, with slope 0.968 (SE 0.023). That is a check that the second-order expansion is complete, not an independent test of Black–Scholes: the regressor uses the same realised returns as the P&L.

**3. "Why doesn't daily hedging cut your risk by √5 relative to weekly?"**

Because discretisation noise is only part of the error. When the hedge vol σ_h differs from realised vol, the P&L contains Σ½ΓS²(σ_h² − σ_r²)Δt. That term depends on the path through Γ (it matters whether the big moves happen near the strike), and rebalancing more often does not remove it.
- **Oracle hedge (realised vol, look-ahead):** sd(ε) scales with slope 0.49 in log h, against 0.48 for the theory curve with N = ⌈21/h⌉, and weekly error is 2.4× daily.
- **Tradable hedge vols:** the slope is 0.27–0.29 and the weekly/daily ratio only about 1.45.

More frequent hedging removes the noise, not the bet on vol.

**4. "Does Derman–Kamal actually fit?"**

The scaling fits; the constant is a little low. Standardising each window's error by its own vega·σ/√N gives an sd of 0.81, 0.79, 0.79, 0.85 and 0.75 for h = 1 to 21. The values are flat in h, as √Δt requires, and about 10% below the theoretical √(π/4) = 0.886. Possible reasons:
- real returns are not GBM;
- vol changes within the month;
- the dividend-model error is mixed into ε.

Plugging *median* vega and σ into the formula gives 28.4 bp against the observed 28.8 bp. That is a coincidence: the formula is nonlinear, the proper aggregate (the RMS over windows) is 42 bp, and the per-window test is the right one.

**5. "Why do you have a look-ahead 'oracle' variant? Isn't that cheating?"**

It is not a strategy. It is an experiment that isolates one source of error. It is labelled "look-ahead" in every table and chart, and its P&L is never presented as achievable. Variants A and B use only information available at the sale-day close:
- **A:** VIX at t0.
- **B:** trailing 21-day realised vol ending at t0.

A unit test perturbs a future price and checks that the variant B hedge vol does not change. The study also pins its end date, so no data after 2026-10-22 enters.

**6. "Your surface uses BSM, but SPY options are American. Isn't that wrong?"**

It is an approximation, and I measured its size:
- **Only OTM options are used:** puts below the forward and calls above it. That is the liquid side and has the smallest early-exercise value.
- **The error is measured.** IVs backed out of a 200-step American CRR tree differ from BSM by at most 0.17 vol points at 1 month. At 1 year they are up to 1.3 vol points lower for puts struck between spot and the forward: those puts are ITM against spot, and with r = 4.4% > q = 1% early exercise has value.
- **The forward check agrees.** Parity-implied forwards fall 33–106 bp below the model forward at the longest expiries, the direction American put premia push C − P.

A desk would de-Americanise every quote or use European-style SPX options.

**7. "Why is the skew negative, and why does it flatten with maturity?"**

- **Why negative:**
  - *Demand.* Long-only investors buy OTM index puts for protection, and dealers who sell them are left short downside gamma and charge for it.
  - *Physical dynamics.* Index vol rises when prices fall (the leverage effect and spot–vol correlation), and crashes are jumps. Both fatten the left tail.
- **Why it flattens:** over longer horizons returns aggregate and look more normal, and a fixed 10% move is fewer standard deviations when σ√T is larger. 1-month 90/110 skew is 13.7 vol points against 5.9 at 1 year, and the ATM slope dσ/dk falls from about −0.94 at 1 month to −0.20 at 2.2 years.

**8. "How do you know your SVI fit isn't overfitting? It has 5 parameters."**

Partly. Holding out every 5th strike gives a median RMSE of 0.13 vol points out of sample, the same as in sample, and the surface has 0 calendar-arbitrage violations on 1,122 grid points. But I'm explicit about what that does and doesn't show:
- **It is an interpolation test.** With $1 strike spacing the held-out strikes sit between close neighbours, so it shows the curve is smooth, not that the parameters are meaningful.
- **The median hides the worst slice:** the 7-day slice has a 0.52 vol-point error.
- **Some parameters are weakly identified:** several short-dated slices have ρ > 0 with a large m despite a steeply negative skew.
- **The wings breach Lee's bound** beyond the data on 3 slices.

A real test would hold out a whole wing or a whole expiry. No reported number relies on the wings outside the quoted strikes.

**9. "Your windows overlap. Are your p-values honest?"**

Consecutive windows share up to 20 of their 21 daily returns. I use Hansen–Hodrick standard errors (a uniform kernel at lag 20), the textbook estimator for overlapping returns. Newey–West's Bartlett kernel at lag 20 down-weights exactly those lags and recovers only about 2/3 of the long-run variance; my first version used it and overstated t-stats by about 20%. A unit test with simulated overlapping sums checks both estimators. As a lag-free check I also run plain t-tests on 261 non-overlapping windows. Applying both to the borderline results:
- **Headline mean:** t = 10.3 HH and 8.9 non-overlapping. It is significant, but see Q1 for what it means.
- **Less frequent hedging earning more:** HH p = 0.04, but the non-overlapping p-values across the 21 possible offsets have a median of 0.21. **Not a finding.**
- **Descriptive only:** Jarque–Bera and the 5-point scaling-slope SE both assume independence.

**10. "How do you know the pricing code is right?"**

There are 192 pytest checks, run on every push by GitHub Actions:
- **Textbook values:** Hull's examples (call 4.76 and put 0.81; the index option 51.83; the Greeks Δ 0.522, Γ 0.066, vega 12.1, Θ −4.31, ρ 8.91; the 23.5% implied vol; the American put at 4.49, 4.263, 4.278 and 4.283 for 5, 30, 100 and 500 steps).
- **Calculus:** every Greek, for calls and puts with q > 0, against central finite differences.
- **Identities:** put–call parity to 10⁻¹⁰; American ≥ European; an American call with no dividend equals the European call.
- **Convergence:** CRR within 0.01 of BSM at 1,000 steps.
- **Monte Carlo:** within 4 standard errors of BSM, with SEs shrinking like 1/√n.
- **Implied vol:** round-trips across a grid, including the Brent fallback.
- **Hedging:** a hand-computed 2-step hedge, with dividends and costs, matches to 10⁻¹²; the decomposition sums exactly; the HAC estimators match hand-computed examples.

An independent review also stress-tested the IV solver on 840 cases (maximum error 4×10⁻⁹ wherever vega is not negligible).

---

## Key numbers to remember

| | |
|---|---|
| Windows / period | 5,465 overlapping 21-day windows, sale dates 2005-01-03 to 2026-09-23 (≈261 independent months) |
| Mean hedged P&L, daily, priced and hedged at VIX | **+40.1 bp of spot** (18.4% of a 220 bp premium), HH t = 10.3; 81% of windows positive |
| …but break-even VIX − ATM spread | **3.5 vol pts** (snapshot-day spread ≈ 3.5); at 3 pts mean = 5 bp, t = 1.4 |
| Decomposition (daily, VIX) | theta +123.5, gamma −82.1, costs −2.4, residual +1.2 bp |
| Residual | sd 20 bp; ex-date windows +20.4 vs −8.5 bp otherwise (dividend model) |
| Worst window | sold 2008-10-08: −521 bp (−80% of premium); VIX 57.5, realised 90% |
| Gamma–theta attribution | slope 0.968 (SE 0.023), R² 0.91 |
| Error scaling | oracle slope 0.49 (theory curve 0.48); standardised sd 0.75–0.85 vs √(π/4) = 0.886; tradable hedges 0.27–0.29 |
| Daily vs weekly P&L sd (VIX hedge) | 66 vs 90 bp; error ratio 1.47 (oracle 2.43) |
| Average VIX vs realised vol over the sample | 19.1% vs 15.5% |
| Surface | 9,280 → 2,688 clean OTM quotes, 23 expiries; ATM 15.5% (1M) / 17.6% (1Y); 90/110 skew 13.7 / 5.9 vol pts; SVI held-out RMSE 0.13 vol pts |
| Monte Carlo | antithetic + control variate cut variance 23× (control alone 5.9×, antithetic 1.8×) |
| CRR | error O(1/n): −0.020 at 100 steps, −0.002 at 1,000; Hull American put 4.49 (5 steps) → 4.283 (500) |

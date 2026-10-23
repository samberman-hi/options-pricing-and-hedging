# Option pricing engine, SPY implied-volatility surface and delta-hedging study

**Sam Berman, University College Dublin**

[![tests](https://github.com/samberman-hi/options-pricing-and-hedging/actions/workflows/tests.yml/badge.svg)](https://github.com/samberman-hi/options-pricing-and-hedging/actions/workflows/tests.yml)

**Headline result.** I sold a 1-month at-the-money SPY call on each of 5,465 trading days from 2005 to 2026 and delta-hedged it to expiry. Two findings carry the study:
- **Hedging at the true volatility works as theory predicts.** Hedging at the realised vol (an ex-post benchmark) makes the error grow like √Δt: log–log slope 0.49, against 0.48 for the theory curve. Measured window by window, the error is a stable 0.75–0.85 × vega·σ/√N at every frequency, against the Derman–Kamal √(π/4) = 0.89.
- **With a hedge vol you can actually trade, frequent rebalancing helps less.** Using implied or trailing realised vol, the slope falls to 0.27–0.29, and daily hedging cuts the error only 1.5× relative to weekly, against roughly 2.2× when the hedge vol is right. A mis-specified vol leaves a path-dependent Γ-weighted term that rebalancing cannot remove.

A second-order gamma–theta attribution, Σ ½ΓS²(σ²Δt − R²), accounts for 91% of the variance of window P&L. Most of what is left comes from modelling dividends as a flat yield.

Pricing the call at the VIX, the hedged seller earned **+40 bp of spot per month**, with a Hansen–Hodrick t of 10.3. That mean is **not cleanly identified**:
- **Break-even.** It disappears if VIX overstates the true 1-month ATM implied vol by 3.5 vol points. On the day of the option snapshot, the gap between VIX and the SPY ATM vol was about 3.5 points.
- **Tail risk.** The worst window, sold 8 Oct 2008, lost 521 bp, 80% of the premium.

The pricing library (Black–Scholes–Merton, CRR trees, Monte Carlo) reproduces Hull's textbook values. The SPY surface from 23 Oct 2026 shows a 1-month 90%/110% skew of 13.7 vol points, falling to 5.9 at one year.

---

## 1. Questions

1. **Pricing.** Can a small, tested library price European and American options consistently three ways: closed form, binomial tree and Monte Carlo? How fast do the numerical methods converge, and how much is the right to exercise an American put early worth?
2. **Implied volatility.** Once noisy quotes are cleaned out, what do the SPY smile and surface look like today, and why are they shaped that way?
3. **Hedging.** If you sell a 1-month ATM SPY call and delta-hedge it, what P&L do you actually get, and why? How much comes from theta, from gamma and from trading costs? How does the hedging error change with rebalancing frequency and with the volatility used to compute the hedge?

## 2. Data

All data is free and downloaded by [`src/data.py`](src/data.py), cached in `data/raw/` (gitignored). No values are simulated or filled in by hand; if a source fails the code raises an error.

| Data | Source | Identifier | Period used |
|---|---|---|---|
| SPY daily close (not dividend-adjusted) and cash dividends | Yahoo Finance via `yfinance` | `SPY` | 2004-01-02 to 2026-10-22 (pinned in `hedging.END`) |
| SPY option chain (all 28 listed expiries, calls and puts, 9,280 contracts) | Yahoo Finance via `yfinance` | `SPY` | **Snapshot: 2026-10-23 14:03:52 UTC (10:03 New York time), spot $760.88** |
| Risk-free rate, hedging study | FRED | `DGS1MO` (1-month T-bill) | 2005 to 2026 |
| Risk-free curve, surface | FRED | `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2` | latest print (21 Oct 2026, FRED publishes with a lag) |
| Implied vol at sale, hedging study | FRED (fallback: Yahoo `^VIX`) | `VIXCLS` | 2005 to 2026 |

Yahoo Finance and the FRED CSV endpoint both worked, so the Stooq fallback was not needed. Stooq only offers dividend-adjusted prices without the dividends themselves, so `data.py` refuses to use it silently for the hedging study.

**Refreshing the snapshot.** `python -m src.pipeline --refresh-chain` takes a new snapshot and rebuilds every surface figure and table. The snapshot timestamp is stored in `data/raw/spy_chain_meta.json` and `results/surface_summary.json`. Every surface number in this README refers to the snapshot above. As a stability check I re-ran the whole pipeline from a clean clone 18 minutes later, which took a new snapshot (spot $759.52, 2,704 clean quotes). The headline numbers barely moved: 1-month ATM vol went from 15.5% to 15.5%, 1-month 90/110 skew from 13.7 to 13.2 vol points, and median held-out SVI RMSE from 0.13 to 0.12. The hedging-study numbers reproduced exactly.

## 3. Method

### 3.1 Pricing library ([`src/bsm.py`](src/bsm.py), [`src/binomial.py`](src/binomial.py), [`src/montecarlo.py`](src/montecarlo.py))

**Black–Scholes–Merton with a continuous dividend yield $q$:**

$$C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2),\qquad P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1),$$
$$d_{1} = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad d_2 = d_1 - \sigma\sqrt T .$$

Greeks are computed in closed form. Delta is $e^{-qT}N(d_1)$ for a call. Gamma is $e^{-qT}n(d_1)/(S\sigma\sqrt T)$. Vega is $Se^{-qT}n(d_1)\sqrt T$, per 1.00 of vol. Theta is per year. Rho is $KTe^{-rT}N(d_2)$ for a call, per 1.00 of rate. Each Greek is checked against central finite differences.

**Put–call parity** for European options says $C - P = Se^{-qT} - Ke^{-rT}$. It is tested to $10^{-10}$ on the model. On market data it is used as a diagnostic (§4.2).

**Cox–Ross–Rubinstein tree.** The tree uses $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and $p = (e^{(r-q)\Delta t} - d)/(u-d)$, and rolls back one array per time step. The American version takes $\max(\text{continuation}, \text{exercise})$ at every node. The early-exercise premium is the American price minus the European price on the same tree.

**Monte Carlo.** Terminal prices are sampled exactly from GBM, $S_T = S e^{(r-q-\sigma^2/2)T + \sigma\sqrt T Z}$. Three variance-reduction schemes are compared with the plain estimator at the same number of payoff evaluations:
- **Antithetic** variates pair $Z$ with $-Z$. The standard error is computed from the n/2 pair averages, because the two halves of a pair are not independent.
- A **control variate** uses the discounted terminal stock price, whose mean $Se^{-qT}$ is known: $\hat V = \bar Y - b^*(\bar X - Se^{-qT})$ with $b^* = \widehat{\mathrm{Cov}}(Y,X)/\widehat{\mathrm{Var}}(X)$.
- **Both** combined.

### 3.2 Implied volatility and the surface ([`src/implied_vol.py`](src/implied_vol.py), [`src/surface.py`](src/surface.py))

**Solver.** Newton–Raphson on vega, $\sigma_{n+1} = \sigma_n - (V(\sigma_n) - V^{mkt})/\mathcal V(\sigma_n)$:
- **Starting point.** Away from the money it starts from the Manaster–Koehler point $\sigma_0 = \sqrt{2|\ln(F/K)|/T}$, the inflection point of price in σ, from which Newton converges monotonically. Near the money it uses the Brenner–Subrahmanyam ATM approximation.
- **Acceptance.** A solution is accepted only if both the price error and the implied step in σ are tiny, because when vega is near zero a small price error can hide a large σ error.
- **Fallback.** If Newton leaves the bracket [0.0001, 5] or vega vanishes, the solver falls back to Brent's method, which always converges because price rises strictly with σ.
- **No solution.** Prices outside the no-arbitrage bounds return NaN.

**Inputs for each option.** (The dividend yield is the same estimator used in the hedging study.)
- **Time to expiry:** $T$ = time from the snapshot to 16:00 New York on the expiry date, in calendar years.
- **Rate:** $r(T)$ is linearly interpolated from the FRED Treasury curve, converted with $r = \ln(1 + y/100)$.
- **Dividend yield:** $q = \ln(1 + D/S)$, where $D$ is the sum of the last four SPY cash dividends and $S$ is spot, 0.99%.
- **Forward:** $F = Se^{(r-q)T}$, and moneyness is $k = \ln(K/F)$.

**Cleaning.** Each step below is counted in `results/chain_cleaning.csv`:
- drop quotes with zero bid or a crossed market;
- drop stale quotes (no trade in the last 3 business days);
- drop quotes whose spread is more than 50% of the mid;
- drop expiries under 7 days;
- keep $-0.4 \le k \le 0.25$;
- keep **out-of-the-money options only** (puts with K < F, calls with K ≥ F);
- price at the bid–ask mid.

SPY options are American. OTM options are the liquid side of the market and carry little early-exercise value, so pricing them with BSM is the standard approximation. Its size is measured in §4.2.

**Smile model.** Each expiry is fitted with raw SVI on total implied variance $w(k) = \sigma^2 T$:
$$w(k) = a + b\left(\rho(k-m) + \sqrt{(k-m)^2 + s^2}\right).$$

The fit is tested out of sample: every 5th strike is held out, the curve is fitted on the rest, and the RMSE is reported on both sets. The final curves are then refitted on all strikes. The 3D surface interpolates total variance linearly in $T$ at fixed $k$. Calendar arbitrage (total variance falling with maturity) is checked on a grid.

### 3.3 Delta-hedging study ([`src/hedging.py`](src/hedging.py), [`src/stats.py`](src/stats.py))

**Trade.** On every trading day $t_0$ from 2005-01-03 that has 21 trading days of data after it (5,465 windows), sell one European call:
- **Strike and expiry:** $K = S_{t_0}$ (the close), expiring after 21 trading days.
- **Price:** BSM with $\sigma_{imp}$ = VIX/100, $r$ = the 1-month T-bill rate and $q$ = the trailing dividend yield, all known at $t_0$.
- **Hedge:** hold Δ SPY shares, trading at the close every $h \in \{1, 2, 5, 10, 21\}$ trading days. Here 1 is daily, 5 is weekly and 21 means only the initial hedge.

**Bookkeeping.**
- **Cash:** earns $r$, and the share position receives the actual cash dividends on ex-dates.
- **Transaction costs:** 1 bp of traded notional on every share trade, including the unwind at expiry, with 0 and 5 bp as sensitivities.
- **Settlement:** the call is cash-settled.
- **Units:** P&L is reported in bp of $S_{t_0}$.

**Volatility used to compute delta.**
- **A, implied:** $\sigma_h = \sigma_{imp}$, the VIX at sale.
- **B, forecast:** $\sigma_h$ = trailing 21-day realised vol at $t_0$, known at sale.
- **C, oracle:** $\sigma_h$ = the realised vol of the window itself. *This uses future data and is only a theoretical benchmark, not a tradable strategy.*

**Theory and decomposition.** Short the option, hold Δ shares and keep $V - \Delta S$ in cash. Over a step, $d\Pi = -dV + \Delta(dS + qS\,dt) + r(V - \Delta S)dt$. A Taylor expansion of $dV$ and the BSM PDE $\Theta + (r-q)S\Delta + \tfrac12\sigma^2S^2\Gamma - rV = 0$ cancel every term except

$$d\Pi \approx \tfrac12 \Gamma S^2\left(\sigma_h^2\,\Delta t - R^2\right),\qquad R = \Delta S/S .$$

Summing over the life of the option gives the textbook result, here with the sign flipped because we are short gamma:

$$\text{P\&L}_{short} \approx \underbrace{V(\sigma_{imp}) - V(\sigma_h)}_{\text{vol edge}} + \sum_i \tfrac12\Gamma_i S_i^2\left(\sigma_h^2 - \sigma_{realised,i}^2\right)\Delta t .$$

Each window's P&L is split into five parts: the vol edge, Σ theta ($+\tfrac12\Gamma S^2\sigma_h^2\Delta t$), Σ gamma ($-\tfrac12\Gamma S^2R^2$), −transaction costs, and a residual. The residual collects discrete rebalancing, higher-order terms, discrete dividends and interest on the P&L.

**Hedging error.** The hedging error is defined as $\varepsilon = \text{P\&L} - [V(\sigma_{imp}) - V(\sigma_{real})]$. It measures how far the result lands from the P&L Black–Scholes says you lock in if you know the realised vol. For a hedge at the true vol rebalanced $N$ times, Boyle–Emanuel and Derman–Kamal give $\mathrm{sd}(\varepsilon) \approx \sqrt{\pi/4}\,\mathcal V\sigma/\sqrt N$, so the error should grow like $\sqrt{\Delta t}$, a slope of 0.5 on a log–log plot.

**Statistics.**
- The windows overlap by up to 20 days, so means are tested with **Hansen–Hodrick standard errors (HAC with a uniform kernel, lag 20)**. This is the standard estimator for overlapping returns. Newey–West's Bartlett weights would capture only about two-thirds of the long-run variance at lag 20 and overstate t-stats by about 20%; a unit test checks both.
- As a robustness check, I also run a plain t-test on the 261 non-overlapping windows.
- The regression of P&L on the theoretical P&L uses the same Hansen–Hodrick standard errors.
- Normality is tested with Jarque–Bera.

## 4. Results

### 4.1 Pricing library

| Check | This library | Hull |
|---|---|---|
| European call / put (S=42, K=40, r=10%, σ=20%, T=0.5), Ex. 15.6 | 4.7594 / 0.8086 | 4.76 / 0.81 |
| Index call with dividends (S=930, K=900, r=8%, q=3%, σ=20%, T=2/12) | 51.833 | 51.83 |
| Greeks (S=49, K=50, r=5%, σ=20%, T=0.3846): Δ, Γ, vega, Θ/yr, ρ | 0.522, 0.0655, 12.11, −4.31, 8.91 | 0.522, 0.066, 12.1, −4.31, 8.91 |
| Implied vol (c=1.875, S=21, K=20, r=10%, T=0.25) | 23.45% | 23.5% |
| American put CRR (S=K=50, r=10%, σ=40%, T=5/12), 5 / 30 / 100 / 500 steps | 4.488 / 4.263 / 4.278 / 4.283 | 4.49 / 4.263 / 4.278 / 4.283 |

**Convergence and early exercise** (S = K = 100, T = 1, r = 4%, q = 1%, σ = 20%):
- **CRR error:** the European put error is −0.0196 at 100 steps and −0.0020 at 1,000, so the error is O(1/n).
- **Oscillation:** the error alternates in sign between odd and even n. Averaging the n- and (n+1)-step prices cuts the error at n = 1,000 to about 5×10⁻⁵.
- **Early-exercise premium:** the American put is worth 6.701 against a European 6.394, a premium of 0.31. The premium rises with the interest rate and with how far in the money the put is (Figure 2). It is exactly zero for a call on a non-dividend-paying stock, as theory requires.

| Monte Carlo, 100,000 payoffs, ATM call (BSM = 9.3197) | Estimate | Std. error | Error / SE | Variance reduction |
|---|---|---|---|---|
| Plain | 9.2882 | 0.0441 | 0.72 | 1.0× |
| Antithetic | 9.2891 | 0.0329 | 0.93 | 1.8× |
| Control variate (discounted S_T) | 9.3080 | 0.0181 | 0.64 | 5.9× |
| Antithetic + control | 9.3108 | 0.0092 | 0.97 | **23.2×** |

![CRR convergence](figures/01_crr_convergence.png)
![American put premium](figures/02_american_put_premium.png)
![Monte Carlo convergence](figures/03_mc_convergence.png)

### 4.2 SPY implied-volatility surface (snapshot 2026-10-23 14:03 UTC, spot $760.88)

**Cleaning:** 9,280 raw contracts left 2,688 clean OTM quotes across 23 expiries, from 7 days to 2.2 years. The stale-quote filter removes the most rows (2,799), mostly in-the-money and far-from-the-money contracts that rarely trade. Newton converged on all 2,688 options, Brent was never needed, and none failed.

| Summary | Value |
|---|---|
| ATM implied vol, 1 month / 1 year | 15.5% / 17.6% |
| Skew, IV(90% K/F) − IV(110% K/F), 1 month / 1 year | 13.7 / 5.9 vol pts |
| SVI RMSE, median across expiries, in-sample / **held-out strikes** | 0.13 / **0.13** vol pts |
| Calendar-arbitrage violations on the fitted surface | 0 of 1,122 grid points |
| Put–call parity forward vs model forward, expiries ≤ 1.2y | within ±16 bp (median −4 bp) |
| American-tree IV − BSM IV, all OTM options in the 1M and 1Y expiries | median 0.04, max 1.27 vol pts |

**What the skew means.** Out-of-the-money puts are much more expensive than out-of-the-money calls. On the 1-month expiry, a put struck 10% below the forward trades at about 14 vol points more than a call struck 10% above. There are two reasons:
- **Demand.** Investors hold long equity portfolios and pay up for crash protection. Dealers who sell them that protection are left short downside gamma and charge for it.
- **Leverage.** Index volatility tends to jump when the market falls, so the market-implied return distribution has a fatter left tail than the lognormal Black–Scholes assumes.

The skew is steepest at short maturities and flattens as maturity grows, falling from 13.7 vol points at 1 month to 5.9 at 1 year and 4.0 at 2.2 years. The ATM slope dσ/dk falls roughly like 1/√T. A given percentage crash is a much bigger move relative to one month of expected volatility than to two years of it. Short-dated smiles also curl back up slightly in the far call wing.

**What the term structure means.** At-the-money vol is slightly inverted over the first few weeks, 16.6% at 7 days against 15.0% at about 5 weeks. That reflects near-term event risk being priced. Beyond about two months it slopes upward, reaching 17.6% at 1 year and 18.9% at 2.2 years. That is the normal shape in a calm market: today's vol (VIX ≈ 19 at the last close, ATM 1-month ≈ 15.5%) sits below the long-run average, so investors expect vol to drift back up over time, and long-dated options also carry a premium for uncertainty about volatility itself.

**Diagnostics (reported, not hidden).**
1. **Parity.** The forward implied by put–call parity, $F = K + e^{rT}(C - P)$, agrees with the model forward to within ±16 bp out to 1.2 years. It sits 33–106 bp below the model forward at the three longest expiries. In those expiries early exercise of the American puts raises P, the dividend stream is not flat, and there are few strike pairs.
2. **Early exercise.** A 200-step American tree recovers almost the same IV as BSM for 1-month options (≤ 0.17 vol pts). For the 1-year expiry it is 0.1–1.3 vol pts lower for puts struck between spot and the forward. Those puts are OTM against the forward but ITM against spot, and with r ≈ 4.4% > q ≈ 1% their early-exercise right is worth something. The BSM surface therefore slightly overstates long-dated put vols near the money.
3. **SVI fit.** Some long-dated SVI slices sit at a bound for ρ or have unstable parameters. The fitted curve is accurate where there is data, but the wings beyond the quoted range should not be trusted (3 slices breach Lee's wing-slope bound of 2).

![IV smiles](figures/04_iv_smiles.png)
![IV surface](figures/05_iv_surface_3d.png)
![ATM term structure and skew](figures/06_atm_term_structure_skew.png)

### 4.3 Delta-hedging a short 1-month ATM SPY call (5,465 windows, 2005-01-03 to 2026-09-23 sale dates, 1 bp costs)

P&L figures are in bp of spot at sale. The average premium is 220 bp. t-stats are Hansen–Hodrick, lag 20; the non-overlapping t-test uses 261 windows.

| Hedge vol | h (days) | Mean P&L | Std | % of premium | t (HH) | Non-overlap t | sd(ε) |
|---|---|---|---|---|---|---|---|
| A: implied (VIX) | 1 (daily) | **40.1** | 66.2 | 18.4% | 10.3 | 8.9 | 54.5 |
| A: implied (VIX) | 5 (weekly) | 45.5 | 89.6 | 20.1% | 10.0 | 9.3 | 80.0 |
| A: implied (VIX) | 21 (initial only) | 50.1 | 147.1 | 22.3% | 7.0 | 6.1 | 131.6 |
| B: trailing 21d RV | 1 | 38.5 | 68.5 | 17.7% | 9.9 | 7.6 | 58.1 |
| B: trailing 21d RV | 5 | 44.3 | 91.4 | 19.5% | 9.8 | 8.8 | 82.1 |
| C: oracle RV (look-ahead) | 1 | 37.0 | 83.0 | 17.0% | 7.1 | 7.4 | **28.8** |
| C: oracle RV (look-ahead) | 5 | 44.2 | 99.4 | 19.9% | 8.3 | 9.1 | 69.9 |

All means are significantly positive, with every p below 10⁻⁷. The full table is in `results/hedging_summary.csv`.
- **Win rate:** 81% of windows make money.
- **Shape:** skewness ranges from −0.5 to −3.1 and excess kurtosis from 7.6 to 21.6. Jarque–Bera rejects normality everywhere, but treat it as descriptive, since it assumes independent windows.

**1. Where the mean comes from, and why it is not a clean premium estimate.** The decomposition (variant A, daily) is **theta +123.5 bp, gamma −82.1 bp, costs −2.4 bp, residual +1.2 bp, total 40.1 bp**. The seller collects time decay priced at VIX and pays out convexity at realised vol, and over this sample VIX averaged 19.1% against realised 1-month vol of 15.5%. But VIX is a variance-swap rate that embeds the put skew, so it sits above the ATM implied vol at which a 1-month ATM call actually trades.

**The mean depends on that spread.** Re-running the study with the option sold and hedged at VIX minus an assumed spread gives:

| Assumed VIX − ATM IV | 0 | 1 | 2 | 3 | 4 vol pts |
|---|---|---|---|---|---|
| Mean P&L (bp) | 40.1 | 28.5 | 16.9 | 5.3 | −6.4 |
| t (HH) | 10.3 | 7.4 | 4.5 | 1.4 | −1.7 |

The break-even spread is **3.5 vol points**. On the snapshot day the spread was about that size: VIX closed at 19.0 the evening before, and the SPY 1-month ATM vol was 15.5%. One day is not a history, but it means **this study cannot establish that delta-hedged call selling earned a premium**. Measuring that needs a history of true ATM implied vols, which is my separate volatility-risk-premium project. The robust results here are the mechanics below.

**2. The gamma–theta attribution.** Regressing each window's P&L on its $\sum \tfrac12\Gamma S^2(\sigma_h^2\Delta t - R^2)$ gives:
- slope **0.968**, HH SE 0.023;
- a test of slope = 1 gives p = 0.17, so slope = 1 is not rejected;
- intercept 0.08 bp, **R² = 0.91**.

This checks that the second-order Taylor attribution is complete; it is not an independent test of Black–Scholes, because the regressor uses the same realised returns as the P&L. The residual has an sd of 20 bp, and its largest single source is the dividend model. Pricing uses a flat yield $q$, but the hedge receives actual discrete dividends, so the 33% of windows that contain an ex-date have a mean residual of +20.4 bp against −8.5 bp for the rest. An ex-date dummy explains 46% of the residual variance. The remainder is third-order terms on crash days and interest on the accumulated P&L.

**3. Hedging error and frequency** (Figure 9).
- **Oracle hedge.** Hedging at the vol that is actually realised removes the average vol mismatch, leaving mostly discretisation noise. sd(ε) then scales with slope **0.49 ± 0.02** in log h, matching the theoretical curve's 0.48 (the theory slope is not exactly 0.5 because N = ⌈21/h⌉). Weekly error is 2.4× daily.
- **Per-window check.** Standardising each window's error by its own vega·σ/√N gives an sd of **0.81, 0.79, 0.79, 0.85, 0.75** for h = 1, 2, 5, 10, 21. That is flat in h, as the √Δt law requires, and about 10% below Derman–Kamal's √(π/4) = 0.886.
- **Why medians are misleading.** Plugging median vega and σ into the formula gives 28.4 bp, apparently matching the oracle's 28.8 bp at h = 1. That match is a coincidence. The right aggregate is the RMS over windows, which gives 42 bp, because the formula is nonlinear and high-vol windows dominate.
- **Tradable hedges.** With implied or trailing-realised vol the slope is only **0.29 / 0.27**, and moving from weekly to daily cuts the error just 1.4–1.5×. With σ_h ≠ σ_realised the P&L contains $\sum\tfrac12\Gamma S^2(\sigma_h^2 - \sigma_{real}^2)\Delta t$, which depends on the path through Γ and does not shrink with rebalancing. **Rebalancing more often removes the discretisation noise, not the vol bet.**
- **Daily vs weekly.** At implied vol, weekly hedging raises the P&L sd from 66 to 90 bp. Daily hedging costs only 2.4 bp at 1 bp per trade.

**4. Mean P&L vs frequency is not significant.** Mean P&L rises from 40 bp with daily hedging to 50 bp with only the initial hedge. The paired difference is +10.0 bp, with an HH t of 2.07 (p = 0.04). But running the non-overlapping t-test from each of the 21 possible starting offsets gives a median p of 0.21 (range 0.05–0.76), so **I don't treat it as a finding**. It is consistent with short-horizon mean reversion in SPY over the sample (daily-return variance ratios VR(5) = 0.82, VR(21) = 0.74), which an under-hedged short-gamma book would benefit from, but those ratios are descriptive, not tested.

**5. Transaction costs.** Daily hedging pays about 2.4 bp of spot per 1 bp of cost rate. At 5 bp, the VIX-priced daily mean falls to 30.5 bp.

**6. Tail risk.** The worst windows are all crash episodes:

| Sold on | P&L | VIX at sale | Realised vol in window |
|---|---|---|---|
| 2008-10-08 | −521 bp (−80% of premium) | 57.5 | 90.4% |
| 2008-10-07 | −467 bp | 53.7 | 88.7% |
| 2020-02-28 | −442 bp | 40.1 | 90.8% |
| 2020-02-27 | −414 bp | 39.2 | 90.1% |
| 2020-03-06 | −393 bp | 41.9 | 91.8% |

![P&L distribution](figures/07_hedge_pnl_distribution.png)
![P&L decomposition](figures/08_pnl_decomposition.png)
![Hedging error scaling](figures/09_hedging_error_scaling.png)
![P&L vs theory](figures/10_pnl_vs_theory.png)

## 5. Assumptions and limitations

**Pricing**
- Constant σ, r and q within each pricing problem; GBM dynamics in the Monte Carlo and the tree.
- Dividends are a continuous yield, not discrete cash amounts.

**Surface**
- *Snapshot.* One intraday snapshot, with Yahoo quotes delayed up to about 15 minutes. Spot was recorded at download time, so the spot and option timestamps are not perfectly synchronised; 1 bp of spot is about 8 cents.
- *Rates.* FRED yields are constant-maturity Treasury yields used as the risk-free rate (not OIS/SOFR), with a 1–2 day publishing lag.
- *Dividend yield.* q is a single trailing yield. SPY pays quarterly, so the true forward has steps. The parity diagnostic shows the effect is a few bp out to about 1 year.
- *American exercise.* IVs use the European BSM model on American options. This is measured to be negligible at 1 month but worth up to 1.3 vol points for some 1-year puts (§4.2).
- *SVI.* No arbitrage-free constraints are imposed beyond the parameter bounds; calendar arbitrage is checked rather than enforced, and the SVI wings are unreliable beyond the quoted strikes.
- *SVI hold-out.* The test holds out every 5th strike, so it is an *interpolation* test between close neighbours. It shows the fit is smooth, not that the wings extrapolate. The median RMSE hides a 0.52 vol-point error on the 7-day slice, and some short-dated slices have weakly identified parameters (ρ > 0, large m).

**Hedging study**
- **VIX stands in for the 1-month ATM implied vol.** A free daily history of SPY ATM implied vol does not exist. VIX is a 30-calendar-day variance-swap rate that includes the skew, so it sits above ATM IV, by about 3.5 points on the snapshot day. This makes the mean P&L unidentified (§4.3, point 1). It barely affects the mechanics: lowering the hedge vol by 2–3.5 points moves the variant-A scaling slope only from 0.285 to 0.27–0.28.
- **Dividends.** Options are priced with a flat dividend yield, while the hedge receives discrete dividends. This is the largest part of the attribution residual. A better model would price with the known in-window dividends (escrowed-dividend spot).
- *Execution.* The option is assumed to be sold at the VIX-implied mid with no option bid–ask cost, and the stock is traded at the close with no market impact. Costs are a flat 1 bp of notional.
- *Contract.* The call is treated as European and cash-settled; the real SPY option would be American and physically settled. There is no early-exercise risk for a short call except around ex-dividend dates, which is not modelled.
- *Time.* Time is measured in trading days, with 252 per year. VIX is quoted on a calendar basis, but 21/252 and 30/365 differ by under 2% in total variance.
- *Overlap.* The 5,465 windows overlap heavily, so the effective sample is about 261 independent months. That is why Hansen–Hodrick and non-overlapping tests are reported. The Jarque–Bera test and the scaling-slope standard error assume independence and are descriptive only.
- *Look-ahead.* Variant C uses future information and exists only as a theoretical benchmark. Variants A and B use only information available at the close on the sale date.

## 6. Reproduce in 3 commands

```bash
pip install -r requirements.txt
python -m src.pipeline          # downloads data on first run, rebuilds figures/ and results/ (~1 min)
python -m pytest                # 192 unit tests (~3 s)
```

**What reproduces exactly.**
- **Pricing and hedging:** the pricing results are deterministic. The hedging sample is pinned to closes up to 2026-10-22, so its results reproduce exactly. The only exception is if Yahoo or FRED revise history, which is rare.
- **Surface:** Yahoo's terms don't allow redistributing the raw quotes, so the option-chain snapshot is not committed. A fresh clone therefore takes a *new* snapshot, and the surface numbers will differ from those above (see the stability check in §2).
- **Refreshing:** add `--refresh-chain` to take a new snapshot in an existing checkout. To re-execute the notebook, run `jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb`.

## Repository layout

```
src/
  bsm.py             BSM price, Greeks, put-call parity gap
  implied_vol.py     Newton + Brent implied-vol solver
  binomial.py        CRR European/American trees, early-exercise premium
  montecarlo.py      MC with antithetic and control variates, standard errors
  data.py            downloads and caching (Yahoo Finance, FRED, Stooq fallback)
  surface.py         chain cleaning, IVs, parity check, SVI fit, surface
  hedging.py         delta-hedging engine, P&L decomposition, hedging error
  stats.py           Newey-West HAC, t-tests, moments
  pricing_report.py  convergence and variance-reduction figures/tables
  pipeline.py        runs everything end to end
notebooks/analysis.ipynb   end-to-end run with outputs
figures/   results/   tests/
docs/methodology-qa.md     the hardest interview questions, with answers
```

## References

- Hull, J. C., *Options, Futures, and Other Derivatives*: the textbook examples used in the tests.
- Cox, Ross & Rubinstein (1979), "Option pricing: a simplified approach".
- Manaster & Koehler (1982), "The calculation of implied variances from the Black–Scholes model".
- Gatheral, J. (2006), *The Volatility Surface* (SVI).
- Boyle & Emanuel (1980), "Discretely adjusted option hedges"; Derman & Kamal (1999), "When you cannot hedge continuously".
- Newey & West (1987), "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix".

MIT licence; see [LICENSE](LICENSE).

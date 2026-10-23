"""Delta-hedging P&L of a short 1-month ATM SPY call, its decomposition and
how the hedging error scales with rebalancing frequency.

Study design
------------
* Every trading day t0 from 2005-01-03 whose t0+21 close is in the data, we
  SELL one European call with K = S_t0 (close) and T = 21 trading days.
  Time is measured in trading days / 252 throughout (pricing, interest).
* The option is sold at the BSM price with sigma_imp = VIX_t0 / 100,
  r = 1-month T-bill at t0 (held for the window), q = trailing-12m dividend
  yield at t0.  VIX/r are aligned to SPY dates and forward-filled at most
  5 days (FRED has no T-bill quote on bond-market holidays).
* CAVEAT on VIX: VIX is a 30-calendar-day variance-swap rate, not the ATM
  implied vol. Because of the equity skew it is typically 1-3 vol points
  above 1-month ATM IV, so using it biases the seller's premium (and hence
  mean P&L) upwards. We use it only because a free daily history of SPY
  ATM IV does not exist; the hedging mechanics do not depend on it.
* The short call is delta-hedged with SPY shares at the close, rebalancing
  every h trading days (h = 1 daily, 5 weekly, 21 = initial hedge only).
  The hedge delta uses sigma_h:
    A "implied"   sigma_h = sigma_imp
    B "forecast"  sigma_h = trailing 21-day realised vol at t0 (known at t0)
    C "oracle"    sigma_h = realised vol of the window itself. LOOK-AHEAD:
                  a theoretical benchmark only, not a tradable strategy.
  Realised vol = sqrt(252 * mean(log-return^2)) of dividend-inclusive
  close-to-close returns (zero-mean convention, as in variance swaps and
  the gamma term below).
* Cash account: premium - Delta_0 S_0 - cost at t0; grows by e^{r/252} each
  trading day; receives Delta x cash dividend on ex-dates (shares held
  over the ex-date close); pays c x |change in Delta| x S on every trade.
  At expiry the call is cash-settled and the hedge unwound:
      P&L = cash_T + Delta S_T - c |Delta| S_T - max(S_T - K, 0).
  Base cost c = 1 bp of traded notional; 0 and 5 bp as sensitivity.
* Reported P&L is per option, in bp of S_0 (and relative to the premium).
* The SPY sample is pinned to closes up to END (2026-10-22) so a fresh clone
  reproduces the same windows.
* Inference: windows overlap, so means and regressions use Hansen-Hodrick
  (uniform-kernel HAC, lag 20) standard errors; see src/stats.py.
* Because VIX overstates ATM IV by an unknown amount, the MEAN P&L is not
  cleanly identified; we report a break-even VIX-minus-ATM spread and a
  sensitivity table instead of claiming a premium.

Decomposition (why it works)
----------------------------
BSM PDE: Theta + (r-q) S Delta + 1/2 sigma^2 S^2 Gamma - r V = 0.
Short the option, hold Delta shares (earning dividends at q) and the
financing balance V - Delta S in cash (earning r). Over a step dt,
    d(portfolio) = -dV + Delta (dS + q S dt) + r (V - Delta S) dt.
Taylor-expand dV = Theta dt + Delta dS + 1/2 Gamma dS^2 and substitute Theta
from the PDE; the Delta dS and r, q terms cancel, leaving
    d(portfolio) = 1/2 Gamma S^2 (sigma_h^2 dt - R^2),   R = dS / S.
So per rebalancing interval i (Delta t_i = days/252, Gamma_i at sigma_h):
    theta_i = +1/2 Gamma_i S_i^2 sigma_h^2 Delta t_i   (time decay earned)
    gamma_i = -1/2 Gamma_i S_i^2 R_i^2                  (convexity paid)
and   total = edge + sum theta + sum gamma - costs + residual,
with edge = premium(sigma_imp) - BSM(sigma_h) (zero for variant A) and the
residual collecting (i) the mismatch between the flat dividend yield used
for pricing and the actual discrete dividends received (the largest part:
windows containing an ex-date have a systematically positive residual),
(ii) third- and higher-order terms on large moves and (iii) interest on
accumulated P&L.  Note the "theory" regressor is built from the same
realised returns, so its fit is a check of the second-order attribution,
not an independent test of Black-Scholes.  Summing,
    P&L_short ~ edge + sum 1/2 Gamma S^2 (sigma_h^2 - sigma_realised^2) dt,
the long-gamma formula of the textbooks with the sign flipped.

Hedging error
-------------
eps = P&L - [BSM(sigma_imp) - BSM(sigma_real_window)]: the deviation from
the P&L that Black-Scholes says you lock in if you knew the window's
realised vol (ex-post attribution only). For an ATM option hedged N times
at the true vol, Boyle-Emanuel / Derman-Kamal give
    std(eps) ~ sqrt(pi/4) * vega * sigma / sqrt(N),
i.e. error proportional to sqrt(Delta t): slope 0.5 in log std vs log h.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import bsm, data, stats
from src.plotting import COLORS, RESULTS_DIR, save

START = "2005-01-03"
T_DAYS = 21                     # option life in trading days
YEAR = 252                      # trading days per year
FREQS = (1, 2, 5, 10, 21)       # rebalancing interval h in trading days
VARIANTS = ("implied", "forecast", "oracle")
VARIANT_LABEL = {"implied": "A: implied (VIX)",
                 "forecast": "B: trailing 21d RV",
                 "oracle": "C: oracle RV (look-ahead)"}
BASE_COST = 1e-4                # 1 bp of traded notional
COSTS = (0.0, 1e-4, 5e-4)
HAC_LAG = 20                    # windows overlap by up to 20 days
HAC_KERNEL = "uniform"          # Hansen-Hodrick; see src/stats.py
END = "2026-10-22"              # last SPY close used (pins the sample)
VIX_SPREADS = (0.0, 1.0, 2.0, 3.0, 4.0)   # assumed VIX - ATM IV, vol pts
FFILL_LIMIT = 5
BP = 1e4
SOURCE = "Yahoo Finance (SPY), FRED (VIXCLS, DGS1MO)"


# --------------------------------------------------------------------------
# Core hedging engine (vectorised over windows)
# --------------------------------------------------------------------------
@dataclass
class HedgeResult:
    """Per-window results, all in currency units per option."""
    pnl: np.ndarray
    premium: np.ndarray
    edge: np.ndarray
    theta: np.ndarray
    gamma: np.ndarray
    costs: np.ndarray
    residual: np.ndarray


def hedge_pnl(S: np.ndarray, div: np.ndarray, r: np.ndarray, q: np.ndarray,
              sigma_imp: np.ndarray, sigma_h: np.ndarray, h: int,
              cost: float = BASE_COST, year: int = YEAR) -> HedgeResult:
    """Simulate a short ATM call delta-hedged every h days.

    S, div : (m, N+1) closes and cash dividends (ex-date) for m windows;
             column 0 is t0, column N is expiry. div[:, 0] is ignored.
    r, q, sigma_imp, sigma_h : (m,) per-window constants known at t0
             (except sigma_h for the oracle variant).
    """
    S, div = np.atleast_2d(S).astype(float), np.atleast_2d(div).astype(float)
    m, n_steps = S.shape[0], S.shape[1] - 1
    r, q, sigma_imp, sigma_h = (np.broadcast_to(np.asarray(a, float), (m,))
                                for a in (r, q, sigma_imp, sigma_h))
    K, T = S[:, 0], n_steps / year
    growth = np.exp(r / year)                   # one trading day of interest

    premium = bsm.price(S[:, 0], K, T, r, q, sigma_imp)
    edge = premium - bsm.price(S[:, 0], K, T, r, q, sigma_h)

    rebal = list(range(0, n_steps, h))          # rebalancing days j < N
    theta = np.zeros(m)
    gamma = np.zeros(m)
    for j, j_next in zip(rebal, rebal[1:] + [n_steps]):
        tau, dt = (n_steps - j) / year, (j_next - j) / year
        g = bsm.gamma(S[:, j], K, tau, r, q, sigma_h)
        half_gs2 = 0.5 * g * S[:, j] ** 2
        R = S[:, j_next] / S[:, j] - 1.0
        theta += half_gs2 * sigma_h**2 * dt
        gamma -= half_gs2 * R**2

    # Cash account, day by day.
    held = bsm.delta(S[:, 0], K, T, r, q, sigma_h)
    costs = cost * np.abs(held) * S[:, 0]
    cash = premium - held * S[:, 0] - costs
    for j in range(1, n_steps + 1):
        cash = cash * growth + held * div[:, j]
        if j < n_steps and j % h == 0:
            new = bsm.delta(S[:, j], K, (n_steps - j) / year, r, q, sigma_h)
            trade_cost = cost * np.abs(new - held) * S[:, j]
            cash -= (new - held) * S[:, j] + trade_cost
            costs = costs + trade_cost
            held = new
    unwind = cost * np.abs(held) * S[:, -1]
    costs = costs + unwind
    payoff = np.maximum(S[:, -1] - K, 0.0)
    pnl = cash + held * S[:, -1] - unwind - payoff

    residual = pnl - (edge + theta + gamma - costs)
    return HedgeResult(pnl, premium, edge, theta, gamma, costs, residual)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def total_log_returns(close: pd.Series, div: pd.Series) -> pd.Series:
    """ln((S_t + D_t) / S_{t-1}): close-to-close return including the cash
    dividend paid on the ex-date (so ex-dates are not fake down-moves)."""
    return np.log((close + div.fillna(0.0)) / close.shift(1))


def trailing_realised_vol(close: pd.Series, div: pd.Series,
                          window: int = T_DAYS) -> pd.Series:
    """Annualised realised vol of the `window` daily returns ending at t
    (inclusive), zero-mean: sqrt(252 * mean(ret^2)). Known at the close of t."""
    ret = total_log_returns(close, div)
    return np.sqrt(YEAR * (ret**2).rolling(window).mean())


def load_windows() -> dict:
    """Build the (m, 22) price/dividend matrices and per-window inputs from
    the cached data only."""
    hist = data.spy_history()
    hist = hist[hist.index <= pd.Timestamp(END)]
    close, div = hist["Close"], hist["Dividends"].fillna(0.0)
    idx = hist.index
    vix = data.vix().reindex(idx).ffill(limit=FFILL_LIMIT) / 100.0
    r = data.risk_free_1m().reindex(idx).ffill(limit=FFILL_LIMIT)
    q = data.trailing_dividend_yield(hist)
    rv = trailing_realised_vol(close, div)

    pos = np.arange(len(idx))
    ok = ((idx >= pd.Timestamp(START)) & (pos + T_DAYS < len(idx))
          & vix.notna().to_numpy() & r.notna().to_numpy()
          & rv.notna().to_numpy())
    starts = pos[ok]
    cols = starts[:, None] + np.arange(T_DAYS + 1)[None, :]
    S = close.to_numpy()[cols]
    D = div.to_numpy()[cols]
    # Window realised vol = trailing RV measured at the expiry date
    # (uses the window's own 21 returns -> look-ahead, oracle/attribution).
    rv_window = rv.to_numpy()[starts + T_DAYS]
    return {"dates": idx[starts], "S": S, "D": D,
            "r": r.to_numpy()[starts], "q": q.to_numpy()[starts],
            "sigma_imp": vix.to_numpy()[starts],
            "sigma_fcst": rv.to_numpy()[starts],
            "sigma_real": rv_window}


def hedge_vol(w: dict, variant: str) -> np.ndarray:
    return {"implied": w["sigma_imp"], "forecast": w["sigma_fcst"],
            "oracle": w["sigma_real"]}[variant]


def hedging_error(w: dict, res: HedgeResult) -> np.ndarray:
    """eps = P&L - [BSM(sigma_imp) - BSM(sigma_real_window)]."""
    S0 = w["S"][:, 0]
    locked = res.premium - bsm.price(S0, S0, T_DAYS / YEAR, w["r"], w["q"],
                                     w["sigma_real"])
    return res.pnl - locked


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------
def _summary_row(variant: str, h: int, res: HedgeResult, eps: np.ndarray,
                 S0: np.ndarray) -> dict:
    pnl = res.pnl / S0 * BP
    mt = stats.mean_test(pnl, HAC_LAG, HAC_KERNEL)
    no = stats.nonoverlap_ttest(pnl, T_DAYS)
    mo = stats.moments(pnl)
    return {"variant": variant, "h": h, "n": pnl.size,
            "mean_pnl_bp": mt["mean"], "std_pnl_bp": pnl.std(ddof=1),
            "q05_bp": np.quantile(pnl, 0.05), "q95_bp": np.quantile(pnl, 0.95),
            "mean_pnl_pct_prem": 100 * np.mean(res.pnl / res.premium),
            "mean_edge_bp": np.mean(res.edge / S0) * BP,
            "mean_theta_bp": np.mean(res.theta / S0) * BP,
            "mean_gamma_bp": np.mean(res.gamma / S0) * BP,
            "mean_costs_bp": np.mean(res.costs / S0) * BP,
            "mean_residual_bp": np.mean(res.residual / S0) * BP,
            "std_eps_bp": np.std(eps / S0 * BP, ddof=1),
            "hac_t": mt["t"], "hac_p": mt["p"],
            "nonoverlap_n": no["n"], "nonoverlap_t": no["t"],
            "nonoverlap_p": no["p"], "skew": mo["skew"], "kurt": mo["kurt"],
            "jb_p": mo["jb_p"]}


def scaling_fit(h: np.ndarray, std_eps: np.ndarray) -> dict[str, float]:
    """OLS of log std(eps) on log h. The SE is the plain OLS SE from 5
    points; it ignores that the 5 stds come from the same windows, so read
    it as descriptive."""
    x, y = np.log(h), np.log(std_eps)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    s2 = u @ u / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"slope": float(beta[1]), "slope_se": float(se[1]),
            "intercept": float(beta[0])}


def run() -> dict:
    """Run the full study from the cache, write results/ and figures/,
    return the headline numbers."""
    RESULTS_DIR.mkdir(exist_ok=True)
    w = load_windows()
    S0 = w["S"][:, 0]
    tau = T_DAYS / YEAR

    rows, tc_rows, results, eps_all = [], [], {}, {}
    for variant in VARIANTS:
        sig_h = hedge_vol(w, variant)
        for h in FREQS:
            for c in COSTS:
                res = hedge_pnl(w["S"], w["D"], w["r"], w["q"],
                                w["sigma_imp"], sig_h, h, c)
                mt = stats.mean_test(res.pnl / S0 * BP, HAC_LAG, HAC_KERNEL)
                tc_rows.append({"variant": variant, "h": h, "cost_bp": c * BP,
                                "mean_pnl_bp": mt["mean"], "hac_t": mt["t"],
                                "hac_p": mt["p"],
                                "mean_costs_bp": np.mean(res.costs / S0) * BP})
                if c == BASE_COST:
                    eps = hedging_error(w, res)
                    results[variant, h], eps_all[variant, h] = res, eps
                    rows.append(_summary_row(variant, h, res, eps, S0))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "hedging_summary.csv", index=False,
                   float_format="%.4f")
    pd.DataFrame(tc_rows).to_csv(RESULTS_DIR / "hedging_tc_sensitivity.csv",
                                 index=False, float_format="%.4f")

    # Scaling of std(eps) with h, against the Derman-Kamal approximation.
    vega_bp = np.median(bsm.vega(S0, S0, tau, w["r"], w["q"],
                                 w["sigma_imp"]) / S0) * BP
    sigma_med = float(np.median(w["sigma_real"]))
    hs = np.array(FREQS, float)
    n_rebal = np.ceil(T_DAYS / hs)
    scaling = pd.DataFrame({"h": FREQS, "n_rebalances": n_rebal.astype(int)})
    for v in VARIANTS:
        scaling[f"std_eps_{v}_bp"] = summary.query("variant == @v")[
            "std_eps_bp"].to_numpy()
    # Derman-Kamal per window i: sd(eps_i) ~ sqrt(pi/4) vega_i sigma_i / sqrt(N),
    # evaluated for the oracle hedge (vega and sigma at the realised vol).
    # The cross-sectional sd it implies is the RMS over windows (plugging in
    # medians would understate it, because high-vol windows dominate).
    scale_i = bsm.vega(S0, S0, tau, w["r"], w["q"], w["sigma_real"]) \
        * w["sigma_real"] / S0 * BP
    scaling["theory_bp"] = (np.sqrt(np.pi / 4) * np.sqrt(np.mean(scale_i**2))
                            / np.sqrt(n_rebal))
    # Standardised error eps_i / (vega_i sigma_i / sqrt(N)): theory says its
    # sd is sqrt(pi/4) = 0.886 at every h.
    scaling["std_standardised_eps_oracle"] = [
        float(np.std(eps_all["oracle", h_] / S0 * BP
                     / (scale_i / np.sqrt(n)), ddof=1))
        for h_, n in zip(FREQS, n_rebal)]
    scaling.to_csv(RESULTS_DIR / "hedging_scaling.csv", index=False,
                   float_format="%.4f")
    fits = {v: scaling_fit(hs, scaling[f"std_eps_{v}_bp"].to_numpy())
            for v in VARIANTS}
    # With N = ceil(21/h) rebalances the theoretical curve is not exactly
    # proportional to sqrt(h); its own log-log slope is the benchmark.
    fits["theory"] = scaling_fit(hs, scaling["theory_bp"].to_numpy())

    # Realised P&L vs the theoretical gamma-theta P&L (variant A, daily).
    resA = results["implied", 1]
    theory = (resA.edge + resA.theta + resA.gamma) / S0 * BP
    pnlA = resA.pnl / S0 * BP
    reg = stats.hac_ols(pnlA, np.column_stack([np.ones_like(theory), theory]),
                        HAC_LAG, HAC_KERNEL)
    slope, slope_se = float(reg["beta"][1]), float(reg["se"][1])
    t1 = (slope - 1.0) / slope_se

    # Per-window file: only what the figures need (variant A, h = 1 and 5).
    frames = []
    for h in (1, 5):
        res = results["implied", h]
        frames.append(pd.DataFrame({
            "t0": w["dates"], "variant": "implied", "h": h,
            "S0": S0, "sigma_imp": w["sigma_imp"],
            "sigma_real": w["sigma_real"],
            "premium_bp": res.premium / S0 * BP, "pnl_bp": res.pnl / S0 * BP,
            "edge_bp": res.edge / S0 * BP, "theta_bp": res.theta / S0 * BP,
            "gamma_bp": res.gamma / S0 * BP, "costs_bp": res.costs / S0 * BP,
            "residual_bp": res.residual / S0 * BP,
            "eps_bp": eps_all["implied", h] / S0 * BP}))
    windows = pd.concat(frames, ignore_index=True)
    windows.to_csv(RESULTS_DIR / "hedging_windows.csv", index=False,
                   float_format="%.5f")

    # Does hedging less often change the MEAN P&L? Paired difference per
    # window (h = 21 minus h = 1, same option), HAC-tested.
    diffs = {v: (results[v, 21].pnl - results[v, 1].pnl) / S0 * BP
             for v in VARIANTS}
    paired = {v: stats.mean_test(diffs[v], HAC_LAG, HAC_KERNEL)
              for v in VARIANTS}
    paired_no = {v: stats.nonoverlap_all_offsets(diffs[v], T_DAYS)
                 for v in VARIANTS}

    # Five worst windows (variant A, daily), one per distinct episode is not
    # enforced: neighbouring start dates share most of their path.
    worst5 = np.argsort(pnlA)[:5]
    worst_table = [{"t0": str(w["dates"][i].date()), "pnl_bp": float(pnlA[i]),
                    "vix_at_sale": float(100 * w["sigma_imp"][i]),
                    "realised_vol": float(100 * w["sigma_real"][i])}
                   for i in worst5]

    worst = int(np.argmin(pnlA))
    rowA = summary.query("variant == 'implied' and h == 1").iloc[0]
    prem_test = stats.mean_test(100 * resA.pnl / resA.premium, HAC_LAG,
                                HAC_KERNEL)

    # Is the mean P&L identified? VIX sits above 1-month ATM IV, so the
    # premium is overstated. Re-run variant A (daily) selling AND hedging at
    # sigma_imp = VIX - spread for a range of assumed spreads.
    vega_imp = bsm.vega(S0, S0, tau, w["r"], w["q"], w["sigma_imp"])
    breakeven_volpts = float(np.mean(resA.pnl) / np.mean(vega_imp) * 100)
    spread_rows = []
    for sp in VIX_SPREADS:
        sig = np.maximum(w["sigma_imp"] - sp / 100, 0.01)
        res = hedge_pnl(w["S"], w["D"], w["r"], w["q"], sig, sig, 1)
        mt = stats.mean_test(res.pnl / S0 * BP, HAC_LAG, HAC_KERNEL)
        spread_rows.append({"vix_minus_atm_volpts": sp,
                            "mean_pnl_bp": mt["mean"], "hh_t": mt["t"],
                            "hh_p": mt["p"]})
    spread_tab = pd.DataFrame(spread_rows)
    spread_tab.to_csv(RESULTS_DIR / "hedging_vix_spread_sensitivity.csv",
                      index=False, float_format="%.4f")

    # Where does the decomposition residual come from? Pricing uses a flat
    # continuous q while the hedge receives the actual discrete dividend, so
    # windows containing an ex-date differ systematically from those without.
    has_div = (w["D"][:, 1:] > 0).any(axis=1)
    resid_bp = resA.residual / S0 * BP
    dummy_r2 = stats.hac_ols(resid_bp, np.column_stack(
        [np.ones_like(resid_bp), has_div.astype(float)]))["r2"]
    eps_or = eps_all["oracle", 1] / S0 * BP

    # Mean reversion check for the "less hedging earns more" result:
    # variance ratios of daily SPY total log returns over the study period.
    rets = np.log((w["S"][:, 1] + w["D"][:, 1]) / w["S"][:, 0])
    vr = {f"VR({q_})": float(np.var(np.convolve(rets, np.ones(q_), "valid"))
                             / (q_ * np.var(rets))) for q_ in (2, 5, 21)}
    headline = {
        "n_windows": int(len(S0)),
        "first_t0": str(w["dates"][0].date()),
        "last_t0": str(w["dates"][-1].date()),
        "cost_bp": BASE_COST * BP,
        "daily_implied": {
            "mean_pnl_bp_of_spot": rowA["mean_pnl_bp"],
            "hh_t": rowA["hac_t"], "hh_p": rowA["hac_p"],
            "mean_pnl_pct_of_premium": prem_test["mean"],
            "pct_prem_hh_t": prem_test["t"], "pct_prem_hh_p": prem_test["p"],
            "mean_premium_bp_of_spot": float(np.mean(resA.premium / S0) * BP),
            "frac_positive": float(np.mean(resA.pnl > 0)),
            "worst_t0": str(w["dates"][worst].date()),
            "worst_pnl_bp_of_spot": float(pnlA[worst]),
            "worst_pnl_pct_of_premium": float(100 * resA.pnl[worst]
                                              / resA.premium[worst]),
        },
        # std(eps) weekly / daily per variant vs sqrt(5) (sqrt(21/5) if we
        # count the actual 21 vs 5 rebalances).
        "std_eps_bp": {v: {"daily": float(scaling.loc[0, f"std_eps_{v}_bp"]),
                           "weekly": float(scaling.loc[2, f"std_eps_{v}_bp"]),
                           "ratio_weekly_to_daily":
                               float(scaling.loc[2, f"std_eps_{v}_bp"]
                                     / scaling.loc[0, f"std_eps_{v}_bp"])}
                       for v in VARIANTS},
        "theory_ratio_sqrt5": float(np.sqrt(5)),
        "theory_ratio_sqrt_21_over_5": float(np.sqrt(21 / 5)),
        "scaling_slope": {v: fits[v] for v in VARIANTS},
        "hac": f"{HAC_KERNEL} kernel (Hansen-Hodrick), lag {HAC_LAG}",
        "theory_inputs": {"median_vega_bp_per_vol": vega_bp,
                          "median_sigma_real": sigma_med},
        "theory_slope_with_ceil_N": fits["theory"]["slope"],
        "derman_kamal_rms_bp": {str(h_): float(v_) for h_, v_ in
                                zip(FREQS, scaling["theory_bp"])},
        "standardised_eps_sd_oracle": {
            str(h_): float(v_) for h_, v_ in
            zip(FREQS, scaling["std_standardised_eps_oracle"])},
        "standardised_eps_sd_theory": float(np.sqrt(np.pi / 4)),
        "pnl_vs_theory_regression": {
            "variant": "implied", "h": 1, "slope": slope,
            "slope_hac_se": slope_se, "t_slope_eq_1": float(t1),
            "p_slope_eq_1": float(2 * stats.sps.norm.sf(abs(t1))),
            "intercept_bp": float(reg["beta"][0]), "r2": reg["r2"]},
        "residual_std_bp_daily_implied": float(np.std(resA.residual / S0 * BP,
                                                      ddof=1)),
        "paired_mean_diff_h21_minus_h1_bp": {
            v: {"mean": paired[v]["mean"], "hh_t": paired[v]["t"],
                "hh_p": paired[v]["p"],
                "nonoverlap_p_over_21_offsets": paired_no[v]}
            for v in VARIANTS},
        "daily_return_variance_ratios": vr,
        "breakeven_vix_minus_atm_volpts": breakeven_volpts,
        "vix_spread_sensitivity": spread_rows,
        "residual_by_exdate_daily_implied": {
            "share_windows_with_exdate": float(has_div.mean()),
            "mean_resid_with_exdate_bp": float(resid_bp[has_div].mean()),
            "mean_resid_without_bp": float(resid_bp[~has_div].mean()),
            "r2_of_exdate_dummy": float(dummy_r2),
            "oracle_eps_mean_with_exdate_bp": float(eps_or[has_div].mean()),
            "oracle_eps_mean_without_bp": float(eps_or[~has_div].mean())},
        "worst_windows_daily_implied": worst_table,
        "mean_sigma_imp_pct": float(100 * np.mean(w["sigma_imp"])),
        "mean_sigma_real_pct": float(100 * np.mean(w["sigma_real"])),
    }
    headline = json.loads(json.dumps(headline, default=float))
    (RESULTS_DIR / "hedging_summary.json").write_text(
        json.dumps(headline, indent=2))

    _plot_distribution(windows)
    _plot_decomposition(summary, windows)
    _plot_scaling(scaling, fits)
    _plot_vs_theory(theory, pnlA, reg)
    return headline


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plot_distribution(windows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in ((axes[0], "pnl_bp", "Hedged P&L of short call"),
                           (axes[1], "eps_bp",
                            "Hedging error ε = P&L − [BSM(σ_imp) − BSM(σ_real)]")):
        lo, hi = np.percentile(windows[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 80)
        for k, (h, lab) in enumerate(((1, "daily (h=1)"), (5, "weekly (h=5)"))):
            x = windows.loc[windows["h"] == h, col]
            ax.hist(x, bins=bins, density=True, alpha=0.55, color=COLORS[k],
                    label=f"{lab}: mean {x.mean():.1f}, sd {x.std():.1f}")
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel("bp of spot S₀ (0.5–99.5% range shown)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Variant A (hedge at σ_imp = VIX), 1-month ATM SPY call, "
                 "1 bp costs", fontsize=12)
    save(fig, "07_hedge_pnl_distribution", SOURCE)


def _plot_decomposition(summary: pd.DataFrame, windows: pd.DataFrame) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9),
                                  gridspec_kw={"height_ratios": [1, 1]})
    comps = [("mean_edge_bp", "vol edge"), ("mean_theta_bp", "Σ theta"),
             ("mean_gamma_bp", "Σ gamma"), ("mean_costs_bp", "− costs"),
             ("mean_residual_bp", "residual")]
    x = np.arange(len(summary))
    width = 0.16
    for k, (col, lab) in enumerate(comps):
        vals = summary[col].to_numpy() * (-1 if col == "mean_costs_bp" else 1)
        ax.bar(x + (k - 2) * width, vals, width, color=COLORS[k], label=lab)
    ax.scatter(x, summary["mean_pnl_bp"], color="black", marker="D", s=22,
               zorder=3, label="total mean P&L")
    letter = {"implied": "A", "forecast": "B", "oracle": "C"}
    labels = [f"{letter[v]} h={h}" for v, h in zip(summary["variant"],
                                                  summary["h"])]
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    for b in (4.5, 9.5):
        ax.axvline(b, color="0.6", lw=0.8, ls=":")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("mean per window, bp of S₀")
    ax.set_title("Mean P&L decomposition by hedge-vol variant "
                 "(A implied, B trailing RV, C oracle) and rebalancing interval h")
    ax.legend(ncol=6, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)

    d = windows[windows["h"] == 1].set_index("t0")["pnl_bp"]
    ax2.plot(d.index, d, color=COLORS[0], lw=0.6, alpha=0.6,
             label="window P&L (by sale date t₀)")
    ax2.plot(d.index, d.rolling(252).mean(), color=COLORS[1], lw=2,
             label="252-window rolling mean")
    ax2.axhline(0, color="0.3", lw=0.8)
    # Label the worst window of the four worst calendar years.
    worst_by_year = d.groupby(d.index.year).idxmin()
    for t0 in d[worst_by_year.to_numpy()].nsmallest(4).index:
        ax2.annotate(t0.strftime("%d %b %Y"), (t0, d[t0]), fontsize=8,
                     xytext=(6, 0), textcoords="offset points")
    ax2.set_ylabel("P&L, bp of S₀")
    ax2.set_title("Variant A, daily hedge: P&L of each 21-day window")
    ax2.legend(loc="upper right")
    save(fig, "08_pnl_decomposition", SOURCE)


def _plot_scaling(scaling: pd.DataFrame, fits: dict) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    h = scaling["h"].to_numpy(float)
    for k, v in enumerate(VARIANTS):
        y = scaling[f"std_eps_{v}_bp"]
        f = fits[v]
        ax.plot(h, y, "o-", color=COLORS[k], ms=7,
                label=f"{VARIANT_LABEL[v]}: slope {f['slope']:.2f} "
                      f"± {f['slope_se']:.2f}")
    ax.plot(h, scaling["theory_bp"], "k--", lw=1.5,
            label=f"Derman–Kamal √(π/4)·vega·σ/√N, N=⌈21/h⌉ "
                  f"(RMS over windows): slope {fits['theory']['slope']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(FREQS, [str(x) for x in FREQS])
    ax.set_xlabel("rebalancing interval h (trading days, log scale)")
    ax.set_ylabel("std of hedging error ε (bp of S₀, log scale)")
    ax.set_title("sd of hedging error vs rebalancing interval")
    ax.legend(fontsize=7.5)
    ax2.plot(h, scaling["std_standardised_eps_oracle"], "o-", color=COLORS[2],
             ms=7, label="C: oracle, sd of ε_i / (vega_i·σ_i/√N)")
    ax2.axhline(np.sqrt(np.pi / 4), color="k", ls="--", lw=1.2,
                label="theory √(π/4) = 0.886")
    ax2.set_xscale("log")
    ax2.set_xticks(FREQS, [str(x) for x in FREQS])
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("rebalancing interval h (trading days, log scale)")
    ax2.set_ylabel("sd of standardised hedging error")
    ax2.set_title("Per-window test of the Derman–Kamal formula")
    ax2.legend(fontsize=8, loc="lower left")
    save(fig, "09_hedging_error_scaling", SOURCE)


def _plot_vs_theory(theory: np.ndarray, pnl: np.ndarray, reg: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(theory, pnl, s=4, alpha=0.3, color=COLORS[0],
               label="one 21-day window")
    lim = [min(theory.min(), pnl.min()), max(theory.max(), pnl.max())]
    ax.plot(lim, lim, "k--", lw=1, label="45° line")
    a, b = reg["beta"]
    ax.plot(lim, a + b * np.array(lim), color=COLORS[1], lw=1.5,
            label=f"OLS: slope {b:.3f} (HAC se {reg['se'][1]:.3f}), "
                  f"R² {reg['r2']:.3f}")
    ax.set_xlabel("theoretical P&L  Σ ½ΓS²(σ_h²Δt − R²)  (bp of S₀)")
    ax.set_ylabel("realised hedged P&L (bp of S₀)")
    ax.set_title("Variant A, daily hedge: realised vs gamma–theta P&L")
    ax.legend(loc="upper left")
    save(fig, "10_pnl_vs_theory", SOURCE)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

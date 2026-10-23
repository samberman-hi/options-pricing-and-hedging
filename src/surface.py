"""Implied-volatility surface of SPY options from one cached chain snapshot.

Pipeline: cached Yahoo chain -> time to expiry, rates, dividend yield,
forwards -> cleaning (with a count after every step) -> BSM implied vols ->
put-call parity and American-exercise diagnostics -> raw-SVI smile per expiry
(with an out-of-sample check) -> calendar-arbitrage check -> figures/tables.

Assumptions (all stated so they can be challenged):
    * Time to expiry. SPY options stop trading at 16:00 America/New_York on
      the expiry date, so T = (expiry 16:00 ET - snapshot time) / 365 days,
      in calendar time. The snapshot is intraday (taken during the session).
    * Risk-free rate r(T): FRED Treasury constant-maturity yields (1M..2Y),
      latest observation on or before the snapshot date (FRED publishes with
      a lag of a day or two), converted to continuous compounding and
      linearly interpolated in maturity (flat beyond 2Y) by
      data.rate_for_maturity.
    * Dividend yield q: trailing-12-month cash dividends / close, from the
      Yahoo daily history, last value on or before the snapshot date
      (data.trailing_dividend_yield). One flat continuous q for all expiries;
      SPY actually pays discrete quarterly dividends, so for a short expiry
      that contains no ex-date the true forward is slightly above our model
      forward. The parity diagnostic below measures how large that is.
    * Forward F = S e^{(r - q) T}; log-moneyness k = ln(K / F).
    * Exercise style. SPY options are American. We use only out-of-the-money
      options (puts with K < F, calls with K >= F): they are the liquid side
      of each strike and their early-exercise premium is negligible, so the
      European BSM formula is a good approximation. The American-tree
      diagnostic quantifies "negligible".
    * Put-call parity C - P = e^{-rT}(F - K) holds exactly only for European
      options and synchronous prices. With American options and a spot price
      whose timestamp need not match the option quotes exactly, the
      parity-implied forward is only an approximate check of the model F.
      The main surface uses the model F (FRED r and data q), as specified;
      the parity numbers are reported as a diagnostic, not used.
"""

from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from src import binomial, data
from src.implied_vol import implied_vol, no_arbitrage_bounds
from src.plotting import COLORS, RESULTS_DIR, plt, save

log = logging.getLogger(__name__)

EXPIRY_CLOSE = "16:00"           # SPY options stop trading 16:00 New York
STALE_BDAYS = 3                  # last trade must be within 3 business days
MAX_REL_SPREAD = 0.5             # (ask - bid) / mid
MIN_DAYS = 7                     # drop expiries shorter than a week
K_MIN, K_MAX = -0.4, 0.25        # log-moneyness window ln(K/F)
PARITY_K = 0.05                  # |k| window for the parity diagnostic
HOLDOUT_EVERY = 5                # out-of-sample: hold out every 5th strike
TREE_STEPS = 200


# --------------------------------------------------------------------------
# Inputs: time, rates, forwards
# --------------------------------------------------------------------------
def time_to_expiry(expiry: pd.Series | str, snapshot: pd.Timestamp
                   ) -> np.ndarray | float:
    """Year fraction (calendar days / 365) from the snapshot to 16:00 New
    York time on the expiry date. Handles EDT/EST automatically."""
    exp = pd.to_datetime(pd.Series(expiry).astype(str) + " " + EXPIRY_CLOSE)
    exp_utc = exp.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    secs = (exp_utc - snapshot.tz_convert("UTC")).dt.total_seconds().to_numpy()
    T = secs / (365.0 * 86400.0)
    return float(T[0]) if isinstance(expiry, str) else T


def forward(S: float, r: np.ndarray | float, q: float,
            T: np.ndarray | float) -> np.ndarray | float:
    """F = S e^{(r - q) T}."""
    return S * np.exp((np.asarray(r) - q) * np.asarray(T))


def market_inputs(snapshot: pd.Timestamp) -> tuple[pd.Series, float]:
    """Treasury curve and trailing dividend yield as of the snapshot date."""
    day = snapshot.tz_convert("America/New_York").tz_localize(None).normalize()
    curve = data.treasury_curve(day)
    q_series = data.trailing_dividend_yield(data.spy_history())
    q = float(q_series[q_series.index <= day].iloc[-1])
    return curve, q


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def add_model_columns(chain: pd.DataFrame, snapshot: pd.Timestamp, S: float,
                      q: float, curve: pd.Series) -> pd.DataFrame:
    """Add T, r, F, k = ln(K/F) and the quote mid to every row."""
    df = chain.copy()
    df["T"] = time_to_expiry(df["expiry"], snapshot)
    df["r"] = data.rate_for_maturity(curve, df["T"].clip(lower=0.0))
    df["F"] = forward(S, df["r"], q, df["T"])
    df["k"] = np.log(df["strike"] / df["F"])
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    return df


def select_otm(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-the-money options only: puts with K < F, calls with K >= F."""
    otm = np.where(df["kind"] == "put", df["strike"] < df["F"],
                   df["strike"] >= df["F"])
    return df[otm]


def solve_ivs(df: pd.DataFrame, S: float, q: float) -> pd.DataFrame:
    """BSM implied vol of each mid, recording which solver branch was used."""
    out = [implied_vol(m, S, K, T, r, q, kind, return_method=True)
           for m, K, T, r, kind in zip(df["mid"], df["strike"], df["T"],
                                       df["r"], df["kind"])]
    df = df.copy()
    df["iv"] = [o[0] for o in out]
    df["iv_method"] = [o[1] for o in out]
    return df


def clean_chain(chain: pd.DataFrame, snapshot: pd.Timestamp, S: float,
                q: float, curve: pd.Series
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the cleaning filters in order.

    Returns (clean OTM points with IVs, step-count table, the two-sided
    quotes before the OTM filter - used for the parity diagnostic)."""
    df = add_model_columns(chain, snapshot, S, q, curve)
    steps = [("raw chain", len(df))]

    def step(name: str, mask: pd.Series) -> None:
        nonlocal df
        df = df[mask]
        steps.append((name, len(df)))

    step("two-sided quote (bid > 0, ask > bid)",
         (df["bid"] > 0) & (df["ask"] > df["bid"]))
    cutoff = (snapshot.tz_convert("UTC").normalize()
              - pd.offsets.BDay(STALE_BDAYS))
    step(f"traded within {STALE_BDAYS} business days",
         df["lastTradeDate"] >= cutoff)
    step(f"relative spread <= {MAX_REL_SPREAD:.0%}",
         (df["ask"] - df["bid"]) / df["mid"] <= MAX_REL_SPREAD)
    step(f"T >= {MIN_DAYS} calendar days", df["T"] >= MIN_DAYS / 365.0)
    step(f"{K_MIN} <= ln(K/F) <= {K_MAX}", df["k"].between(K_MIN, K_MAX))
    pre_otm = df.copy()
    df = select_otm(df)
    steps.append(("out-of-the-money only", len(df)))
    bounds = [no_arbitrage_bounds(S, K, T, r, q, kind) for K, T, r, kind
              in zip(df["strike"], df["T"], df["r"], df["kind"])]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    step("mid inside European no-arbitrage bounds",
         (df["mid"] > lo) & (df["mid"] < hi))
    df = solve_ivs(df, S, q)
    step("implied vol solved", df["iv"].notna())
    table = pd.DataFrame(steps, columns=["step", "rows_remaining"])
    return df.reset_index(drop=True), table, pre_otm


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def parity_forward(call: np.ndarray | float, put: np.ndarray | float,
                   K: np.ndarray | float, T: float, r: float
                   ) -> np.ndarray | float:
    """Forward implied by European put-call parity:
    C - P = e^{-rT}(F - K)  =>  F = K + e^{rT}(C - P)."""
    return np.asarray(K) + np.exp(r * T) * (np.asarray(call) - np.asarray(put))


def parity_check(quotes: pd.DataFrame, S: float) -> pd.DataFrame:
    """Per expiry: median parity-implied forward from near-ATM strikes with
    both a clean call and put, versus the model forward."""
    rows = []
    for expiry, g in quotes[quotes["k"].abs() <= PARITY_K].groupby("expiry"):
        pairs = g.pivot_table(index="strike", columns="kind", values="mid")
        pairs = pairs.dropna()
        if pairs.empty or "call" not in pairs or "put" not in pairs:
            continue
        T, r, F = g["T"].iloc[0], g["r"].iloc[0], g["F"].iloc[0]
        F_par = float(np.median(parity_forward(pairs["call"], pairs["put"],
                                               pairs.index.values, T, r)))
        rows.append({
            "expiry": expiry, "T": T, "n_strikes": len(pairs),
            "F_model": F, "F_parity_median": F_par,
            "diff_bp": 1e4 * (F_par / F - 1.0),
            # S e^{(r - q)T} = F  =>  q = r - ln(F / S) / T
            "q_implied": r - math.log(F_par / S) / T,
        })
    return pd.DataFrame(rows)


def _tree_iv(price: float, S: float, K: float, T: float, r: float, q: float,
             kind: str, american: bool) -> float:
    """Volatility at which the CRR tree reproduces `price` (NaN if none)."""
    def f(sig: float) -> float:
        return binomial.crr_price(S, K, T, r, q, sig, TREE_STEPS, kind,
                                  american) - price
    try:
        return brentq(f, 0.01, 3.0, xtol=1e-8)
    except ValueError:  # no sign change: price outside the tree's range
        return math.nan


def american_iv_diagnostic(points: pd.DataFrame, S: float, q: float,
                           target_days: tuple[int, ...] = (30, 365)
                           ) -> pd.DataFrame:
    """For the expiries nearest `target_days`, back out IV from a 200-step
    CRR tree, American and European, and compare with the BSM IV.

    IV_amer - IV_bsm is what the brief asks for; it mixes the early-exercise
    effect with tree discretisation error. IV_amer - IV_euro_tree uses the
    same tree for both, so it isolates the early-exercise effect."""
    rows = []
    for expiry in _nearest_expiries(points, target_days):
        g = points[points["expiry"] == expiry]
        for p in g.itertuples():
            iv_am = _tree_iv(p.mid, S, p.strike, p.T, p.r, q, p.kind, True)
            iv_eu = _tree_iv(p.mid, S, p.strike, p.T, p.r, q, p.kind, False)
            rows.append({"expiry": expiry, "T": p.T, "kind": p.kind,
                         "strike": p.strike, "k": p.k, "iv_bsm": p.iv,
                         "iv_american_tree": iv_am, "iv_european_tree": iv_eu})
    df = pd.DataFrame(rows)
    df["amer_minus_bsm_volpts"] = 100 * (df["iv_american_tree"] - df["iv_bsm"])
    df["amer_minus_euro_tree_volpts"] = 100 * (df["iv_american_tree"]
                                               - df["iv_european_tree"])
    return df


# --------------------------------------------------------------------------
# SVI
# --------------------------------------------------------------------------
def svi_total_variance(k: np.ndarray | float, a: float, b: float, rho: float,
                       m: float, s: float) -> np.ndarray | float:
    """Raw SVI (Gatheral 2004): w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))."""
    x = np.asarray(k) - m
    return a + b * (rho * x + np.sqrt(x * x + s * s))


def fit_svi(k: np.ndarray, iv: np.ndarray, T: float) -> np.ndarray:
    """Least-squares fit of raw SVI to one smile; returns (a, b, rho, m, s).

    Residuals are in implied vol, so the objective matches the RMSE we
    report. The no-negative-variance condition a + b s sqrt(1 - rho^2) >= 0
    (the minimum of w) is enforced by optimising over that minimum, v >= 0,
    instead of over a; then every constraint is a simple box bound:
    v >= 0, b >= 0, |rho| < 1, s > 0. A few starting points guard against
    local minima."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)

    def unpack(x: np.ndarray) -> np.ndarray:
        v, b, rho, m, s = x
        return np.array([v - b * s * math.sqrt(1 - rho * rho), b, rho, m, s])

    def resid(x: np.ndarray) -> np.ndarray:
        w = svi_total_variance(k, *unpack(x))
        return np.sqrt(np.maximum(w, 1e-12) / T) - iv

    w_max = float(np.max(iv) ** 2 * T)
    lower = [0.0, 0.0, -0.999, -1.0, 1e-4]
    upper = [w_max, 10.0, 0.999, 1.0, 2.0]
    v0 = float(np.min(iv) ** 2 * T)
    best = None
    for rho0, m0, s0 in [(-0.7, 0.0, 0.1), (-0.3, 0.05, 0.2), (-0.9, -0.05, 0.05)]:
        x0 = [min(v0, 0.99 * w_max), 0.1, rho0, m0, s0]
        res = least_squares(resid, x0, bounds=(lower, upper), x_scale="jac",
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or res.cost < best.cost:
            best = res
    return unpack(best.x)


def svi_vol(k: np.ndarray | float, params: np.ndarray, T: float
            ) -> np.ndarray | float:
    return np.sqrt(np.maximum(svi_total_variance(k, *params), 0.0) / T)


def _rmse_volpts(model: np.ndarray, market: np.ndarray) -> float:
    return float(100 * np.sqrt(np.mean((model - market) ** 2)))


def fit_all_slices(points: pd.DataFrame, min_points: int = 10) -> pd.DataFrame:
    """Per expiry: out-of-sample check (every 5th strike in k order held out),
    then a final fit on all points. One row per expiry."""
    rows = []
    for expiry, g in points.groupby("expiry"):
        g = g.sort_values("k")
        if len(g) < min_points:
            continue
        T = g["T"].iloc[0]
        k, iv = g["k"].to_numpy(), g["iv"].to_numpy()
        held = np.arange(len(g)) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1
        p_train = fit_svi(k[~held], iv[~held], T)
        p_all = fit_svi(k, iv, T)
        rows.append({
            "expiry": expiry, "T": T, "days": T * 365, "n_points": len(g),
            "n_train": int((~held).sum()), "n_test": int(held.sum()),
            "k_min": k.min(), "k_max": k.max(),
            "rmse_in_sample_volpts": _rmse_volpts(svi_vol(k[~held], p_train, T),
                                                  iv[~held]),
            "rmse_out_of_sample_volpts": _rmse_volpts(svi_vol(k[held], p_train, T),
                                                      iv[held]),
            "rmse_full_fit_volpts": _rmse_volpts(svi_vol(k, p_all, T), iv),
            **dict(zip(["a", "b", "rho", "m", "s"], p_all)),
            # Lee (2004): total-variance wing slopes must not exceed 2.
            # Not enforced in the fit, only reported.
            "max_wing_slope": p_all[1] * (1 + abs(p_all[2])),
        })
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def calendar_violations(fits: pd.DataFrame, n_grid: int = 51,
                        tol: float = 1e-6) -> tuple[int, int, list[str]]:
    """Count grid points where total variance falls from one expiry to the
    next (calendar arbitrage). For each adjacent pair the k-grid covers only
    the overlap of the two slices' data ranges, so SVI extrapolation is not
    tested. Returns (violations, grid points checked, offending pairs)."""
    bad, total, pairs = 0, 0, []
    for (_, a), (_, b) in zip(fits.iloc[:-1].iterrows(), fits.iloc[1:].iterrows()):
        lo, hi = max(a.k_min, b.k_min), min(a.k_max, b.k_max)
        if lo >= hi:
            continue
        grid = np.linspace(lo, hi, n_grid)
        w_a = svi_total_variance(grid, a.a, a.b, a.rho, a.m, a.s)
        w_b = svi_total_variance(grid, b.a, b.b, b.rho, b.m, b.s)
        n = int(np.sum(w_b < w_a - tol))
        bad += n
        total += n_grid
        if n:
            pairs.append(f"{a.expiry}->{b.expiry} ({n}/{n_grid})")
    return bad, total, pairs


def surface_vol(k: np.ndarray | float, T: float, fits: pd.DataFrame
                ) -> np.ndarray | float:
    """Implied vol at (k, T), linear in total variance between the two
    neighbouring SVI slices (flat-vol extrapolation outside the T range)."""
    Ts = fits["T"].to_numpy()
    params = fits[["a", "b", "rho", "m", "s"]].to_numpy()
    if T <= Ts[0]:
        return svi_vol(k, params[0], Ts[0])
    if T >= Ts[-1]:
        return svi_vol(k, params[-1], Ts[-1])
    i = int(np.searchsorted(Ts, T)) - 1
    w0 = svi_total_variance(k, *params[i])
    w1 = svi_total_variance(k, *params[i + 1])
    lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
    return np.sqrt(np.maximum((1 - lam) * w0 + lam * w1, 0.0) / T)


def _nearest_expiries(df: pd.DataFrame, target_days: tuple[int, ...]
                      ) -> list[str]:
    """Distinct expiries closest to each target maturity (in days)."""
    days = df.groupby("expiry")["T"].first() * 365
    out = []
    for t in target_days:
        e = (days - t).abs().idxmin()
        if e not in out:
            out.append(e)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plot_smiles(points: pd.DataFrame, fits: pd.DataFrame, source: str,
                 date: str) -> None:
    chosen = _nearest_expiries(fits, (14, 30, 60, 91, 182, 365, 730))
    palette = [c for c in COLORS if c != "#F0E442"]   # yellow is too faint
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for color, expiry in zip(palette, chosen):
        g = points[points["expiry"] == expiry]
        f = fits[fits["expiry"] == expiry].iloc[0]
        grid = np.linspace(g["k"].min(), g["k"].max(), 200)
        vol = svi_vol(grid, f[["a", "b", "rho", "m", "s"]].to_numpy(float),
                      f["T"])
        ax.scatter(g["k"], 100 * g["iv"], s=7, color=color, alpha=0.45)
        ax.plot(grid, 100 * vol, color=color, lw=1.6,
                label=f"{expiry} ({f['days']:.0f}d)")
    ax.axvline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("log-moneyness  k = ln(K / F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"SPY implied-vol smiles, {date}: market (dots) vs SVI fit (lines)")
    ax.legend(title="expiry (days to expiry)", ncol=2)
    save(fig, "04_iv_smiles", source)


def _plot_surface(fits: pd.DataFrame, source: str, date: str) -> None:
    # Geometric T grid: the smile changes fastest at short maturities.
    T_grid = np.geomspace(fits["T"].iloc[0], fits["T"].iloc[-1], 80)
    m_grid = np.linspace(0.80, 1.10, 50)   # K / F, inside the quoted range
    vol = np.array([surface_vol(np.log(m_grid), T, fits) for T in T_grid])
    M, TT = np.meshgrid(m_grid, T_grid)
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(M, TT, 100 * vol, cmap="viridis", linewidth=0,
                           antialiased=True, rcount=80, ccount=50)
    ax.set_xlabel("moneyness K / F")
    ax.set_ylabel("maturity (years)")
    ax.set_zlabel("implied vol (%)")
    ax.view_init(elev=22, azim=-55)
    ax.set_title(f"SPY implied-volatility surface (SVI), {date}")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="implied vol (%)")
    save(fig, "05_iv_surface_3d", source)


def _plot_term_structure(term: pd.DataFrame, source: str, date: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(term["T"], 100 * term["atm_vol"], "o-", ms=4)
    ax1.set_xlabel("maturity (years)")
    ax1.set_ylabel("ATM implied vol, k = 0 (%)")
    ax1.set_title("ATM term structure")
    ok = term["skew_in_data_range"]
    ax2.plot(term.loc[ok, "T"], term.loc[ok, "skew_90_110_volpts"], "o-", ms=4,
             color=COLORS[1], label="SVI, both strikes inside quoted range")
    if (~ok).any():
        ax2.plot(term.loc[~ok, "T"], term.loc[~ok, "skew_90_110_volpts"], "o",
                 ms=4, mfc="none", color=COLORS[1],
                 label="SVI, one strike extrapolated")
        ax2.legend()
    ax2.set_xlabel("maturity (years)")
    ax2.set_ylabel("IV(K/F = 0.9) - IV(K/F = 1.1)  (vol points)")
    ax2.set_title("Skew: 90% minus 110% moneyness")
    fig.suptitle(f"SPY ATM volatility and skew by maturity, {date}")
    save(fig, "06_atm_term_structure_skew", source)


def term_structure(fits: pd.DataFrame) -> pd.DataFrame:
    """ATM vol, 90/110 skew and ATM slope d sigma / dk per fitted slice."""
    rows = []
    k90, k110 = math.log(0.9), math.log(1.1)
    for f in fits.itertuples():
        p = np.array([f.a, f.b, f.rho, f.m, f.s])
        atm = float(svi_vol(0.0, p, f.T))
        # w'(k) = b (rho + (k - m) / sqrt((k - m)^2 + s^2));
        # sigma = sqrt(w / T)  =>  d sigma / dk = w'(k) / (2 sigma T).
        w_prime = f.b * (f.rho - f.m / math.sqrt(f.m ** 2 + f.s ** 2))
        rows.append({
            "expiry": f.expiry, "T": f.T, "atm_vol": atm,
            "atm_slope_dsigma_dk": w_prime / (2 * atm * f.T),
            "skew_90_110_volpts": 100 * float(svi_vol(k90, p, f.T)
                                              - svi_vol(k110, p, f.T)),
            "skew_in_data_range": f.k_min <= k90 and f.k_max >= k110,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run() -> dict:
    """Build the surface from the cached snapshot; write figures/results and
    return the headline summary."""
    chain, meta = data.spy_option_chain()           # cached, never refreshed
    snapshot = pd.Timestamp(meta["snapshot_utc"])
    S = float(meta["spot"])
    curve, q = market_inputs(snapshot)
    snap_txt = snapshot.strftime("%Y-%m-%d %H:%M UTC")
    date = snapshot.strftime("%Y-%m-%d")
    source = (f"Yahoo Finance SPY option chain snapshot {snap_txt}; "
              "FRED Treasury yields")
    RESULTS_DIR.mkdir(exist_ok=True)

    points, table, quotes = clean_chain(chain, snapshot, S, q, curve)
    table.to_csv(RESULTS_DIR / "chain_cleaning.csv", index=False)
    # Rows dropped at the last cleaning step are exactly the solver failures.
    method_counts = {m: int((points["iv_method"] == m).sum())
                     for m in ("newton", "brent")}
    method_counts["failed"] = int(table["rows_remaining"].iloc[-2]
                                  - table["rows_remaining"].iloc[-1])
    points.drop(columns=["contractSize", "currency", "change",
                         "percentChange"]).to_csv(
        RESULTS_DIR / "iv_points.csv", index=False)

    parity = parity_check(quotes, S)
    parity.to_csv(RESULTS_DIR / "parity_check.csv", index=False)

    t0 = time.perf_counter()
    amer = american_iv_diagnostic(points, S, q)
    amer.to_csv(RESULTS_DIR / "american_iv_diagnostic.csv", index=False)
    log.info("American IV diagnostic took %.1fs", time.perf_counter() - t0)

    fits = fit_all_slices(points)
    fits.to_csv(RESULTS_DIR / "svi_fit.csv", index=False)
    n_bad, n_checked, bad_pairs = calendar_violations(fits)
    term = term_structure(fits)
    term.to_csv(RESULTS_DIR / "term_structure.csv", index=False)

    _plot_smiles(points, fits, source, date)
    _plot_surface(fits, source, date)
    _plot_term_structure(term, source, date)

    def headline(T: float) -> tuple[float, float]:
        atm = float(surface_vol(0.0, T, fits))
        skew = float(surface_vol(math.log(0.9), T, fits)
                     - surface_vol(math.log(1.1), T, fits))
        return 100 * atm, 100 * skew

    atm_1m, skew_1m = headline(1 / 12)
    atm_1y, skew_1y = headline(1.0)
    yahoo_gap = 100 * (points["iv"] - points["impliedVolatility"]).abs()
    summary = {
        "snapshot_utc": meta["snapshot_utc"],
        "spot": S,
        "dividend_yield_q": q,
        "r_range": [float(points["r"].min()), float(points["r"].max())],
        "n_raw": int(table["rows_remaining"].iloc[0]),
        "n_clean": len(points),
        "n_expiries_fitted": len(fits),
        "atm_vol_1m_pct": atm_1m,
        "atm_vol_1y_pct": atm_1y,
        "skew_90_110_1m_volpts": skew_1m,
        "skew_90_110_1y_volpts": skew_1y,
        "svi_median_rmse_in_sample_volpts":
            float(fits["rmse_in_sample_volpts"].median()),
        "svi_median_rmse_out_of_sample_volpts":
            float(fits["rmse_out_of_sample_volpts"].median()),
        "iv_solver_counts": method_counts,
        "parity_median_diff_bp": float(parity["diff_bp"].median()),
        "parity_median_implied_q": float(parity["q_implied"].median()),
        "american_minus_bsm_iv_median_abs_volpts":
            float(amer["amer_minus_bsm_volpts"].abs().median()),
        "american_minus_bsm_iv_max_abs_volpts":
            float(amer["amer_minus_bsm_volpts"].abs().max()),
        "american_minus_euro_tree_iv_max_abs_volpts":
            float(amer["amer_minus_euro_tree_volpts"].abs().max()),
        "svi_slices_wing_slope_above_2": int((fits["max_wing_slope"] > 2).sum()),
        "calendar_arb_violations": n_bad,
        "calendar_arb_grid_points_checked": n_checked,
        "calendar_arb_pairs": bad_pairs,
        "yahoo_iv_median_abs_diff_volpts": float(yahoo_gap.median()),
    }
    (RESULTS_DIR / "surface_summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=2))

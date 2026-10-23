"""Numerical-methods report: CRR tree convergence, the American put
early-exercise premium, and Monte Carlo variance reduction, all checked
against the closed-form BSM price.

    python -c "from src import pricing_report; print(pricing_report.run())"

Writes figures/01-03 and results/{crr_convergence, american_put_premium,
mc_variance_reduction}.csv. No market data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import binomial, bsm, montecarlo
from src.plotting import COLORS, RESULTS_DIR, save

SOURCE = "Model output (no market data)"

# Representative case used for the convergence studies.
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.04, q=0.01, sigma=0.20)
BASE_LABEL = "S=K=100, T=1y, r=4%, q=1%, σ=20%"

# Hull, Example 21.1: American put on a non-dividend stock.
HULL_PUT = dict(S=50.0, K=50.0, T=5 / 12, r=0.10, q=0.0, sigma=0.40)


def _crr_convergence(fig_prefix: str) -> dict:
    """Figure 01 and crr_convergence.csv."""
    n_grid = np.arange(5, 501)
    bs_call = float(bsm.price(**BASE, kind="call"))
    bs_put = float(bsm.price(**BASE, kind="put"))
    err_call = np.array([binomial.crr_price(**BASE, n_steps=n, kind="call")
                         for n in n_grid]) - bs_call
    # Average of n and n+1 steps, from the arrays already computed.
    err_call_smooth = 0.5 * (err_call[:-1] + err_call[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    # Plot even and odd n separately: joined up, they alternate in sign every
    # step and would render as a solid band. Only the call is shown because
    # the European put error is identical: the tree prices the forward
    # exactly, so CRR call - CRR put = S e^{-qT} - K e^{-rT} for every n.
    even = n_grid % 2 == 0
    ax1.plot(n_grid[even], err_call[even], lw=1.2, label="Even n")
    ax1.plot(n_grid[~even], err_call[~even], lw=1.2, label="Odd n")
    ax1.axhline(0, color="0.3", lw=0.8)
    ax1.set_ylim(-0.1, 0.1)
    ax1.set_xlabel("Number of tree steps n")
    ax1.set_ylabel("CRR price − BSM price ($)")
    ax1.set_title("Signed error of the call: odd/even oscillation")
    ax1.text(0.97, 0.05, "European put error is identical\n"
             "(put-call parity holds exactly on the tree)",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=8,
             color="0.3")
    ax1.legend()

    ax2.loglog(n_grid, np.abs(err_call), lw=0.8, color=COLORS[0],
               label="Call, n steps")
    ax2.loglog(n_grid[:-1], np.abs(err_call_smooth), lw=1.2, color=COLORS[2],
               label="Call, average of n and n+1 steps")
    ax2.loglog(n_grid, 1.0 / n_grid, "--", color="0.3", label="Reference 1/n")
    ax2.set_xlabel("Number of tree steps n (log scale)")
    ax2.set_ylabel("|CRR − BSM| ($, log scale)")
    ax2.set_title("Absolute error of the call: O(1/n) convergence")
    ax2.legend()
    fig.suptitle(f"CRR binomial tree vs Black-Scholes-Merton, European options "
                 f"({BASE_LABEL})")
    save(fig, f"{fig_prefix}01_crr_convergence", SOURCE)

    rows = []
    for n in (10, 50, 100, 500, 1000, 2000):
        euro_put = binomial.crr_price(**BASE, n_steps=n, kind="put")
        rows.append({
            "n_steps": n,
            "euro_put_crr": euro_put,
            "bsm_put": bs_put,
            "euro_put_error": euro_put - bs_put,
            "amer_put_crr": binomial.crr_price(**BASE, n_steps=n, kind="put",
                                               american=True),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "crr_convergence.csv", index=False,
                 float_format="%.6f")
    at_1000 = table.set_index("n_steps").loc[1000]
    return {
        "bsm_call": bs_call,
        "bsm_put": bs_put,
        "crr_call_error_n1000": binomial.crr_price(**BASE, n_steps=1000)
        - bs_call,
        "crr_put_error_n1000": float(at_1000["euro_put_error"]),
        "crr_american_put_n1000": float(at_1000["amer_put_crr"]),
    }


def _american_put_premium(fig_prefix: str) -> dict:
    """Figure 02 and american_put_premium.csv (q = 0, n = 1000 steps)."""
    K, T, sigma, n = 100.0, 1.0, 0.20, 1000
    moneyness = np.linspace(0.6, 1.4, 41)          # S / K
    rates = (0.01, 0.04, 0.08)

    def premium(s_over_k: float, r: float, kind: str) -> float:
        return binomial.early_exercise_premium(s_over_k * K, K, T, r, 0.0,
                                               sigma, n, kind)

    fig, ax = plt.subplots()
    for r in rates:
        ax.plot(moneyness, [premium(m, r, "put") for m in moneyness],
                label=f"Put, r = {r:.0%}")
    # Calls on a non-dividend stock are never exercised early: show it.
    call_prem = [premium(m, 0.08, "call") for m in moneyness]
    ax.plot(moneyness, call_prem, ":", color="0.3",
            label="Call, r = 8% (identically 0)")
    ax.text(1.38, 3.2, "With q = 0 an American call is never\n"
            "exercised early, so its premium is zero\n"
            "for every S/K and r (dotted line).",
            ha="right", fontsize=9, color="0.3")
    ax.set_xlabel("Moneyness S/K")
    ax.set_ylabel("American − European price ($, K = 100)")
    ax.set_title("Early-exercise premium of an American put\n"
                 f"CRR tree, n = {n}, K = 100, T = 1y, σ = 20%, q = 0",
                 fontsize=11)
    ax.legend()
    save(fig, f"{fig_prefix}02_american_put_premium", SOURCE)

    rows = [{"S_over_K": m, "r": r, "american_put": binomial.crr_price(
                m * K, K, T, r, 0.0, sigma, n, "put", american=True),
             "european_put": binomial.crr_price(m * K, K, T, r, 0.0, sigma,
                                                n, "put"),
             } for r in rates for m in (0.8, 0.9, 1.0, 1.1, 1.2)]
    table = pd.DataFrame(rows)
    table["premium"] = table["american_put"] - table["european_put"]
    table.to_csv(RESULTS_DIR / "american_put_premium.csv", index=False,
                 float_format="%.6f")
    return {"max_abs_call_premium_q0": float(np.max(np.abs(call_prem))),
            "atm_put_premium_r4pct": float(table.query(
                "S_over_K == 1.0 and r == 0.04")["premium"].iloc[0])}


def _mc_convergence(fig_prefix: str) -> dict:
    """Figure 03 and mc_variance_reduction.csv (European call, BASE case)."""
    bs_call = float(bsm.price(**BASE, kind="call"))
    n_grid = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000,
                       1_000_000])
    methods = ("plain", "antithetic", "control")
    # Small horizontal offsets (in log space) so the error bars don't overlap.
    offsets = (0.92, 1.0, 1.08)

    fig, ax = plt.subplots()
    for method, off, color in zip(methods, offsets, COLORS):
        res = [montecarlo.mc_price(**BASE, kind="call", n_paths=int(n),
                                   method=method, seed=1) for n in n_grid]
        ax.errorbar(n_grid * off, [x.price for x in res],
                    yerr=[1.96 * x.std_error for x in res], fmt="o-",
                    ms=4, lw=1, capsize=3, color=color, label=method)
    ax.axhline(bs_call, color="0.2", ls="--", lw=1,
               label=f"BSM price {bs_call:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of payoff evaluations (log scale)")
    ax.set_ylabel("European call price estimate ($)")
    ax.set_title("Monte Carlo estimate ± 1.96 standard errors\n"
                 f"{BASE_LABEL}", fontsize=11)
    ax.legend()
    save(fig, f"{fig_prefix}03_mc_convergence", SOURCE)

    # Like-for-like comparison at n = 100,000 payoff evaluations.
    results = {m: montecarlo.mc_price(**BASE, kind="call", n_paths=100_000,
                                      method=m, seed=2)
               for m in montecarlo.METHODS}
    se_plain = results["plain"].std_error
    table = pd.DataFrame([{
        "method": m,
        "price": x.price,
        "std_error": x.std_error,
        "bsm_price": bs_call,
        "abs_error_in_se": abs(x.price - bs_call) / x.std_error,
        "variance_reduction_factor": (se_plain / x.std_error) ** 2,
    } for m, x in results.items()])
    table.to_csv(RESULTS_DIR / "mc_variance_reduction.csv", index=False,
                 float_format="%.6f")
    t = table.set_index("method")
    return {"mc_n_paths": 100_000,
            "mc_se": t["std_error"].round(5).to_dict(),
            "mc_vrf": t["variance_reduction_factor"].round(2).to_dict(),
            "mc_abs_error_in_se": t["abs_error_in_se"].round(2).to_dict()}


def run(fig_prefix: str = "") -> dict:
    """Make all figures and tables; return the headline numbers."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "hull_21_1_american_put_5_steps": binomial.crr_price(
            **HULL_PUT, n_steps=5, kind="put", american=True),
        "hull_21_1_american_put_500_steps": binomial.crr_price(
            **HULL_PUT, n_steps=500, kind="put", american=True),
    }
    out.update(_crr_convergence(fig_prefix))
    out.update(_american_put_premium(fig_prefix))
    out.update(_mc_convergence(fig_prefix))
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in out.items()}

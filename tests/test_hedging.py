"""Tests for the delta-hedging engine. Synthetic prices are used here only;
the study itself uses market data."""

import numpy as np
import pandas as pd
import pytest

from src import bsm
from src.hedging import hedge_pnl, trailing_realised_vol

YEAR = 252


def _gbm(m: int, n: int, sigma: float, r: float, seed: int) -> np.ndarray:
    """m GBM paths of n daily steps starting at 100 (drift = r)."""
    rng = np.random.default_rng(seed)
    dt = 1 / YEAR
    z = rng.standard_normal((m, n))
    logret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return 100 * np.exp(np.hstack([np.zeros((m, 1)), logret.cumsum(axis=1)]))


def test_two_step_cash_account_by_hand():
    S = np.array([[100.0, 102.0, 101.0]])
    D = np.array([[0.0, 0.0, 0.5]])            # ex-dividend at expiry day
    r, q, sig, c = 0.05, 0.01, 0.2, 1e-4
    K, g = 100.0, np.exp(r / YEAR)

    prem = bsm.price(100, K, 2 / YEAR, r, q, sig)
    d0 = bsm.delta(100, K, 2 / YEAR, r, q, sig)
    cash = prem - d0 * 100 - c * d0 * 100
    cash *= g                                   # day 1 interest
    d1 = bsm.delta(102, K, 1 / YEAR, r, q, sig)
    cash -= (d1 - d0) * 102 + c * abs(d1 - d0) * 102
    cash = cash * g + d1 * 0.5                  # day 2 interest + dividend
    expected = cash + d1 * 101 - c * d1 * 101 - max(101 - K, 0)

    res = hedge_pnl(S, D, r, q, sig, sig, h=1, cost=c)
    assert res.pnl[0] == pytest.approx(float(expected), abs=1e-12)
    assert res.premium[0] == pytest.approx(float(prem))
    exp_costs = c * (d0 * 100 + abs(d1 - d0) * 102 + d1 * 101)
    assert res.costs[0] == pytest.approx(float(exp_costs))


def test_components_sum_to_total():
    S = _gbm(200, 21, 0.25, 0.03, seed=1)
    res = hedge_pnl(S, np.zeros_like(S), 0.03, 0.01, 0.22, 0.18, h=5,
                    cost=1e-4)
    total = res.edge + res.theta + res.gamma - res.costs + res.residual
    np.testing.assert_allclose(total, res.pnl, rtol=0, atol=1e-12)
    assert np.all(res.edge > 0)                 # sold at 22, hedged at 18


def test_gbm_daily_hedge_small_and_sqrt_h_scaling():
    sigma, r, n = 0.2, 0.02, 20
    S = _gbm(4000, n, sigma, r, seed=42)
    D = np.zeros_like(S)
    daily = hedge_pnl(S, D, r, 0.0, sigma, sigma, h=1, cost=0.0)
    every4 = hedge_pnl(S, D, r, 0.0, sigma, sigma, h=4, cost=0.0)
    prem = daily.premium.mean()
    # Fair price, hedged 20 times: mean ~ 0 and std ~ sqrt(pi/4)/sqrt(20)
    # ~ 20% of premium (Derman-Kamal).
    assert abs(daily.pnl.mean()) < 0.03 * prem
    assert daily.pnl.std() < 0.3 * prem
    # 20 -> 5 rebalances: std should grow by about sqrt(4) = 2.
    ratio = every4.pnl.std() / daily.pnl.std()
    assert 1.6 < ratio < 2.5
    # The gamma-theta terms explain most of the daily P&L.
    assert daily.residual.std() < 0.35 * daily.pnl.std()


def test_zero_cost_relation():
    S = _gbm(100, 21, 0.3, 0.0, seed=7)
    D = np.zeros_like(S)
    free = hedge_pnl(S, D, 0.0, 0.0, 0.2, 0.2, h=2, cost=0.0)
    paid = hedge_pnl(S, D, 0.0, 0.0, 0.2, 0.2, h=2, cost=5e-4)
    assert np.all(free.costs == 0.0)
    # With r = 0 costs are not compounded: P&L(c=0) = P&L(c) + costs.
    np.testing.assert_allclose(free.pnl, paid.pnl + paid.costs, atol=1e-12)
    # With r > 0 the gap is costs plus at most one month of interest on them.
    free_r = hedge_pnl(S, D, 0.05, 0.0, 0.2, 0.2, h=2, cost=0.0)
    paid_r = hedge_pnl(S, D, 0.05, 0.0, 0.2, 0.2, h=2, cost=5e-4)
    gap = free_r.pnl - paid_r.pnl
    assert np.all(gap >= paid_r.costs - 1e-12)
    assert np.all(gap <= paid_r.costs * np.exp(0.05 * 21 / YEAR) + 1e-12)


def test_trailing_vol_has_no_look_ahead():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(_gbm(1, 59, 0.2, 0.0, seed=3)[0], index=idx)
    div = pd.Series(0.0, index=idx)
    t0 = idx[30]
    base = trailing_realised_vol(close, div).loc[t0]
    bumped = close.copy()
    bumped.iloc[31:] *= 1.5                     # change every future price
    assert trailing_realised_vol(bumped, div).loc[t0] == base
    # ... while a change inside the trailing window does move it.
    moved = close.copy()
    moved.iloc[25] *= 1.05
    assert trailing_realised_vol(moved, div).loc[t0] != base

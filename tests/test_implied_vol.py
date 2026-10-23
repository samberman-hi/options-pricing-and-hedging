"""Implied volatility: round trips, Hull example, bounds, Brent fallback."""

import itertools
import math

import pytest

from src import bsm
from src.implied_vol import implied_vol

S, R, Q = 100.0, 0.03, 0.01


# Implied vol is only identifiable where the price actually responds to sigma.
# Deep in/out-of-the-money short-dated low-vol options (e.g. K=80, T=0.1,
# sigma=10%) have vega ~1e-10: the price equals its no-arbitrage bound to ~15
# digits, so no solver can recover sigma to 1e-6 in double precision. Such
# points are excluded from the round trip.
GRID = [(K, T, sigma, kind) for K, T, sigma, kind in itertools.product(
            (80.0, 90.0, 100.0, 110.0, 120.0), (0.1, 0.5, 2.0),
            (0.1, 0.3, 0.6), ("call", "put"))
        if bsm.vega(S, K, T, R, Q, sigma) > 1e-3]


def test_grid_is_mostly_well_conditioned():
    assert len(GRID) >= 85  # of 90 points, only a few are excluded


@pytest.mark.parametrize("K, T, sigma, kind", GRID)
def test_round_trip(K, T, sigma, kind):
    target = float(bsm.price(S, K, T, R, Q, sigma, kind))
    assert implied_vol(target, S, K, T, R, Q, kind) == pytest.approx(
        sigma, abs=1e-6)


def test_hull_example():
    # Hull: c = 1.875 with S=21, K=20, r=10%, T=0.25 -> sigma = 0.235 (23.5%).
    assert implied_vol(1.875, 21, 20, 0.25, 0.10, 0.0, "call") == \
        pytest.approx(0.235, abs=0.0005)


def test_price_below_intrinsic_is_nan():
    # Lower bound for a put is K e^{-rT} - S e^{-qT} = 5.57 here.
    assert math.isnan(implied_vol(5.0, 90, 100, 0.5, R, Q, "put"))


def test_price_above_upper_bound_is_nan():
    # A call can never be worth more than S e^{-qT}.
    assert math.isnan(implied_vol(S, S, 100, 1.0, R, Q, "call"))


def test_brent_fallback_round_trips():
    # Short-dated, 30% OTM call (price ~ 0.001) started from a poor guess
    # (sigma0 = 2%): vega there is ~0, so Newton bails and Brent takes over.
    K, T, sigma = 130.0, 0.1, 0.25
    target = float(bsm.price(S, K, T, R, Q, sigma, "call"))
    iv, method = implied_vol(target, S, K, T, R, Q, "call", initial=0.02,
                             return_method=True)
    assert method == "brent"
    assert iv == pytest.approx(sigma, abs=1e-6)

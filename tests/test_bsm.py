"""Black-Scholes-Merton: textbook values (Hull), finite-difference Greeks,
put-call parity and limiting cases."""

import itertools

import numpy as np
import pytest

from src import bsm

# Hull quotes results to 2-3 significant decimals, so tolerances are half a
# unit in the last quoted digit.


def test_hull_example_15_6_call_and_put():
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76, p = 0.81.
    args = dict(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20)
    assert bsm.price(**args, kind="call") == pytest.approx(4.76, abs=0.005)
    assert bsm.price(**args, kind="put") == pytest.approx(0.81, abs=0.005)


def test_hull_greeks_example():
    # Hull's running Greeks example (ch. 19): S=49, K=50, r=5%, sigma=20%,
    # T=20 weeks=0.3846. Vega and rho are per 1.00, theta per year.
    g = bsm.greeks(S=49, K=50, T=0.3846, r=0.05, q=0.0, sigma=0.20)
    assert g["price"] == pytest.approx(2.40, abs=0.005)
    assert g["delta"] == pytest.approx(0.522, abs=0.0005)
    assert g["gamma"] == pytest.approx(0.066, abs=0.0005)
    assert g["vega"] == pytest.approx(12.1, abs=0.05)
    assert g["theta"] == pytest.approx(-4.31, abs=0.005)
    assert g["rho"] == pytest.approx(8.91, abs=0.005)


def test_hull_index_option_with_dividend_yield():
    # S=930, K=900, r=8%, q=3%, sigma=20%, T=2 months -> c = 51.83.
    c = bsm.price(S=930, K=900, T=2 / 12, r=0.08, q=0.03, sigma=0.20)
    assert c == pytest.approx(51.83, abs=0.005)


GRID = list(itertools.product(
    (80.0, 100.0, 120.0),   # S
    (0.25, 1.0),            # T
    (0.0, 0.03),            # q
    ("call", "put"),
))


@pytest.mark.parametrize("S, T, q, kind", GRID)
def test_greeks_match_central_differences(S, T, q, kind):
    K, r, sigma = 100.0, 0.05, 0.25

    def V(S=S, T=T, r=r, sigma=sigma):
        return float(bsm.price(S, K, T, r, q, sigma, kind))

    h_s, h = 1e-4 * S, 1e-5
    fd = {
        "delta": (V(S=S + h_s) - V(S=S - h_s)) / (2 * h_s),
        "gamma": (V(S=S + h_s) - 2 * V() + V(S=S - h_s)) / h_s**2,
        "vega": (V(sigma=sigma + h) - V(sigma=sigma - h)) / (2 * h),
        # theta is dV/dt with t calendar time, i.e. -dV/dT.
        "theta": -(V(T=T + h) - V(T=T - h)) / (2 * h),
        "rho": (V(r=r + h) - V(r=r - h)) / (2 * h),
    }
    analytic = bsm.greeks(S, K, T, r, q, sigma, kind)
    for name, value in fd.items():
        assert analytic[name] == pytest.approx(value, rel=1e-5, abs=1e-6), name


def test_put_call_parity_identity():
    S = np.array([70.0, 100.0, 140.0])[:, None]
    K = np.array([80.0, 100.0, 120.0])[None, :]
    args = dict(T=0.75, r=0.04, q=0.02, sigma=0.3)
    c = bsm.price(S, K, kind="call", **args)
    p = bsm.price(S, K, kind="put", **args)
    gap = bsm.parity_gap(c, p, S, K, args["T"], args["r"], args["q"])
    assert np.max(np.abs(gap)) < 1e-10


def test_deep_itm_call_tends_to_discounted_forward_intrinsic():
    S, K, T, r, q = 300.0, 50.0, 1.0, 0.05, 0.02
    c = bsm.price(S, K, T, r, q, 0.2, "call")
    assert c == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_gamma_and_vega_are_the_same_for_calls_and_puts():
    # Put-call parity: C - P does not depend on sigma and is linear in S, so
    # its second S-derivative and its sigma-derivative are both zero.
    for S in (80.0, 100.0, 120.0):
        def fd_gamma(kind, h=0.01):
            v = [float(bsm.price(S + d, 100, 0.5, 0.03, 0.01, 0.2, kind))
                 for d in (h, 0.0, -h)]
            return (v[0] - 2 * v[1] + v[2]) / h**2
        assert fd_gamma("call") == pytest.approx(fd_gamma("put"), rel=1e-4)
        dvol = 1e-5
        vega_c, vega_p = [
            (bsm.price(S, 100, 0.5, 0.03, 0.01, 0.2 + dvol, k)
             - bsm.price(S, 100, 0.5, 0.03, 0.01, 0.2 - dvol, k)) / (2 * dvol)
            for k in ("call", "put")]
        assert vega_c == pytest.approx(vega_p, rel=1e-6)

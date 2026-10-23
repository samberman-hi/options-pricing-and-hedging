"""CRR binomial tree: Hull's American put example, convergence to BSM,
and early-exercise properties."""

import itertools

import pytest

from src import binomial, bsm

# Hull Example 21.1: S=50, K=50, r=10%, sigma=40%, T=5 months, q=0.
HULL = dict(S=50.0, K=50.0, T=5 / 12, r=0.10, q=0.0, sigma=0.40)


def test_hull_american_put_five_steps():
    # Hull Figure 21.3: 5-step tree value 4.49.
    p = binomial.crr_price(**HULL, n_steps=5, kind="put", american=True)
    assert p == pytest.approx(4.49, abs=0.005)


def test_hull_american_put_many_steps():
    # Hull reports 4.283 with 500 steps (4.263 with 30, 4.278 with 100).
    for n, hull in ((30, 4.263), (100, 4.278), (500, 4.283)):
        p = binomial.crr_price(**HULL, n_steps=n, kind="put", american=True)
        assert p == pytest.approx(hull, abs=0.0005), n


@pytest.mark.parametrize("kind, q, K", list(itertools.product(
    ("call", "put"), (0.0, 0.02, 0.05), (90.0, 100.0, 110.0))))
def test_european_converges_to_bsm(kind, q, K):
    args = dict(S=100.0, K=K, T=1.0, r=0.04, q=q, sigma=0.25)
    tree = binomial.crr_price(**args, n_steps=1000, kind=kind)
    assert tree == pytest.approx(bsm.price(**args, kind=kind), abs=0.01)


def test_smoothed_price_is_closer_than_raw():
    args = dict(S=100.0, K=100.0, T=1.0, r=0.04, q=0.01, sigma=0.2)
    exact = float(bsm.price(**args))
    raw = abs(binomial.crr_price(**args, n_steps=100) - exact)
    smooth = abs(binomial.crr_price_smoothed(**args, n_steps=100) - exact)
    assert smooth < raw / 10


@pytest.mark.parametrize("S", (70.0, 100.0, 130.0))
def test_american_call_without_dividends_equals_european(S):
    args = dict(S=S, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.3, n_steps=500)
    amer = binomial.crr_price(**args, kind="call", american=True)
    euro = binomial.crr_price(**args, kind="call", american=False)
    assert amer == pytest.approx(euro, abs=1e-10)


@pytest.mark.parametrize("S, q, kind", list(itertools.product(
    (70.0, 100.0, 130.0), (0.0, 0.04), ("call", "put"))))
def test_american_at_least_european(S, q, kind):
    prem = binomial.early_exercise_premium(S, 100.0, 1.0, 0.05, q, 0.3, 300,
                                           kind)
    assert prem >= -1e-12


def test_american_put_never_below_intrinsic():
    p = binomial.crr_price(60.0, 100.0, 1.0, 0.05, 0.0, 0.2, 300, "put",
                           american=True)
    assert p == pytest.approx(40.0, abs=1e-12)  # deep ITM: exercise now

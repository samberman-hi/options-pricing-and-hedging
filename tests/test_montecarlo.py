"""Monte Carlo: unbiasedness, 1/sqrt(n) error and variance reduction."""

import pytest

from src import bsm, montecarlo

ARGS = dict(S=100.0, K=100.0, T=1.0, r=0.04, q=0.01, sigma=0.2)


@pytest.mark.parametrize("method", montecarlo.METHODS)
@pytest.mark.parametrize("kind", ("call", "put"))
def test_within_four_standard_errors_of_bsm(method, kind):
    res = montecarlo.mc_price(**ARGS, kind=kind, n_paths=200_000,
                              method=method, seed=7)
    exact = float(bsm.price(**ARGS, kind=kind))
    assert abs(res.price - exact) < 4 * res.std_error
    assert res.ci_low < res.price < res.ci_high


@pytest.mark.parametrize("method", montecarlo.METHODS)
def test_standard_error_scales_as_one_over_sqrt_n(method):
    se = [montecarlo.mc_price(**ARGS, n_paths=n, method=method,
                              seed=3).std_error for n in (50_000, 200_000)]
    assert se[1] / se[0] == pytest.approx(0.5, rel=0.05)


def test_variance_reduction_beats_plain():
    se = {m: montecarlo.mc_price(**ARGS, n_paths=100_000, method=m,
                                 seed=11).std_error
          for m in montecarlo.METHODS}
    assert se["antithetic"] < se["plain"]
    assert se["control"] < se["plain"]
    assert se["antithetic+control"] < min(se["antithetic"], se["control"])
    # For an ATM call the payoff is strongly correlated with S_T.
    assert (se["plain"] / se["control"]) ** 2 > 3


def test_antithetic_needs_even_paths():
    with pytest.raises(ValueError):
        montecarlo.mc_price(**ARGS, n_paths=1001, method="antithetic")

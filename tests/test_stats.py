import numpy as np
import pytest

from src import stats


def test_newey_west_lag0_is_iid_se():
    x = np.random.default_rng(0).normal(size=500)
    iid = np.std(x, ddof=0) / np.sqrt(x.size)
    assert stats.newey_west_se(x, lag=0) == pytest.approx(iid, rel=1e-12)


def test_newey_west_lag1_hand_example():
    # x = 1,2,3,4: deviations -1.5,-0.5,0.5,1.5
    # gamma0 = 5/4 = 1.25; gamma1 = (0.75 - 0.25 + 0.75)/4 = 0.3125
    # S = 1.25 + 2 * (1 - 1/2) * 0.3125 = 1.5625;  SE = sqrt(S/4) = 0.625
    assert stats.newey_west_se([1, 2, 3, 4], lag=1) == pytest.approx(0.625)


def test_hac_ols_on_constant_matches_mean_se():
    x = np.random.default_rng(1).normal(size=300).cumsum() * 0.1
    res = stats.hac_ols(x, np.ones((x.size, 1)), lag=5)
    assert res["beta"][0] == pytest.approx(x.mean())
    assert res["se"][0] == pytest.approx(stats.newey_west_se(x, lag=5))


def test_hac_ols_recovers_coefficients():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.1, size=x.size)
    res = stats.hac_ols(y, np.column_stack([np.ones_like(x), x]), lag=3)
    np.testing.assert_allclose(res["beta"], [1.0, 2.0], atol=0.02)
    assert res["r2"] > 0.99


def test_mean_test_and_nonoverlap():
    x = np.random.default_rng(3).normal(loc=0.0, size=2100)
    mt = stats.mean_test(x, lag=20)
    assert abs(mt["t"]) < 4 and 0 <= mt["p"] <= 1
    no = stats.nonoverlap_ttest(x, step=21)
    assert no["n"] == 100


def test_moments_of_normal_sample():
    x = np.random.default_rng(4).normal(size=20000)
    m = stats.moments(x)
    assert abs(m["skew"]) < 0.1 and abs(m["kurt"]) < 0.15
    assert m["jb_p"] > 0.001

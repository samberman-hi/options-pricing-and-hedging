"""Small statistics toolkit: HAC (Hansen-Hodrick / Newey-West) inference
and distribution moments.

Why HAC?  Consecutive hedging windows start one trading day apart but each
lasts 21 days, so window i and window i+k share 21-k daily returns. Their
P&Ls are therefore autocorrelated up to lag 20 and the iid standard error
of the mean is too small. Newey-West (1987) replaces the variance of the
mean, gamma_0 / n, by the long-run variance

    S = gamma_0 + 2 * sum_{l=1}^{L} w_l * gamma_l,   w_l = 1 - l/(L+1)

(Bartlett kernel, which keeps S >= 0), where
gamma_l = (1/n) sum_t (x_t - xbar)(x_{t-l} - xbar).  SE(mean) = sqrt(S/n).
With L = 0 this is the usual iid standard error (with a 1/n variance).

Kernel choice. For P&Ls of overlapping 21-day windows the autocorrelation
is (roughly) a triangle that reaches zero at lag 21. Bartlett weights
down-weight exactly those lags, so Newey-West with L = 20 captures only
about two-thirds of the long-run variance and overstates t-stats by ~20%.
Hansen-Hodrick (1980) - the textbook estimator for overlapping returns -
uses uniform weights w_l = 1 for l <= L = overlap. It is not guaranteed to
be positive; if it ever is not, we fall back to Newey-West with lag 2L.
The hedging study therefore uses kernel="uniform", L = 20.

p-values use the normal approximation (the HAC t-stat is only
asymptotically normal). No small-sample degrees-of-freedom correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

# Overlapping 21-day windows -> autocorrelation up to lag 20.
DEFAULT_LAG = 20


def _bartlett(L: int) -> np.ndarray:
    """Bartlett weights w_l = 1 - l/(L+1) for l = 1..L."""
    lags = np.arange(1, L + 1)
    return 1.0 - lags / (L + 1.0)


def _weights(L: int, kernel: str) -> np.ndarray:
    if kernel == "bartlett":
        return _bartlett(L)
    if kernel == "uniform":
        return np.ones(L)
    raise ValueError(f"unknown kernel {kernel!r}")


def newey_west_se(x: ArrayLike, lag: int = DEFAULT_LAG,
                  kernel: str = "bartlett") -> float:
    """HAC standard error of the sample mean of x: Newey-West (Bartlett
    kernel, default) or Hansen-Hodrick (kernel="uniform")."""
    x = np.asarray(x, float)
    n = x.size
    u = x - x.mean()
    s = u @ u / n                                    # gamma_0
    for l, w in zip(range(1, lag + 1), _weights(lag, kernel)):
        s += 2.0 * w * (u[l:] @ u[:-l]) / n          # 2 w_l gamma_l
    if s <= 0:                                       # HH can go negative
        return newey_west_se(x, 2 * lag, "bartlett")
    return float(np.sqrt(s / n))


def mean_test(x: ArrayLike, lag: int = DEFAULT_LAG,
              kernel: str = "bartlett") -> dict[str, float]:
    """Mean of x with HAC standard error, t-stat for H0: mean = 0 and
    two-sided normal p-value."""
    x = np.asarray(x, float)
    m = float(x.mean())
    se = newey_west_se(x, lag, kernel)
    t = m / se
    return {"mean": m, "se": se, "t": t, "p": float(2 * sps.norm.sf(abs(t)))}


def nonoverlap_ttest(x: ArrayLike, step: int = 21) -> dict[str, float]:
    """Robustness check: keep every `step`-th observation so the windows do
    not overlap, then run an ordinary one-sample t-test (t distribution,
    n-1 degrees of freedom)."""
    sub = np.asarray(x, float)[::step]
    res = sps.ttest_1samp(sub, 0.0)
    return {"n": int(sub.size), "mean": float(sub.mean()),
            "t": float(res.statistic), "p": float(res.pvalue)}


def nonoverlap_all_offsets(x: ArrayLike, step: int = 21) -> dict[str, float]:
    """The non-overlapping test depends on which window we start from; run
    it for every offset 0..step-1 and summarise the p-values."""
    x = np.asarray(x, float)
    ps = np.array([sps.ttest_1samp(x[o::step], 0.0).pvalue
                   for o in range(step)])
    return {"p_median": float(np.median(ps)), "p_min": float(ps.min()),
            "p_max": float(ps.max())}


def hac_ols(y: ArrayLike, X: ArrayLike, lag: int = DEFAULT_LAG,
            kernel: str = "bartlett") -> dict[str, np.ndarray | float]:
    """OLS of y on X (X must already contain a constant column if wanted)
    with HAC covariance (Newey-West, or Hansen-Hodrick if kernel="uniform")

        Cov(b) = (X'X)^{-1} [ sum_l w_l (G_l + G_l') ] (X'X)^{-1},
        G_l = sum_t u_t u_{t-l} x_t x_{t-l}',   w_0 = 1/2 so G_0 counts once.

    Returns coefficients, HAC standard errors, covariance matrix and R^2."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]                              # rows: x_t u_t
    meat = xu.T @ xu                                 # G_0
    for l, w in zip(range(1, lag + 1), _weights(lag, kernel)):
        g = xu[l:].T @ xu[:-l]
        meat += w * (g + g.T)
    cov = xtx_inv @ meat @ xtx_inv
    r2 = 1.0 - (u @ u) / np.sum((y - y.mean()) ** 2)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "cov": cov,
            "r2": float(r2), "n": int(y.size)}


def moments(x: ArrayLike) -> dict[str, float]:
    """Skewness, excess kurtosis and the Jarque-Bera normality test.
    (JB assumes iid data; with overlapping windows treat its p-value as
    descriptive.)"""
    x = np.asarray(x, float)
    jb = sps.jarque_bera(x)
    return {"skew": float(sps.skew(x)),
            "kurt": float(sps.kurtosis(x)),          # Fisher: normal = 0
            "jb": float(jb.statistic), "jb_p": float(jb.pvalue)}

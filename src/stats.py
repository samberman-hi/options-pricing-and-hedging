"""Small statistics toolkit: HAC (Newey-West) inference and distribution
moments.

Why HAC?  Consecutive hedging windows start one trading day apart but each
lasts 21 days, so window i and window i+k share 21-k daily returns. Their
P&Ls are therefore autocorrelated up to lag 20 and the iid standard error
of the mean is too small. Newey-West (1987) replaces the variance of the
mean, gamma_0 / n, by the long-run variance

    S = gamma_0 + 2 * sum_{l=1}^{L} w_l * gamma_l,   w_l = 1 - l/(L+1)

(Bartlett kernel, which keeps S >= 0), where
gamma_l = (1/n) sum_t (x_t - xbar)(x_{t-l} - xbar).  SE(mean) = sqrt(S/n).
With L = 0 this is the usual iid standard error (with a 1/n variance).

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


def newey_west_se(x: ArrayLike, lag: int = DEFAULT_LAG) -> float:
    """Newey-West HAC standard error of the sample mean of x."""
    x = np.asarray(x, float)
    n = x.size
    u = x - x.mean()
    s = u @ u / n                                    # gamma_0
    for l, w in zip(range(1, lag + 1), _bartlett(lag)):
        s += 2.0 * w * (u[l:] @ u[:-l]) / n          # 2 w_l gamma_l
    return float(np.sqrt(s / n))


def mean_test(x: ArrayLike, lag: int = DEFAULT_LAG) -> dict[str, float]:
    """Mean of x with HAC standard error, t-stat for H0: mean = 0 and
    two-sided normal p-value."""
    x = np.asarray(x, float)
    m = float(x.mean())
    se = newey_west_se(x, lag)
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


def hac_ols(y: ArrayLike, X: ArrayLike, lag: int = DEFAULT_LAG
            ) -> dict[str, np.ndarray | float]:
    """OLS of y on X (X must already contain a constant column if wanted)
    with Newey-West covariance

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
    for l, w in zip(range(1, lag + 1), _bartlett(lag)):
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

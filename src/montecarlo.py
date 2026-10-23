"""Monte Carlo pricing of European options under geometric Brownian motion,
with antithetic and control-variate variance reduction.

Under the risk-neutral measure the terminal price is exactly log-normal,

    S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),

so a European payoff needs one normal draw per path and no time stepping.
The price estimate is the sample mean of the discounted payoff e^{-rT} f(S_T)
and its standard error is sample std / sqrt(number of i.i.d. samples).

Methods (all use exactly n_paths payoff evaluations, so they are compared at
equal cost):
    plain              n_paths independent draws.
    antithetic         n_paths / 2 draws Z, each also used as -Z.
    control            plain draws, with the discounted stock e^{-rT} S_T as a
                       control variate (its mean S e^{-qT} is known exactly).
    antithetic+control both: the control is applied to the pair averages.

Same parameter conventions as src/bsm.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("plain", "antithetic", "control", "antithetic+control")


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    ci_low: float     # 95% confidence interval: price -/+ 1.96 SE
    ci_high: float
    n_paths: int      # payoff evaluations (not independent samples)
    method: str


def _terminal_price(S: float, T: float, r: float, q: float, sigma: float,
                    z: np.ndarray) -> np.ndarray:
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def _payoff(ST: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _apply_control(y: np.ndarray, x: np.ndarray, x_mean: float) -> np.ndarray:
    """Return y - b* (x - E[x]) with b* = Cov(y, x) / Var(x).

    b* is estimated from the same sample used for pricing. This makes the
    estimator very slightly biased (the bias is O(1/n) and negligible next
    to the O(1/sqrt(n)) standard error); a pilot run would remove it.
    """
    b = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - b * (x - x_mean)


def mc_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             kind: str = "call", n_paths: int = 100_000,
             method: str = "plain", seed: int | None = 0) -> MCResult:
    """Monte Carlo price of a European option; see module docstring."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    antithetic = method.startswith("antithetic")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic methods need an even n_paths")

    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        st_up = _terminal_price(S, T, r, q, sigma, z)
        st_dn = _terminal_price(S, T, r, q, sigma, -z)
        # The two halves of a pair are negatively correlated, NOT independent,
        # so the n draws cannot be treated as n i.i.d. samples. The pair
        # averages are i.i.d., so the mean and SE are taken over those n/2.
        y = 0.5 * disc * (_payoff(st_up, K, kind) + _payoff(st_dn, K, kind))
        x = 0.5 * disc * (st_up + st_dn)
    else:
        st = _terminal_price(S, T, r, q, sigma, rng.standard_normal(n_paths))
        y = disc * _payoff(st, K, kind)
        x = disc * st

    if method.endswith("control"):
        # E[e^{-rT} S_T] = S e^{-qT} under the risk-neutral measure (this also
        # holds for the antithetic pair average of the control).
        y = _apply_control(y, x, S * np.exp(-q * T))

    price = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(y.size))
    return MCResult(price, se, price - 1.96 * se, price + 1.96 * se,
                    n_paths, method)

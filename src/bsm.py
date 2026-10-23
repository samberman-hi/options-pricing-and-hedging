"""Black-Scholes-Merton pricing and Greeks for European options with a
continuous dividend yield.

Conventions (used throughout the project):
    S      spot price of the underlying
    K      strike
    T      time to expiry in years
    r      continuously compounded risk-free rate (annual)
    q      continuous dividend yield (annual)
    sigma  volatility (annual, as a decimal: 0.20 = 20%)
    kind   "call" or "put"

All functions accept scalars or NumPy arrays and broadcast.

Greek units:
    delta  dV/dS
    gamma  d2V/dS2
    vega   dV/dsigma   per 1.00 change in sigma (divide by 100 for "per vol point")
    theta  dV/dt       per year, calendar decay (divide by 365 for "per day")
    rho    dV/dr       per 1.00 change in r (divide by 100 for "per 1%")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

Array = np.ndarray | float


def _is_call(kind: str) -> bool:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind == "call"


def d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
          q: ArrayLike, sigma: ArrayLike) -> tuple[Array, Array]:
    """Return (d1, d2) of the BSM formula."""
    S, K, T, r, q, sigma = map(np.asarray, (S, K, T, r, q, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
          q: ArrayLike, sigma: ArrayLike, kind: str = "call") -> Array:
    """BSM price:  C = S e^{-qT} N(d1) - K e^{-rT} N(d2),
                   P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)."""
    S, K, T, r, q = map(np.asarray, (S, K, T, r, q))
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    disc_s = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)
    if _is_call(kind):
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def delta(S, K, T, r, q, sigma, kind: str = "call") -> Array:
    """Call: e^{-qT} N(d1);  put: e^{-qT} (N(d1) - 1)."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    dq = np.exp(-np.asarray(q) * np.asarray(T))
    return dq * norm.cdf(d1) if _is_call(kind) else dq * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, q, sigma) -> Array:
    """Same for calls and puts: e^{-qT} n(d1) / (S sigma sqrt(T))."""
    S, T, q, sigma = map(np.asarray, (S, T, q, sigma))
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, q, sigma) -> Array:
    """Same for calls and puts: S e^{-qT} n(d1) sqrt(T)  (per 1.00 of vol)."""
    S, T, q = map(np.asarray, (S, T, q))
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, q, sigma, kind: str = "call") -> Array:
    """dV/dt (per year) - the change in value as calendar time passes,
    i.e. minus the derivative with respect to time-to-expiry T."""
    S, K, T, r, q, sigma = map(np.asarray, (S, K, T, r, q, sigma))
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if _is_call(kind):
        return (decay + q * S * np.exp(-q * T) * norm.cdf(d1)
                - r * K * np.exp(-r * T) * norm.cdf(d2))
    return (decay - q * S * np.exp(-q * T) * norm.cdf(-d1)
            + r * K * np.exp(-r * T) * norm.cdf(-d2))


def rho(S, K, T, r, q, sigma, kind: str = "call") -> Array:
    """Call: K T e^{-rT} N(d2);  put: -K T e^{-rT} N(-d2)."""
    K, T, r = map(np.asarray, (K, T, r))
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    if _is_call(kind):
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def greeks(S, K, T, r, q, sigma, kind: str = "call") -> dict[str, Array]:
    """All five Greeks plus the price, in the units documented above."""
    return {
        "price": price(S, K, T, r, q, sigma, kind),
        "delta": delta(S, K, T, r, q, sigma, kind),
        "gamma": gamma(S, K, T, r, q, sigma),
        "vega": vega(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, kind),
        "rho": rho(S, K, T, r, q, sigma, kind),
    }


def parity_gap(call: ArrayLike, put: ArrayLike, S: ArrayLike, K: ArrayLike,
               T: ArrayLike, r: ArrayLike, q: ArrayLike) -> Array:
    """Put-call parity residual for European options:
        (C - P) - (S e^{-qT} - K e^{-rT}),
    which is zero under no arbitrage. For American options (e.g. listed SPY
    options) parity only holds as the inequality
        S e^{-qT} - K <= C - P <= S - K e^{-rT}   (Hull, ch. 11),
    so a small non-zero gap is expected there."""
    call, put, S, K, T, r, q = map(np.asarray, (call, put, S, K, T, r, q))
    return (call - put) - (S * np.exp(-q * T) - K * np.exp(-r * T))

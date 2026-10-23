"""Implied volatility: Newton-Raphson on vega with a Brent fallback.

Newton converges quadratically near the root but can overshoot when vega is
tiny (deep in/out of the money, very short expiry). Each Newton step is
therefore checked: if it leaves the bracket [SIGMA_LO, SIGMA_HI], produces a
non-finite value, or fails to converge within MAX_NEWTON iterations, we fall
back to Brent's method, which is guaranteed to converge because the BSM price
is continuous and strictly increasing in sigma.

Prices outside the no-arbitrage bounds have no implied volatility and return
NaN rather than a misleading number.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from src import bsm

SIGMA_LO = 1e-4
SIGMA_HI = 5.0
MAX_NEWTON = 50
TOL = 1e-10


def no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float,
                        kind: str) -> tuple[float, float]:
    """European price bounds: max(intrinsic on forwards, 0) < V < upper."""
    disc_s, disc_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if kind == "call":
        return max(disc_s - disc_k, 0.0), disc_s
    return max(disc_k - disc_s, 0.0), disc_k


def implied_vol(target: float, S: float, K: float, T: float, r: float,
                q: float, kind: str = "call", initial: float | None = None,
                return_method: bool = False):
    """Solve bsm.price(sigma) = target for sigma.

    Returns NaN if the target is outside the no-arbitrage bounds. With
    return_method=True returns (sigma, "newton" | "brent" | "none").
    """
    lo, hi = no_arbitrage_bounds(S, K, T, r, q, kind)
    if not (np.isfinite(target) and lo < target < hi) or T <= 0:
        return (math.nan, "none") if return_method else math.nan

    def f(sig: float) -> float:
        return float(bsm.price(S, K, T, r, q, sig, kind)) - target

    # Starting point. Away from the money use Manaster-Koehler (1982):
    # sigma0 = sqrt(2 |ln(F/K)| / T), the inflection point of price(sigma),
    # from which Newton converges monotonically. Near the money that is ~0,
    # so use the Brenner-Subrahmanyam ATM approximation instead.
    if initial is None:
        mk = math.sqrt(2.0 * abs(math.log(S / K) + (r - q) * T) / T)
        bs = math.sqrt(2.0 * math.pi / T) * target / S
        initial = min(max(mk if mk > 0.05 else bs, 0.05), 2.0)
    sig = initial
    for _ in range(MAX_NEWTON):
        diff = f(sig)
        v = float(bsm.vega(S, K, T, r, q, sig))
        # Accept only if the price matches AND the implied step in sigma is
        # tiny: with near-zero vega a small price error can hide a large
        # sigma error, so a price tolerance alone is not enough.
        if abs(diff) < TOL and v > 0 and abs(diff / v) < 1e-8:
            return (sig, "newton") if return_method else sig
        if v < 1e-8:
            break
        step = sig - diff / v
        if not (np.isfinite(step) and SIGMA_LO < step < SIGMA_HI):
            break
        sig = step

    # Brent fallback: the root is bracketed because f is increasing in sigma.
    if f(SIGMA_LO) * f(SIGMA_HI) > 0:
        return (math.nan, "none") if return_method else math.nan
    sig = brentq(f, SIGMA_LO, SIGMA_HI, xtol=1e-12, maxiter=200)
    return (sig, "brent") if return_method else sig


def implied_vol_vector(targets, S, K, T, r, q, kinds) -> np.ndarray:
    """Element-wise implied_vol over equal-length (or broadcastable) inputs."""
    b = np.broadcast_arrays(np.asarray(targets, float), np.asarray(S, float),
                            np.asarray(K, float), np.asarray(T, float),
                            np.asarray(r, float), np.asarray(q, float),
                            np.asarray(kinds, object))
    return np.array([implied_vol(*args) for args in zip(*(a.ravel() for a in b))]
                    ).reshape(b[0].shape)

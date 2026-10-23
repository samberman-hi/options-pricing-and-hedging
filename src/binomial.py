"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

The tree splits [0, T] into n steps of length dt = T / n. Over each step the
stock moves up by u or down by d, with

    u = e^{sigma sqrt(dt)},   d = 1 / u,
    p = (e^{(r - q) dt} - d) / (u - d)     (risk-neutral up probability),

so that the expected growth of the stock over a step is e^{(r - q) dt}.
Because d = 1/u the tree recombines: after i steps the node reached by j up
moves has price S u^j d^{i-j} = S u^{2j - i}, so step i has only i + 1 nodes.

Pricing is by backward induction from the payoff at expiry:

    V_i,j = e^{-r dt} [p V_{i+1,j+1} + (1 - p) V_{i+1,j}],

and, for an American option, V_i,j = max(continuation, exercise value).
Each time step is one NumPy operation on the whole layer of nodes, so the
only Python loop is over the n time steps (n = 2000 takes a few milliseconds).

Same parameter conventions as src/bsm.py.
"""

from __future__ import annotations

import numpy as np


def _payoff(S: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(S - K, 0.0)
    if kind == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def crr_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
              n_steps: int, kind: str = "call", american: bool = False) -> float:
    """Price a European or American option on an n_steps CRR tree."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        # Happens only if dt is so large that |r - q| sqrt(dt) > sigma.
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1); "
                         "use more steps")
    disc = np.exp(-r * dt)

    # Terminal layer: j = 0..n up moves, stock = S u^{2j - n}.
    j = np.arange(n_steps + 1)
    values = _payoff(S * u ** (2 * j - n_steps), K, kind)

    # Roll back one layer at a time; layer i has i + 1 nodes.
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            stock = S * u ** (2 * np.arange(i + 1) - i)
            values = np.maximum(values, _payoff(stock, K, kind))
    return float(values[0])


def crr_price_smoothed(S: float, K: float, T: float, r: float, q: float,
                       sigma: float, n_steps: int, kind: str = "call",
                       american: bool = False) -> float:
    """Average of the n- and (n+1)-step CRR prices.

    The CRR error oscillates with the parity of n: whether the strike falls
    on, or between, terminal nodes alternates from one n to the next, so
    consecutive errors tend to have opposite signs. Averaging two consecutive
    trees cancels most of that oscillation and leaves the smooth O(1/n) part.
    """
    return 0.5 * (crr_price(S, K, T, r, q, sigma, n_steps, kind, american)
                  + crr_price(S, K, T, r, q, sigma, n_steps + 1, kind, american))


def early_exercise_premium(S: float, K: float, T: float, r: float, q: float,
                           sigma: float, n_steps: int,
                           kind: str = "put") -> float:
    """American minus European price on the same tree (so tree error largely
    cancels). Always >= 0; zero for a call on a non-dividend stock (q = 0),
    which is never optimal to exercise early."""
    amer = crr_price(S, K, T, r, q, sigma, n_steps, kind, american=True)
    euro = crr_price(S, K, T, r, q, sigma, n_steps, kind, american=False)
    return amer - euro

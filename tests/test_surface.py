"""Offline tests of the IV-surface helpers (no network, synthetic inputs)."""

import math

import numpy as np
import pandas as pd
import pytest

from src import bsm, surface

SNAP = pd.Timestamp("2026-10-23 14:00:00", tz="UTC")   # 10:00 New York (EDT)
FLAT = pd.Series({0.25: 0.04, 1.0: 0.04})               # flat 4% curve


def test_time_to_expiry_uses_4pm_new_york():
    # 10:00 -> 16:00 EDT the same day is 6 hours.
    assert surface.time_to_expiry("2026-10-23", SNAP) == pytest.approx(
        6 / 24 / 365)
    # DST ends 1 Nov 2026, so 16:00 EST on 6 Nov is 21:00 UTC:
    # 14 days + 7 hours after the snapshot.
    assert surface.time_to_expiry("2026-11-06", SNAP) == pytest.approx(
        (14 + 7 / 24) / 365)


def test_svi_fit_recovers_known_parameters():
    true = np.array([0.01, 0.12, -0.6, 0.02, 0.15])     # a, b, rho, m, s
    T = 0.5
    k = np.linspace(-0.4, 0.25, 40)
    iv = np.sqrt(surface.svi_total_variance(k, *true) / T)
    fitted = surface.fit_svi(k, iv, T)
    np.testing.assert_allclose(fitted, true, atol=1e-5)


def test_parity_forward_recovers_bsm_forward():
    S, T, r, q, sig = 100.0, 0.75, 0.045, 0.012, 0.22
    K = np.array([80.0, 95.0, 100.0, 110.0])
    C = bsm.price(S, K, T, r, q, sig, "call")
    P = bsm.price(S, K, T, r, q, sig, "put")
    F = surface.parity_forward(C, P, K, T, r)
    np.testing.assert_allclose(F, S * math.exp((r - q) * T), rtol=1e-12)


def test_select_otm_keeps_puts_below_and_calls_above_forward():
    df = pd.DataFrame({"strike": [90.0, 110.0, 90.0, 110.0, 100.0, 100.0],
                       "kind": ["put", "put", "call", "call", "put", "call"],
                       "F": 100.0})
    kept = surface.select_otm(df)
    assert set(zip(kept["kind"], kept["strike"])) == {
        ("put", 90.0), ("call", 110.0), ("call", 100.0)}


def test_clean_chain_drops_zero_bid_crossed_and_stale_quotes():
    S, q = 100.0, 0.01
    expiry = "2026-12-18"
    T = surface.time_to_expiry(expiry, SNAP)
    fresh = SNAP - pd.Timedelta(hours=1)
    stale = SNAP - pd.Timedelta(days=30)
    # OTM puts priced off BSM so the surviving rows have a valid IV.
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0]
    fair = bsm.price(S, np.array(strikes), T, 0.04, q, 0.2, "put")
    bid, ask = fair - 0.01, fair + 0.01
    bid[0] = 0.0                    # zero bid
    ask[1] = bid[1] - 0.01          # crossed market
    last = [fresh, fresh, stale, fresh, fresh]
    chain = pd.DataFrame({"strike": strikes, "bid": bid, "ask": ask,
                          "lastTradeDate": last, "kind": "put",
                          "expiry": expiry})
    clean, table, _ = surface.clean_chain(chain, SNAP, S, q, FLAT)
    assert sorted(clean["strike"]) == [96.0, 98.0]
    assert table["rows_remaining"].iloc[0] == 5
    assert table["rows_remaining"].iloc[-1] == 2
    np.testing.assert_allclose(clean["iv"], 0.2, atol=1e-3)

"""Offline tests of the data helpers (no network)."""

import numpy as np
import pandas as pd
import pytest

from src import data


def test_trailing_dividend_yield_uses_only_past_dividends():
    idx = pd.bdate_range("2024-01-01", "2026-06-30")
    hist = pd.DataFrame({"Close": 100.0, "Dividends": 0.0}, index=idx)
    for d, amt in [("2024-03-15", 1.0), ("2024-06-14", 1.0),
                   ("2024-09-13", 1.0), ("2024-12-13", 1.0),
                   ("2025-03-14", 2.0)]:
        hist.loc[d, "Dividends"] = amt
    q = data.trailing_dividend_yield(hist)
    # Before the first ex-date nothing has been paid yet.
    assert q.loc["2024-03-14"] == 0.0
    # After two ex-dates: ln(1 + 2/100).
    assert q.loc["2024-07-01"] == pytest.approx(np.log1p(0.02))
    # Only the last four payments count: 1 + 1 + 1 + 2 = 5.
    assert q.loc["2025-03-14"] == pytest.approx(np.log1p(0.05))
    # More than 400 days after the last payment the estimate is dropped.
    assert q.loc["2026-06-30"] == 0.0


def test_rate_interpolation_is_linear_and_flat_outside():
    curve = pd.Series({1 / 12: 0.04, 1.0: 0.05})
    assert data.rate_for_maturity(curve, 0.01) == pytest.approx(0.04)
    assert data.rate_for_maturity(curve, 5.0) == pytest.approx(0.05)
    mid = (1 / 12 + 1.0) / 2
    assert data.rate_for_maturity(curve, mid) == pytest.approx(0.045)

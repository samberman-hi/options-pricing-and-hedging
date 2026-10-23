"""Download and cache all market data used in the project.

Sources (all free, no API key):
    * Yahoo Finance via yfinance  - SPY daily prices, SPY cash dividends,
                                    SPY option chain snapshot.
    * FRED public CSV endpoint    - Treasury constant-maturity yields
                                    (risk-free rate) and VIXCLS (CBOE VIX).
    * Stooq CSV endpoint          - fallback for SPY daily prices only.

Everything is cached under data/raw/ (gitignored). Functions read the cache if
it exists and only hit the network when the file is missing or
refresh=True. Nothing is ever simulated: if every source fails we raise.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

# FRED Treasury constant-maturity yields used to build the risk-free curve.
TREASURY_SERIES = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5,
                   "DGS1": 1.0, "DGS2": 2.0}


def _raw(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    return RAW / name


# --------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------
def fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    """A FRED series as a float Series indexed by date (missing values,
    e.g. market holidays, are dropped)."""
    path = _raw(f"fred_{series_id}.csv")
    if refresh or not path.exists():
        resp = requests.get(FRED_URL.format(sid=series_id), timeout=60)
        resp.raise_for_status()
        path.write_text(resp.text)
    df = pd.read_csv(path)
    date_col = df.columns[0]  # "observation_date" (older files: "DATE")
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    return s.dropna()


def vix(refresh: bool = False) -> pd.Series:
    """CBOE VIX close (in vol points). FRED VIXCLS, falling back to ^VIX
    on Yahoo Finance."""
    try:
        return fred_series("VIXCLS", refresh)
    except Exception as exc:  # network / format failure
        log.warning("FRED VIXCLS failed (%s); trying Yahoo ^VIX", exc)
        import yfinance as yf
        h = yf.Ticker("^VIX").history(start="2000-01-01", auto_adjust=False)
        if h.empty:
            raise RuntimeError("VIX unavailable from FRED and Yahoo") from exc
        s = h["Close"].copy()
        s.index = s.index.tz_localize(None).normalize()
        return s.rename("VIXCLS")


def risk_free_1m(refresh: bool = False) -> pd.Series:
    """1-month Treasury yield as a continuously compounded decimal rate.
    FRED quotes DGS1MO in percent on a bond-equivalent basis; we convert
    with r = ln(1 + y/100)."""
    y = fred_series("DGS1MO", refresh)
    return np.log1p(y / 100.0).rename("r")


def treasury_curve(asof: pd.Timestamp, refresh: bool = False) -> pd.Series:
    """Continuously compounded zero-ish rates by maturity (years) using the
    latest FRED observation on or before `asof` for each tenor."""
    out = {}
    for sid, tenor in TREASURY_SERIES.items():
        s = fred_series(sid, refresh)
        s = s[s.index <= asof]
        if s.empty:
            raise RuntimeError(f"No {sid} data on or before {asof.date()}")
        out[tenor] = float(np.log1p(s.iloc[-1] / 100.0))
    return pd.Series(out).sort_index()


def rate_for_maturity(curve: pd.Series, T: np.ndarray | float) -> np.ndarray:
    """Linear interpolation of the curve in maturity, flat extrapolation."""
    return np.interp(np.asarray(T, float), curve.index.values, curve.values)


# --------------------------------------------------------------------------
# SPY prices and dividends
# --------------------------------------------------------------------------
def spy_history(start: str = "2004-01-01", refresh: bool = False) -> pd.DataFrame:
    """Daily SPY OHLC (not dividend-adjusted) plus cash dividends.

    'Close' is the actual traded close (Yahoo adjusts for splits only; SPY
    has not split). Dividends are the cash amount per share on the ex-date.
    """
    path = _raw("spy_daily.csv")
    if refresh or not path.exists():
        df = _spy_from_yahoo(start)
        if df is None:
            raise RuntimeError(
                "Yahoo Finance download failed. Stooq only provides adjusted "
                "prices without dividends, which this study needs; aborting "
                "rather than mixing price conventions. Try again later.")
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def _spy_from_yahoo(start: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        h = yf.Ticker("SPY").history(start=start, auto_adjust=False,
                                     actions=True)
    except Exception as exc:
        log.warning("yfinance history failed: %s", exc)
        return None
    if h.empty:
        return None
    # Drop today's bar if the US session has not closed yet: an intraday
    # "close" would be a look-ahead-free but incomplete observation.
    now_et = pd.Timestamp.now(tz="America/New_York")
    if h.index[-1].date() == now_et.date() and now_et.hour < 17:
        h = h.iloc[:-1]
    h.index = h.index.tz_localize(None).normalize()
    return h[["Open", "High", "Low", "Close", "Adj Close", "Volume",
              "Dividends"]]


def spy_close_stooq() -> pd.Series:
    """Fallback/cross-check: SPY close from Stooq (dividend-adjusted)."""
    resp = requests.get(STOOQ_URL.format(sym="spy.us"), timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    return df["Close"]


def trailing_dividend_yield(hist: pd.DataFrame, window_days: int = 365
                            ) -> pd.Series:
    """Continuous dividend yield estimate known at each date:
    q_t = ln(1 + D_t / S_t) where D_t is the sum of cash dividends with
    ex-date in the trailing `window_days` calendar days (inclusive of t)
    and S_t is the close. Uses only information available at t."""
    divs = hist["Dividends"].fillna(0.0)
    trailing = divs.rolling(f"{window_days}D").sum()
    return np.log1p(trailing / hist["Close"]).rename("q")


# --------------------------------------------------------------------------
# SPY option chain snapshot
# --------------------------------------------------------------------------
CHAIN_FILE = "spy_chain.csv"
CHAIN_META = "spy_chain_meta.json"


def spy_option_chain(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """All listed SPY expiries (calls and puts) as one DataFrame, plus
    metadata (snapshot UTC time and the underlying price at the snapshot).

    The snapshot is cached; pass refresh=True to take a new one."""
    path, meta_path = _raw(CHAIN_FILE), _raw(CHAIN_META)
    if refresh or not (path.exists() and meta_path.exists()):
        import yfinance as yf
        tk = yf.Ticker("SPY")
        expiries = tk.options
        if not expiries:
            raise RuntimeError("Yahoo returned no SPY option expiries")
        snap = datetime.now(timezone.utc)
        spot = float(tk.fast_info["last_price"])
        frames = []
        for exp in expiries:
            ch = tk.option_chain(exp)
            for kind, df in (("call", ch.calls), ("put", ch.puts)):
                df = df.copy()
                df["kind"] = kind
                df["expiry"] = exp
                frames.append(df)
        chain = pd.concat(frames, ignore_index=True)
        chain.to_csv(path, index=False)
        meta = {"snapshot_utc": snap.isoformat(timespec="seconds"),
                "spot": spot, "n_expiries": len(expiries),
                "source": "Yahoo Finance via yfinance"}
        meta_path.write_text(json.dumps(meta, indent=2))
    chain = pd.read_csv(path, parse_dates=["lastTradeDate"])
    meta = json.loads(meta_path.read_text())
    return chain, meta


def download_all(refresh: bool = False, refresh_chain: bool = False) -> None:
    """Populate the cache with every dataset the pipeline needs."""
    spy_history(refresh=refresh)
    vix(refresh=refresh)
    for sid in TREASURY_SERIES:
        fred_series(sid, refresh=refresh)
    spy_option_chain(refresh=refresh_chain)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true",
                   help="re-download price/rate history")
    p.add_argument("--refresh-chain", action="store_true",
                   help="take a new option-chain snapshot")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    download_all(a.refresh, a.refresh_chain)
    print("cached files:", sorted(x.name for x in RAW.iterdir()))

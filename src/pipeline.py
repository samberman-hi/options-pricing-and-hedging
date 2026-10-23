"""Run the whole project end to end: data -> pricing checks -> IV surface ->
delta-hedging study. Figures go to figures/, tables to results/.

    python -m src.pipeline                  # use cached data where present
    python -m src.pipeline --refresh-chain  # take a new SPY option snapshot
    python -m src.pipeline --refresh        # re-download everything
"""

from __future__ import annotations

import argparse
import json
import logging

from src import data, hedging, pricing_report, surface
from src.plotting import RESULTS_DIR


def main(refresh: bool = False, refresh_chain: bool = False) -> dict:
    data.download_all(refresh=refresh, refresh_chain=refresh or refresh_chain)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "pricing": pricing_report.run(),
        "surface": surface.run(),
        "hedging": hedging.run(),
    }
    (RESULTS_DIR / "headline.json").write_text(
        json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--refresh-chain", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = main(a.refresh, a.refresh_chain)
    print(json.dumps(res, indent=2, default=str))

"""Shared figure style so every chart in figures/ looks the same."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# Colour-blind-safe palette (Okabe-Ito).
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.prop_cycle": matplotlib.cycler(color=COLORS),
})


def save(fig: plt.Figure, name: str, source: str) -> Path:
    """Add a data-source footnote and save to figures/<name>.png."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.text(0.01, 0.005, f"Source: {source}", fontsize=7, color="0.4",
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

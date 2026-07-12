"""Consistent notebook plotting and chart export utilities."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.titleweight": "bold"})


def save_fig(fig: plt.Figure, name: str) -> Path:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    charts_dir = config.ROOT_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    path = charts_dir / f"{safe_name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_monthly_trend(monthly: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly.index, monthly[config.COL_SALES], color=config.COLOR_PALETTE["actual"], linewidth=2)
    ax.set(title=title, xlabel="Date", ylabel="Sales ($)")
    fig.tight_layout()
    return fig

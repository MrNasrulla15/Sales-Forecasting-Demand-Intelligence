"""Exploratory analysis helpers used by the notebook."""

from __future__ import annotations

import pandas as pd

from . import config


def revenue_by_category(df: pd.DataFrame) -> dict[str, object]:
    table = df.groupby(config.COL_CATEGORY)[config.COL_SALES].sum().sort_values(ascending=False)
    table = table.rename("Total Revenue").to_frame()
    winner = table.index[0]
    return {"table": table, "verdict": f"{winner} generates the highest total revenue."}


def regional_growth_consistency(df: pd.DataFrame) -> dict[str, object]:
    yearly_pivot = df.pivot_table(
        index=config.COL_YEAR, columns=config.COL_REGION, values=config.COL_SALES, aggfunc="sum"
    ).sort_index()
    yoy = yearly_pivot.pct_change().iloc[1:] * 100
    summary = pd.DataFrame({
        "Average YoY Growth (%)": yoy.mean(),
        "Growth Volatility (%)": yoy.std(),
        "Positive Growth Years": (yoy > 0).sum(),
    })
    summary["Consistency Score"] = summary["Positive Growth Years"] / (1 + summary["Growth Volatility (%)"].fillna(0))
    summary = summary.sort_values(["Positive Growth Years", "Growth Volatility (%)"], ascending=[False, True])
    winner = summary.index[0]
    return {
        "yearly_pivot": yearly_pivot,
        "summary": summary,
        "verdict": f"{winner} has the most consistent sales growth using positive growth years and volatility.",
    }


def shipping_delay_analysis(df: pd.DataFrame) -> dict[str, object]:
    by_region = df.groupby(config.COL_REGION)[config.COL_SHIP_DELAY_DAYS].agg(["mean", "median", "min", "max"])
    return {
        "by_region": by_region,
        "verdict": f"Average shipping delay is {df[config.COL_SHIP_DELAY_DAYS].mean():.2f} days overall.",
    }


def seasonality_check(df: pd.DataFrame) -> dict[str, object]:
    monthly = df.groupby([config.COL_MONTH, config.COL_MONTH_NAME])[config.COL_SALES].mean()
    table = monthly.rename("avg_sales").reset_index().set_index(config.COL_MONTH)
    table = table.sort_index()
    threshold = table["avg_sales"].quantile(0.75)
    spike_months = table.index[table["avg_sales"] >= threshold].tolist()
    return {
        "monthly_rank_table": table,
        "spike_months": spike_months,
        "verdict": "Higher-than-usual average sales occur in the upper-quartile months highlighted in the chart.",
    }

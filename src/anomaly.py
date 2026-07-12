from __future__ import annotations

import numpy as np
import pandas as pd


def detect_anomalies_isolation_forest(series: pd.Series, contamination: float = 0.05) -> dict:
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=contamination, random_state=42)
    values = np.array(series.astype(float).values).reshape(-1, 1)
    labels = model.fit_predict(values)
    mask = labels == -1
    result = pd.Series(mask, index=series.index, name="is_anomaly")
    return {"is_anomaly": result, "model": model}


def detect_anomalies_zscore(series: pd.Series, window: int = 4, threshold: float = 2.5) -> dict:
    rolling_mean = series.shift(1).rolling(window=window, min_periods=window).mean()
    rolling_std = series.shift(1).rolling(window=window, min_periods=window).std()
    z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    mask = z_score.abs() > threshold
    result = pd.DataFrame({"z_score": z_score, "is_anomaly": mask})
    return {"is_anomaly": result["is_anomaly"], "z_score": result["z_score"]}


def compare_anomaly_methods(if_result: dict, z_result: dict) -> dict:
    if_mask = if_result["is_anomaly"]
    z_mask = z_result["is_anomaly"]
    return {
        "isolation_forest_count": int(if_mask.sum()),
        "zscore_count": int(z_mask.sum()),
        "agreed_both": int((if_mask & z_mask).sum()),
    }


def investigate_anomaly_drivers(df: pd.DataFrame, weekly_sales: pd.Series, anomaly_dates: list, z_scores: pd.Series | None = None) -> pd.DataFrame:
    if not anomaly_dates:
        return pd.DataFrame(columns=["Week", "Week Total Sales", "Top Order Product", "Top Order Share (%)"])
    rows = []
    for week in anomaly_dates:
        if week not in weekly_sales.index:
            continue
        week_df = df[df["Order Date"].dt.to_period("W-MON") == week.to_period("W-MON")] if hasattr(week, "to_period") else df
        if week_df.empty:
            continue
        top = week_df.nlargest(1, "Sales")
        rows.append({
            "Week": str(week),
            "Week Total Sales": float(week_df["Sales"].sum()),
            "Top Order Product": top["Product Name"].iloc[0] if not top.empty else "n/a",
            "Top Order Share (%)": float(top["Sales"].iloc[0] / week_df["Sales"].sum() * 100) if not top.empty and week_df["Sales"].sum() else 0.0,
        })
    return pd.DataFrame(rows)

from __future__ import annotations

from pathlib import Path
import pandas as pd
from . import config


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))

    fallback_candidates = [config.TRAIN_PATH]
    for candidate in fallback_candidates:
        if candidate.exists():
            candidates.append(candidate)

    resolved_path = None
    for item in candidates:
        full_path = item.expanduser().resolve()
        if full_path.exists():
            resolved_path = full_path
            break

    if resolved_path is None:
        raise FileNotFoundError(
            "Could not find sales data. Looked for: " + ", ".join(str(p) for p in candidates)
        )

    df = pd.read_csv(resolved_path)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    if "Postal Code" in df.columns:
        df["Postal Code"] = pd.to_numeric(df["Postal Code"], errors="coerce")
    required_columns = {config.COL_ORDER_DATE, config.COL_SHIP_DATE, config.COL_SALES}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Sales data is missing required columns: {sorted(missing_columns)}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[config.COL_ORDER_DATE] = pd.to_datetime(out[config.COL_ORDER_DATE], dayfirst=True, errors="coerce")
    out[config.COL_SHIP_DATE] = pd.to_datetime(out[config.COL_SHIP_DATE], dayfirst=True, errors="coerce")
    out[config.COL_YEAR] = out[config.COL_ORDER_DATE].dt.year
    out[config.COL_MONTH] = out[config.COL_ORDER_DATE].dt.month
    out[config.COL_MONTH_NAME] = out[config.COL_ORDER_DATE].dt.month_name()
    out[config.COL_WEEK] = out[config.COL_ORDER_DATE].dt.isocalendar().week.astype(int)
    out[config.COL_DAY_OF_WEEK] = out[config.COL_ORDER_DATE].dt.dayofweek
    out[config.COL_QUARTER] = out[config.COL_ORDER_DATE].dt.quarter
    out[config.COL_SEASON] = out[config.COL_MONTH].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
    out[config.COL_SHIP_DELAY_DAYS] = (out[config.COL_SHIP_DATE] - out[config.COL_ORDER_DATE]).dt.days
    return out


def get_weekly_monthly_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if config.COL_ORDER_DATE in work.columns:
        work = work.copy()
        work[config.COL_ORDER_DATE] = pd.to_datetime(work[config.COL_ORDER_DATE], errors="coerce")
        work = work.dropna(subset=[config.COL_ORDER_DATE]).set_index(config.COL_ORDER_DATE)
    if work.empty:
        raise ValueError("No valid order dates are available to create time series.")
    weekly = work[[config.COL_SALES]].resample("W").sum().rename(columns={config.COL_SALES: "Sales"})
    monthly = work[[config.COL_SALES]].resample("MS").sum().rename(columns={config.COL_SALES: "Sales"})
    return weekly, monthly


def validate_data_quality(df: pd.DataFrame) -> dict:
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_postal_code": int(df["Postal Code"].isna().sum()) if "Postal Code" in df.columns else 0,
        "negative_sales": int((df[config.COL_SALES] < 0).sum()) if config.COL_SALES in df.columns else 0,
    }

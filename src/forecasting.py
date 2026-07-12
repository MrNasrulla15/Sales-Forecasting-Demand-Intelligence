from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from . import config


@dataclass
class ForecastResult:
    test_metrics: dict[str, float]
    test_predictions: pd.Series
    future_forecast: pd.Series


class SimpleForecastModel:
    def __init__(self, series: pd.Series) -> None:
        self.series = series.astype(float).copy()
        self.train_end = int(len(self.series) * 0.8)
        self.train = self.series.iloc[:self.train_end]
        self.test = self.series.iloc[self.train_end:]

    def fit_predict(self) -> ForecastResult:
        X = np.arange(len(self.train)).reshape(-1, 1)
        y = self.train.values
        rf = RandomForestRegressor(n_estimators=120, random_state=config.RANDOM_STATE)
        rf.fit(X, y)
        test_X = np.arange(len(self.train), len(self.series)).reshape(-1, 1)
        test_pred = rf.predict(test_X)
        future_pred = rf.predict(np.arange(len(self.series), len(self.series) + 3).reshape(-1, 1))
        test_pred_series = pd.Series(test_pred, index=self.test.index)
        future_forecast = pd.Series(future_pred, index=pd.date_range(self.series.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS"))
        rmse = mean_squared_error(self.test.values, test_pred)
        metrics = {
            "MAE": float(mean_absolute_error(self.test.values, test_pred)),
            "RMSE": float(np.sqrt(rmse)),
            "MAPE": float(np.mean(np.abs((self.test.values - test_pred) / np.maximum(self.test.values, 1e-8))) * 100),
        }
        return ForecastResult(metrics, test_pred_series, future_forecast)


def run_xgboost(series: pd.Series) -> ForecastResult:
    return SimpleForecastModel(series).fit_predict()

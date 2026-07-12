from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config


def build_subcategory_features(df: pd.DataFrame) -> pd.DataFrame:
    total_volume = df.groupby(config.COL_SUBCATEGORY)[config.COL_SALES].sum().rename("TotalSalesVolume")
    avg_order_value = df.groupby(config.COL_SUBCATEGORY)[config.COL_SALES].mean().rename("AvgOrderValue")
    monthly = (
        df.set_index(config.COL_ORDER_DATE)
        .groupby(config.COL_SUBCATEGORY)[config.COL_SALES]
        .resample("MS")
        .sum()
        .rename("Sales")
        .reset_index()
    )
    volatility = monthly.groupby(config.COL_SUBCATEGORY)["Sales"].std().rename("SalesVolatility")
    yearly = (
        df.groupby([config.COL_SUBCATEGORY, df[config.COL_ORDER_DATE].dt.year])[config.COL_SALES]
        .sum()
        .rename("Sales")
        .reset_index()
        .rename(columns={config.COL_ORDER_DATE: "Year"})
    )
    growth_rows = []
    for subcat, sub in yearly.groupby(config.COL_SUBCATEGORY):
        sub = sub.sort_values("Year")
        yoy = sub["Sales"].pct_change().dropna() * 100
        growth_rows.append({config.COL_SUBCATEGORY: subcat, "AvgYoYGrowth": yoy.mean() if len(yoy) else 0.0})
    growth = pd.DataFrame(growth_rows).set_index(config.COL_SUBCATEGORY)["AvgYoYGrowth"]
    features = pd.concat([total_volume, growth, volatility, avg_order_value], axis=1)
    return features.fillna(0)


def compute_elbow_curve(features_scaled: np.ndarray, k_range=range(2, 8), random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(features_scaled)
        rows.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def fit_kmeans(features_scaled: np.ndarray, k: int, random_state: int = config.RANDOM_STATE) -> KMeans:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(features_scaled)
    return km


def label_clusters(features: pd.DataFrame, labels: np.ndarray) -> dict:
    df = features.copy()
    df["cluster"] = labels
    centroids = df.groupby("cluster")[list(features.columns)].mean()
    n = len(centroids)
    growth_rank = centroids["AvgYoYGrowth"].rank(ascending=False, method="first")
    volume_rank = centroids["TotalSalesVolume"].rank(ascending=False, method="first")
    volatility_rank = centroids["SalesVolatility"].rank(ascending=False, method="first")
    aov_rank = centroids["AvgOrderValue"].rank(ascending=False, method="first")
    names = {}
    for cluster_id in centroids.index:
        g_rank = growth_rank[cluster_id]
        v_rank = volume_rank[cluster_id]
        vol_rank = volatility_rank[cluster_id]
        aov_r = aov_rank[cluster_id]
        avg_growth = centroids.loc[cluster_id, "AvgYoYGrowth"]
        if g_rank == 1 and avg_growth > 0:
            names[cluster_id] = "Growing Demand"
        elif g_rank == n and avg_growth < 0:
            names[cluster_id] = "Declining Demand"
        elif aov_r == 1 and vol_rank <= n / 2:
            names[cluster_id] = "High-Value, Low-Frequency (Bulk Equipment)"
        elif v_rank <= n / 2 and vol_rank > n / 2:
            names[cluster_id] = "High Volume, Stable Demand"
        elif v_rank > n / 2 and vol_rank <= n / 2:
            names[cluster_id] = "Low Volume, High Volatility"
        elif v_rank <= n / 2:
            names[cluster_id] = "High Volume, Volatile Demand"
        else:
            names[cluster_id] = "Low Volume, Stable Demand"
    return names


def run_pca_2d(features_scaled: np.ndarray, random_state: int = config.RANDOM_STATE) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(features_scaled)
    return coords, pca


def build_segmentation(df: pd.DataFrame, k: int | None = None, k_range=range(2, 8)) -> dict:
    features = build_subcategory_features(df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    elbow_df = compute_elbow_curve(features_scaled, k_range=k_range)
    if k is None:
        k = 4
    model = fit_kmeans(features_scaled, k)
    labels = model.labels_
    cluster_names = label_clusters(features, labels)
    features_out = features.copy()
    features_out["cluster"] = labels
    features_out["cluster_name"] = features_out["cluster"].map(cluster_names)
    coords, pca = run_pca_2d(features_scaled)
    return {
        "features": features_out,
        "scaler": scaler,
        "elbow_df": elbow_df,
        "selected_k": k,
        "model": model,
        "cluster_names": cluster_names,
        "pca_coords": coords,
        "pca_explained_variance": pca.explained_variance_ratio_,
    }

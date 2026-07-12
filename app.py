"""
Sales Forecasting & Demand Intelligence Dashboard (Task 7).

Imports directly from src/ -- the same modules the notebook uses -- so the
dashboard can never silently drift out of sync with the analysis. All
heavy computation (data loading, model fitting) is wrapped in Streamlit's
caching so the app stays responsive after the first load.

Run locally:   streamlit run app.py
"""

import os
import sys
import warnings
from pathlib import Path

# Must be set before numpy/sklearn/xgboost are imported anywhere in the
# process. Running XGBoost (its own OpenMP thread pool) and scikit-learn
# (its own joblib/OpenBLAS thread pool) inside Streamlit's script-runner
# threads -- rather than the main thread, as in a plain script -- can
# trigger native thread-pool conflicts. This was caught during testing:
# the app ran fine as a plain Python script but segfaulted specifically
# under `streamlit.testing.v1.AppTest` on the clustering page. Pinning
# every native thread pool to 1 thread trades a little parallelism (the
# dataset is tiny -- this costs milliseconds) for eliminating an entire
# class of environment-dependent native crashes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import anomaly, config, data_loader, forecasting, segmentation


# Page config & visual identity


st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette shared with the notebook's matplotlib charts (src/config.py)
# so a manager moving between the notebook and this dashboard sees the same
# visual language for "actual", "forecast", and "anomaly" throughout.
NAVY = "#1b2735"
SLATE = "#334155"
BG = "#f7f8fa"
CARD_BG = "#ffffff"
BORDER = "#e5e7eb"
BLUE = config.COLOR_PALETTE["actual"]       # #1f77b4 -- actuals
ORANGE = config.COLOR_PALETTE["forecast"]   # #ff7f0e -- forecasts
RED = config.COLOR_PALETTE["anomaly"]       # #d62728 -- anomalies
GREEN = "#2ca02c"
PURPLE = "#9467bd"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Inter+Tight:wght@600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background-color: {BG};
}}

section[data-testid="stSidebar"] {{
    background-color: {NAVY};
}}
section[data-testid="stSidebar"] * {{
    color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.95rem;
}}

h1, h2, h3 {{
    font-family: 'Inter Tight', 'Inter', sans-serif !important;
    color: {NAVY} !important;
    letter-spacing: -0.01em;
}}

.dashboard-header {{
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1.5rem;
}}
.dashboard-eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 600;
    color: {BLUE};
    margin-bottom: 0.15rem;
}}

.kpi-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
.kpi-label {{
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {SLATE};
    opacity: 0.75;
    margin-bottom: 0.35rem;
}}
.kpi-value {{
    font-family: 'Inter Tight', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1.1;
}}
.kpi-delta-pos {{ color: {GREEN}; font-weight: 600; font-size: 0.85rem; }}
.kpi-delta-neg {{ color: {RED}; font-weight: 600; font-size: 0.85rem; }}

.section-note {{
    background: #eef2f7;
    border-left: 3px solid {BLUE};
    padding: 0.7rem 1rem;
    border-radius: 4px;
    font-size: 0.88rem;
    color: {SLATE};
    margin: 0.75rem 0 1.25rem 0;
}}

div[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
}}
</style>
""", unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-eyebrow">{eyebrow}</div>
        <h1 style="margin:0; font-size:1.9rem;">{title}</h1>
        {f'<div style="color:{SLATE}; font-size:0.95rem; margin-top:0.35rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = None, positive: bool = True):
    delta_html = ""
    if delta:
        cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", color=NAVY, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ============================================================================
# Cached data / model loading
# ============================================================================

@st.cache_data(show_spinner="Loading sales data...")
def load_data():
    df = data_loader.load_raw_data()
    df = data_loader.add_time_features(df)
    weekly, monthly = data_loader.get_weekly_monthly_series(df)
    return df, weekly, monthly


@st.cache_data(show_spinner="Training forecast model...")
def get_forecast(segment_key: str, _series: pd.Series):
    """
    `segment_key` (a short string) is what Streamlit hashes to key the
    cache; `_series` is deliberately unhashed (leading underscore is
    Streamlit's convention for "don't hash this argument"). Hashing a
    full pandas object on every call is unnecessary here (the data is
    static for the session) and was found, during testing, to crash
    inside pandas' hashing internals for larger DataFrames -- see
    `get_segmentation` below for the case that actually reproduced it.
    """
    return forecasting.run_xgboost(_series)


@st.cache_data(show_spinner="Scanning for anomalies...")
def get_anomalies(_weekly_sales: pd.Series):
    if_result = anomaly.detect_anomalies_isolation_forest(_weekly_sales)
    z_result = anomaly.detect_anomalies_zscore(_weekly_sales)
    return if_result, z_result


@st.cache_data(show_spinner="Segmenting product demand...")
def get_segmentation(_df: pd.DataFrame):
    """
    Leading underscore on `_df` is required, not stylistic: passing the
    full transactions DataFrame as a normally-hashed cache argument
    reproduced a segmentation fault inside Streamlit's cache-hashing of a
    pandas DataFrame with Arrow-backed string columns (traced via
    `python -X faulthandler` to `pandas/core/arrays/string_arrow.py`
    during development). Since this app loads one static dataset per
    session, skipping the hash and caching unconditionally on first call
    is both correct and the fix.
    """
    return segmentation.build_segmentation(_df, k=4)


df, weekly, monthly = load_data()


# ============================================================================
# Sidebar navigation
# ============================================================================

st.sidebar.markdown(f"""
<div style="padding: 0.5rem 0 1.5rem 0;">
    <div style="font-family:'Inter Tight',sans-serif; font-weight:800; font-size:1.15rem; color:white;">
        📦 Demand Intelligence
    </div>
    <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.1rem;">Superstore Sales System</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Sales Overview", "🔮 Forecast Explorer", "🚨 Anomaly Report", "🧩 Demand Segments"],
    label_visibility="collapsed",
)

st.sidebar.markdown(f"""
<div style="position: fixed; bottom: 1.5rem; font-size: 0.75rem; color: #94a3b8; padding-right: 1rem;">
    Data: {df['Order Date'].min().strftime('%b %Y')} – {df['Order Date'].max().strftime('%b %Y')}<br>
    {len(df):,} order lines
</div>
""", unsafe_allow_html=True)


# ============================================================================
# PAGE 1 — Sales Overview
# ============================================================================

if page == "📊 Sales Overview":
    page_header("Page 1", "Sales Overview Dashboard", "Total revenue, trend, and regional/category breakdown")

    col1, col2, col3, col4 = st.columns(4)
    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    avg_order_value = df.groupby("Order ID")["Sales"].sum().mean()
    yearly_totals = df.groupby(df["Order Date"].dt.year)["Sales"].sum()
    yoy = (yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2] * 100

    with col1:
        kpi_card("Total Revenue", f"${total_sales/1e6:.2f}M")
    with col2:
        kpi_card("Total Orders", f"{total_orders:,}")
    with col3:
        kpi_card("Avg Order Value", f"${avg_order_value:,.0f}")
    with col4:
        kpi_card(f"{yearly_totals.index[-1]} vs {yearly_totals.index[-2]}", f"{yoy:+.1f}%", delta="Year-over-year", positive=yoy > 0)

    st.write("")

    # Filters
    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        selected_regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    with fcol2:
        selected_categories = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    with fcol3:
        year_range = st.slider("Year range", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())))

    filtered = df[
        df["Region"].isin(selected_regions)
        & df["Category"].isin(selected_categories)
        & df["Year"].between(*year_range)
    ]

    if filtered.empty:
        st.warning("No data matches the selected filters. Try widening your selection.")
    else:
        left, right = st.columns([1, 1.4])

        with left:
            st.markdown("##### Total Sales by Year")
            yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
            fig = px.bar(yearly, x="Year", y="Sales", text_auto=".2s")
            fig.update_traces(marker_color=BLUE, textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown("##### Monthly Sales Trend")
            _, monthly_filtered = data_loader.get_weekly_monthly_series(filtered)
            fig = px.line(monthly_filtered.reset_index(), x="Order Date", y="Sales")
            fig.update_traces(line_color=BLUE, line_width=2.5)
            fig.update_layout(**PLOTLY_LAYOUT, height=340)
            st.plotly_chart(fig, width="stretch")

        left2, right2 = st.columns(2)
        with left2:
            st.markdown("##### Sales by Region")
            by_region = filtered.groupby("Region")["Sales"].sum().sort_values(ascending=True).reset_index()
            fig = px.bar(by_region, x="Sales", y="Region", orientation="h", text_auto=".2s")
            fig.update_traces(marker_color=GREEN, textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with right2:
            st.markdown("##### Sales by Category")
            by_cat = filtered.groupby("Category")["Sales"].sum().reset_index()
            fig = px.pie(by_cat, names="Category", values="Sales", hole=0.55,
                         color_discrete_sequence=[BLUE, ORANGE, GREEN])
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig, width="stretch")


# ============================================================================
# PAGE 2 — Forecast Explorer
# ============================================================================

elif page == "🔮 Forecast Explorer":
    page_header("Page 2", "Forecast Explorer", "XGBoost forecasts by category or region (recommended model per Task 3's holdout backtest)")

    st.markdown(f"""
    <div class="section-note">
        <b>Model note:</b> XGBoost was selected in the notebook's Task 3 comparison for the lowest
        MAE and MAPE on a 3-month holdout backtest (vs. SARIMA and Prophet). Forecasts below are
        generated live using the same <code>src/forecasting.py</code> logic as the notebook.
    </div>
    """, unsafe_allow_html=True)

    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    with fcol1:
        view_by = st.selectbox("View by", ["Overall", "Category", "Region"])
    with fcol2:
        if view_by == "Category":
            segment_value = st.selectbox("Select Category", sorted(df["Category"].unique()))
        elif view_by == "Region":
            segment_value = st.selectbox("Select Region", sorted(df["Region"].unique()))
        else:
            segment_value = None
    with fcol3:
        horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    if view_by == "Overall":
        series = monthly["Sales"]
        series_label = "Overall"
    else:
        col = "Category" if view_by == "Category" else "Region"
        sub_df = df[df[col] == segment_value]
        _, seg_monthly = data_loader.get_weekly_monthly_series(sub_df)
        series = seg_monthly["Sales"]
        series_label = segment_value

    cache_key = f"{view_by}_{segment_value}"
    result = get_forecast(cache_key, series)

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        kpi_card("MAE (holdout)", f"${result.test_metrics['MAE']:,.0f}")
    with mcol2:
        kpi_card("RMSE (holdout)", f"${result.test_metrics['RMSE']:,.0f}")
    with mcol3:
        kpi_card("MAPE (holdout)", f"{result.test_metrics['MAPE']:.1f}%")

    st.write("")

    forecast_slice = result.future_forecast.iloc[:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", name="Actual",
        line=dict(color=BLUE, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=result.test_predictions.index, y=result.test_predictions.values, mode="lines+markers",
        name="Backtest (holdout)", line=dict(color=PURPLE, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[series.index[-1]] + list(forecast_slice.index),
        y=[series.values[-1]] + list(forecast_slice.values),
        mode="lines+markers", name=f"Forecast (+{horizon}mo)",
        line=dict(color=ORANGE, width=2.5), marker=dict(size=9),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=440, title=f"{series_label}: Actual vs. Forecast")
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Forecast values")
    forecast_table = pd.DataFrame({
        "Month": [d.strftime("%B %Y") for d in forecast_slice.index],
        "Forecasted Sales": [f"${v:,.2f}" for v in forecast_slice.values],
    })
    st.dataframe(forecast_table, width="stretch", hide_index=True)


# ============================================================================
# PAGE 3 — Anomaly Report
# ============================================================================

elif page == "🚨 Anomaly Report":
    page_header("Page 3", "Anomaly Report", "Weekly sales anomalies — Isolation Forest & rolling Z-score")

    if_result, z_result = get_anomalies(weekly["Sales"])
    comparison = anomaly.compare_anomaly_methods(if_result, z_result)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        kpi_card("Isolation Forest flags", str(comparison["isolation_forest_count"]))
    with mcol2:
        kpi_card("Z-score flags", str(comparison["zscore_count"]))
    with mcol3:
        kpi_card("Both methods agree", str(comparison["agreed_both"]))
    with mcol4:
        kpi_card("Weeks analyzed", str(len(weekly)))

    st.write("")

    method_filter = st.radio("Show anomalies from", ["Both methods agree", "Isolation Forest", "Z-score", "Either method"], horizontal=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly.index, y=weekly["Sales"], mode="lines", name="Weekly Sales",
        line=dict(color="#cbd5e1", width=1.5),
    ))

    both_mask = if_result["is_anomaly"] & z_result["is_anomaly"]
    if method_filter == "Both methods agree":
        show_dates = weekly.index[both_mask]
    elif method_filter == "Isolation Forest":
        show_dates = weekly.index[if_result["is_anomaly"]]
    elif method_filter == "Z-score":
        show_dates = weekly.index[z_result["is_anomaly"]]
    else:
        show_dates = weekly.index[if_result["is_anomaly"] | z_result["is_anomaly"]]

    fig.add_trace(go.Scatter(
        x=show_dates, y=weekly.loc[show_dates, "Sales"], mode="markers", name="Anomaly",
        marker=dict(color=RED, size=11, symbol="diamond", line=dict(width=1, color="white")),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=420, title="Weekly Sales with Detected Anomalies")
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Detected anomaly weeks")
    explanation_table = anomaly.investigate_anomaly_drivers(
        df, weekly["Sales"], list(show_dates), z_scores=z_result["z_score"]
    )
    if explanation_table.empty:
        st.info("No anomalies for the selected method.")
    else:
        display_table = explanation_table.rename(columns={
            "Week Total Sales": "Sales ($)",
            "Top Order Share (%)": "Top Order Share",
        })
        st.dataframe(display_table, width="stretch", hide_index=True)

    st.markdown(f"""
    <div class="section-note">
        <b>Reading this table:</b> most spike weeks are driven by a single large B2B equipment
        order (copiers, industrial printers, etc.) rather than broad demand -- see the "Top Order
        Product" and "Top Order Share" columns for the specific driver of each flagged week.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE 4 — Product Demand Segments
# ============================================================================

elif page == "🧩 Demand Segments":
    page_header("Page 4", "Product Demand Segments", "K-Means clustering of sub-categories by volume, growth, volatility & order value")

    seg = get_segmentation(df)
    features = seg["features"]

    cluster_colors = {name: c for name, c in zip(sorted(features["cluster_name"].unique()),
                                                    [BLUE, ORANGE, GREEN, PURPLE, RED])}

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("##### Cluster Map (PCA projection)")
        coords = seg["pca_coords"]
        plot_df = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "Sub-Category": features.index,
            "Cluster": features["cluster_name"].values,
            "Total Sales": features["TotalSalesVolume"].values,
        })
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Cluster", text="Sub-Category",
            size="Total Sales", size_max=32,
            color_discrete_map=cluster_colors,
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        var_exp = seg["pca_explained_variance"]
        fig.update_layout(
            **PLOTLY_LAYOUT, height=480,
            xaxis_title=f"PC1 ({var_exp[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({var_exp[1]*100:.1f}% variance)",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("##### Cluster Composition")
        cluster_counts = features["cluster_name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Sub-Categories"]
        fig = px.bar(cluster_counts, x="Sub-Categories", y="Cluster", orientation="h",
                     color="Cluster", color_discrete_map=cluster_colors)
        fig.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        strategy_map = {
            "High Volume, Volatile Demand": "Higher safety stock; core revenue driver with real week-to-week swings.",
            "High-Value, Low-Frequency (Bulk Equipment)": "Lean / made-to-order inventory -- large infrequent orders, don't stock against average demand.",
            "Growing Demand": "Track closely; increase stocking ahead of demand if the trend holds over more quarters.",
            "Low Volume, Stable Demand": "Standard reorder-point management; low forecasting priority.",
        }
        for cname in sorted(features["cluster_name"].unique()):
            strategy = strategy_map.get(cname, "Review case-by-case.")
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:0.6rem; border-left: 4px solid {cluster_colors.get(cname, BLUE)};">
                <div style="font-weight:700; color:{NAVY}; font-size:0.9rem;">{cname}</div>
                <div style="font-size:0.82rem; color:{SLATE}; margin-top:0.25rem;">{strategy}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("##### Sub-Category → Cluster Assignment")
    table = features[["TotalSalesVolume", "AvgYoYGrowth", "SalesVolatility", "AvgOrderValue", "cluster_name"]].copy()
    table.columns = ["Total Sales Volume", "Avg YoY Growth (%)", "Sales Volatility", "Avg Order Value", "Cluster"]
    table = table.round(2).sort_values("Cluster")
    st.dataframe(table, width="stretch")
"""
============================================================
  ECONOMIC INTELLIGENCE PREDICTION AND RISK ANALYSIS SYSTEM
  India — 50 Years Economic Data
  Built with Streamlit | VS Code Compatible
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ──────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error,
                            mean_squared_error, accuracy_score,
                            confusion_matrix, classification_report)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ── Statsmodels (ARIMA) ───────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ── Optional XGBoost ──────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ═════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Economic Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a237e, #0d47a1, #1565c0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .subtitle {
        text-align: center; color: #546e7a;
        font-size: 1.05rem; margin-bottom: 1.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1a237e, #283593);
        color: white; padding: 0.6rem 1.2rem;
        border-radius: 8px; font-size: 1.15rem;
        font-weight: 700; margin: 1.2rem 0 0.8rem 0;
    }
    .metric-card {
        background: #f8f9ff; border: 1px solid #c5cae9;
        border-radius: 10px; padding: 1rem;
        text-align: center; margin: 0.3rem;
    }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #1a237e; }
    .metric-label { font-size: 0.85rem; color: #546e7a; margin-top: 0.2rem; }
    .risk-high   { background:#ffebee; border-left:5px solid #c62828; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .risk-medium { background:#fff3e0; border-left:5px solid #e65100; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .risk-low    { background:#e8f5e9; border-left:5px solid #2e7d32; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; padding:0.6rem 1.2rem; }
    .info-box {
        background:#e3f2fd; border:1px solid #90caf9;
        border-radius:8px; padding:0.8rem 1rem; margin:0.6rem 0;
        font-size:0.92rem; color:#0d47a1;
    }
    .desc-box {
        background:#f3f4ff; border:1px solid #c5cae9;
        border-radius:8px; padding:1rem 1.2rem; margin:0.8rem 0;
        font-size:0.93rem; color:#1a237e; line-height:1.7;
    }
    .graph-caption {
        background:#fffde7; border-left:4px solid #f9a825;
        padding:0.6rem 1rem; border-radius:0 6px 6px 0;
        font-size:0.9rem; color:#555; margin: 0.4rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═════════════════════════════════════════════════════════
C = {
    "blue":   "#1565c0",
    "red":    "#c62828",
    "green":  "#2e7d32",
    "orange": "#e65100",
    "purple": "#6a1b9a",
    "navy":   "#1a237e",
    "gray":   "#546e7a",
    "gold":   "#f9a825",
    "bg":     "#f5f7ff",
}

plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["bg"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "font.family":      "DejaVu Sans",
})


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def graph_caption(text):
    st.markdown(
        f'<div class="graph-caption">📌 <b>Insight:</b> {text}</div>',
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════
#  MODULE 1 — DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df.replace("..", np.nan, inplace=True)

    numeric_cols = [
        "GDP growth (annual %)", "GDP deflator",
        "Inflation, consumer prices (annual %)",
        "Inflation, GDP deflator (annual %)",
        "Unemployment with basic education", "Unemployment, total",
        "Unemployment, youth female", "Unemployment, youth male",
        "Unemployment, youth total", "Population growth (annual %)",
        "Population, total", "Real interest rate (%)",
        "Central government debt", "Trade",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df.sort_values("Time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# ═════════════════════════════════════════════════════════
#  MODULE 2 — ARIMA FORECASTING
# ═════════════════════════════════════════════════════════
def adf_test(series, name):
    result = adfuller(series.dropna())
    return {
        "Variable":   name,
        "ADF Stat":   round(result[0], 4),
        "p-value":    round(result[1], 4),
        "Stationary": "Yes ✅" if result[1] < 0.05 else "No ❌"
    }


def fit_arima(series):
    for order in [(1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0)]:
        try:
            return ARIMA(series.dropna(), order=order).fit()
        except Exception:
            continue
    return None


def forecast_arima(fit, steps):
    fc = fit.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()


def plot_arima_forecast(series, fit, steps, title, color, ylabel, years):
    forecast_mean, conf_int = forecast_arima(fit, steps)
    last_yr      = int(years.iloc[-1])
    future_years = list(range(last_yr + 1, last_yr + steps + 1))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, series, color=color, lw=2.2, label="Historical", zorder=3)
    ax.plot(future_years, forecast_mean.values,
            color=C["gold"], lw=2.2, ls="--", marker="o",
            markersize=5, label="Forecast", zorder=4)
    ax.fill_between(future_years,
                    conf_int.iloc[:, 0].values,
                    conf_int.iloc[:, 1].values,
                    color=C["gold"], alpha=0.25, label="95% CI")
    ax.axvline(x=last_yr, color=C["gray"], ls=":", lw=1.2, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, forecast_mean, future_years


# ═════════════════════════════════════════════════════════
#  MODULE 3 — REGRESSION
# ═════════════════════════════════════════════════════════
FEATURES = [
    "GDP deflator",
    "Inflation, consumer prices (annual %)",
    "Unemployment, total",
    "Population growth (annual %)",
    "Population, total",
    "Real interest rate (%)",
    "Central government debt",
    "Trade",
]
TARGET = "GDP growth (annual %)"


def prepare_regression_data(df):
    sub = df[FEATURES + [TARGET]].dropna()
    return sub[FEATURES], sub[TARGET]


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "Model": name,
        "R²":    round(r2_score(y_te, y_pred), 4),
        "MAE":   round(mean_absolute_error(y_te, y_pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
        "model": model, "y_pred": y_pred, "y_test": y_te
    }


def plot_feature_importance(importances, feature_names, title, color):
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color=color, edgecolor="white", height=0.65)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(importances[idx]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name, color):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(y_test, y_pred, color=color, alpha=0.75,
               edgecolors="white", s=70, zorder=3)
    lims = [min(y_test.min(), y_pred.min()) - 0.5,
            max(y_test.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, color=C["gray"], ls="--", lw=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted\n{model_name}")
    ax.set_xlabel("Actual GDP Growth (%)")
    ax.set_ylabel("Predicted GDP Growth (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════
#  MODULE 4 — ECONOMIC HEALTH INDEX
# ═════════════════════════════════════════════════════════
def compute_ehi(df):
    sub = df[["Time", "GDP growth (annual %)",
              "Inflation, consumer prices (annual %)",
              "Unemployment, total"]].copy().dropna()
    sub["EHI_raw"] = (
        0.4 * sub["GDP growth (annual %)"]
        - 0.3 * sub["Inflation, consumer prices (annual %)"]
        - 0.3 * sub["Unemployment, total"]
    )
    scaler = MinMaxScaler()
    sub["EHI"] = scaler.fit_transform(sub[["EHI_raw"]])
    return sub


def classify_ehi(val):
    if val >= 0.70:   return "Strong"
    elif val >= 0.45: return "Stable"
    elif val >= 0.25: return "Risky"
    else:             return "Recession"


def plot_ehi_trend(ehi_df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ax = axes[0]
    ax.fill_between(ehi_df["Time"], 0.70, 1.0,  color=C["green"],  alpha=0.12, label="Strong (≥0.70)")
    ax.fill_between(ehi_df["Time"], 0.45, 0.70, color=C["blue"],   alpha=0.12, label="Stable (0.45–0.70)")
    ax.fill_between(ehi_df["Time"], 0.25, 0.45, color=C["orange"], alpha=0.12, label="Risky (0.25–0.45)")
    ax.fill_between(ehi_df["Time"], 0.0,  0.25, color=C["red"],    alpha=0.12, label="Recession (<0.25)")
    ax.plot(ehi_df["Time"], ehi_df["EHI"], color=C["navy"],
            lw=2.3, marker="o", markersize=4, label="EHI Score")
    ax.set_title("Economic Health Index — Normalized Score (0–1)")
    ax.set_ylabel("EHI Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    color_map = {"Strong": C["green"], "Stable": C["blue"],
                 "Risky": C["orange"], "Recession": C["red"]}
    cats = ehi_df["EHI"].apply(classify_ehi)
    ax2.bar(ehi_df["Time"], [1] * len(ehi_df),
            color=[color_map[c] for c in cats], width=0.85, edgecolor="white")
    ax2.set_title("Economic Condition Category per Year")
    ax2.set_yticks([])
    ax2.set_xlabel("Year")
    patches = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
    ax2.legend(handles=patches, loc="upper left", fontsize=8)
    ax2.grid(False)
    fig.tight_layout(pad=2)
    return fig


# ═════════════════════════════════════════════════════════
#  MODULE 5 — CLASSIFICATION  (stratify-free, no ValueError)
# ═════════════════════════════════════════════════════════
def prepare_classification_data(ehi_df):
    ehi_df = ehi_df.copy()
    ehi_df["Label"] = ehi_df["EHI"].apply(classify_ehi)
    le = LabelEncoder()
    ehi_df["Label_enc"] = le.fit_transform(ehi_df["Label"])
    return ehi_df[["EHI"]], ehi_df["Label_enc"], le


def safe_split(X, y, test_size=0.2, random_state=42):
    """
    Simple split WITHOUT stratify.
    With 50 rows and 4 classes, stratified splitting causes
    'least populated class < 2' ValueError — this avoids it entirely.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def plot_confusion_matrix(cm, classes, title, color):
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = LinearSegmentedColormap.from_list("custom", ["white", color])
    im   = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════
#  MODULE 6 — RISK DETECTION
# ═════════════════════════════════════════════════════════
def detect_risks(df):
    latest = df.iloc[-1]
    year   = int(latest.get("Time", 0))
    alerts = []

    gdp   = latest.get("GDP growth (annual %)", np.nan)
    inf_  = latest.get("Inflation, consumer prices (annual %)", np.nan)
    unemp = latest.get("Unemployment, total", np.nan)

    if not np.isnan(inf_) and inf_ > 7:
        alerts.append(("🔴 HIGH INFLATION RISK",
                        f"Inflation = {inf_:.2f}% (Threshold: >7%)", "risk-high"))
    if not np.isnan(unemp) and unemp > 8:
        alerts.append(("🟠 EMPLOYMENT CRISIS",
                        f"Unemployment = {unemp:.2f}% (Threshold: >8%)", "risk-medium"))
    if not np.isnan(gdp) and gdp < 0:
        alerts.append(("🔴 ECONOMIC SLOWDOWN",
                        f"GDP Growth = {gdp:.2f}% (Negative Growth)", "risk-high"))
    if len(df) >= 3:
        if (df["GDP growth (annual %)"].iloc[-3:].diff().dropna() < 0).all():
            alerts.append(("🟠 CONSECUTIVE GDP DECLINE",
                            "GDP growth has declined for 3 consecutive years.", "risk-medium"))
    if len(df) >= 3:
        if (df["Unemployment, total"].iloc[-3:].diff().dropna() > 0).all():
            alerts.append(("🟡 RISING UNEMPLOYMENT TREND",
                            "Unemployment rising for 3 consecutive years.", "risk-medium"))
    if not alerts:
        alerts.append(("🟢 NO CRITICAL RISKS DETECTED",
                        "All indicators are within acceptable thresholds.", "risk-low"))
    return alerts, year


# ═════════════════════════════════════════════════════════
#  MODULE 7 — CORRELATION HEATMAP
# ═════════════════════════════════════════════════════════
def plot_correlation_heatmap(df):
    cols = [c for c in [
        "GDP growth (annual %)", "GDP deflator",
        "Inflation, consumer prices (annual %)",
        "Unemployment, total", "Population growth (annual %)",
        "Real interest rate (%)", "Central government debt", "Trade",
    ] if c in df.columns]
    corr  = df[cols].corr()
    short = [c.replace("Inflation, consumer prices (annual %)", "Inflation CPI")
              .replace("GDP growth (annual %)", "GDP Growth")
              .replace("Population growth (annual %)", "Pop Growth")
              .replace("Real interest rate (%)", "Real Rate")
              .replace("Central government debt", "Govt Debt")
              for c in cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list("rg", [C["red"], "white", C["blue"]])
    im   = ax.imshow(corr.values, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Correlation")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(short, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black" if abs(corr.values[i,j]) < 0.6 else "white")
    ax.set_title("Correlation Heatmap — Economic Indicators")
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Flag_of_India.svg", width=80)
    st.markdown("## 🧠 Economic Intelligence System")
    st.markdown("India — 50 Years Economic Data")
    st.divider()

    uploaded_file  = st.file_uploader("📂 Upload CSV Dataset", type=["csv"],
                                       help="Upload PMF 50yrs data.csv")
    forecast_steps = st.slider("Forecast Horizon (Years)", 3, 10, 5)
    test_size      = st.slider("Regression Test Split (%)", 10, 40, 20) / 100

    st.divider()
    st.markdown("""
    **Modules:**
    1. 📈 Time Series Forecast
    2. 🔮 Regression Modeling
    3. 💹 Economic Health Index
    4. 🏷️ Classification System
    5. ⚠️ Risk Detection
    6. 🔗 Correlation & Insights
    """)
    st.divider()
    st.caption("© 2025 Economic Intelligence System")


# ═════════════════════════════════════════════════════════
#  MAIN HEADER
# ═════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-title">📊 Economic Intelligence Prediction & Risk Analysis System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">India — 50 Years of Economic Data | AI & Data Science Project</div>',
    unsafe_allow_html=True
)

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
    👈 <b>Upload your CSV dataset</b> from the sidebar to begin.<br>
    Expected file: <code>PMF 50yrs data.csv</code>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Analysis Modules</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-value">50</div><div class="metric-label">Years of Data</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-value">18</div><div class="metric-label">Economic Indicators</div></div>', unsafe_allow_html=True)
    st.stop()


# ── Load Data ─────────────────────────────────────────────
with st.spinner("Loading and preprocessing dataset..."):
    df = load_and_preprocess(uploaded_file)

st.success(f"✅ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")

# ── KPI Cards ─────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
yr_range = f"{int(df['Time'].min())}–{int(df['Time'].max())}" if "Time" in df.columns else "N/A"
with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{yr_range}</div><div class="metric-label">Year Range</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["GDP growth (annual %)"].mean():.2f}%</div><div class="metric-label">Avg GDP Growth</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Inflation, consumer prices (annual %)"].mean():.2f}%</div><div class="metric-label">Avg Inflation</div></div>', unsafe_allow_html=True)
with c5: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Unemployment, total"].mean():.2f}%</div><div class="metric-label">Avg Unemployment</div></div>', unsafe_allow_html=True)

with st.expander("🔍 View Raw Dataset", expanded=False):
    st.dataframe(df, use_container_width=True)


# ═════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 OUTPUT 1 — Forecast",
    "🔮 OUTPUT 2 — Regression",
    "💹 OUTPUT 3 — EHI",
    "🏷️ OUTPUT 4 — Classification",
    "⚠️ OUTPUT 5 — Risk Detection",
    "🔗 Correlation & Insights",
])


# ═════════════════════════════════════════════════════════
#  TAB 1 — TIME SERIES FORECAST
# ═════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(
        '<div class="section-header">📈 Output 1 — Time Series Forecasting using ARIMA</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="desc-box">
    <b>What this output does — Forecast:</b><br>
    This module uses the <b>ARIMA (AutoRegressive Integrated Moving Average)</b> model to forecast
    future values of three critical economic indicators based on 50 years of historical data.<br><br>
    <b>Indicators forecasted:</b> GDP Growth (%), Inflation CPI (%), Unemployment (%).<br>
    <b>Forecast horizon:</b> Next <b>{forecast_steps} years</b>.<br><br>
    <b>Steps performed:</b>
    &nbsp;&nbsp;• ADF Stationarity Test — checks whether each series needs differencing before ARIMA is applied.<br>
    &nbsp;&nbsp;• ARIMA(1,1,1) model fitted for each indicator with automatic fallback to simpler orders.<br>
    &nbsp;&nbsp;• Forecast values + 95% confidence interval are computed and plotted.<br>
    &nbsp;&nbsp;• Historical trend and forecast are overlaid on the same chart for clear comparison.<br><br>
    <b>Why ARIMA?</b> It is the industry-standard model for yearly economic time-series.
    It handles trends, autocorrelation, and differencing — all common in macroeconomic data.
    </div>
    """, unsafe_allow_html=True)

    years = df["Time"].dropna()

    # ── ADF Tests ─────────────────────────────────────────
    st.markdown("#### 🔬 Stationarity Test (ADF) Results")
    adf_rows = []
    for col, lbl in [
        ("GDP growth (annual %)",                "GDP Growth"),
        ("Inflation, consumer prices (annual %)", "Inflation CPI"),
        ("Unemployment, total",                  "Unemployment"),
    ]:
        if col in df.columns:
            adf_rows.append(adf_test(df[col].dropna(), lbl))
    st.dataframe(pd.DataFrame(adf_rows), use_container_width=True)
    graph_caption(
        "A p-value < 0.05 means the series is stationary (no unit root) and is directly suited for ARIMA. "
        "A non-stationary series is automatically differenced (d=1) before fitting."
    )

    # ── Forecasts ─────────────────────────────────────────
    series_config = [
        ("GDP growth (annual %)",                "GDP Growth",    C["blue"],  "Growth (%)"),
        ("Inflation, consumer prices (annual %)", "Inflation CPI", C["red"],   "Inflation (%)"),
        ("Unemployment, total",                  "Unemployment",  C["green"], "Unemployment (%)"),
    ]

    forecast_table   = {}
    last_future_yrs  = []

    for col, label, color, ylabel in series_config:
        st.markdown(f"---\n#### 📊 {label} — ARIMA Forecast")
        series = df[col].dropna().reset_index(drop=True)
        yr_arr = df.loc[df[col].notna(), "Time"].reset_index(drop=True)

        with st.spinner(f"Fitting ARIMA for {label}..."):
            fit = fit_arima(series)

        if fit is None:
            st.warning(f"ARIMA could not be fitted for {label}.")
            continue

        col1, col2 = st.columns([3, 1])
        with col1:
            fig, fc_mean, future_yrs = plot_arima_forecast(
                series, fit, forecast_steps,
                f"{label} — Historical + {forecast_steps}-Year ARIMA Forecast",
                color, ylabel, yr_arr
            )
            show_fig(fig)

        with col2:
            st.markdown(f"**{label} Predicted Values**")
            fc_df = pd.DataFrame({"Year": future_yrs,
                                   "Forecast (%)": fc_mean.values.round(3)})
            st.dataframe(fc_df, use_container_width=True)
            forecast_table[label] = dict(zip(future_yrs, fc_mean.values.round(3)))

        # Caption with year-by-year prediction summary
        caption_parts = ", ".join(
            [f"{yr}: {v:.2f}%" for yr, v in zip(future_yrs, fc_mean.values)]
        )
        graph_caption(
            f"This graph forecasts <b>{label}</b> for the next {forecast_steps} years. "
            f"Predicted values → {caption_parts}. "
            f"The yellow dashed line is the ARIMA forecast; the shaded band is the 95% confidence interval "
            f"(actual value will likely fall within this range). "
            f"The vertical dotted line separates historical data from the forecast period."
        )

        with st.expander(f"📋 ARIMA Model Summary — {label}", expanded=False):
            st.text(str(fit.summary()))

        last_future_yrs = future_yrs

    if forecast_table:
        st.markdown("---\n#### 📋 Combined Forecast Summary Table")
        combined = pd.DataFrame(forecast_table, index=last_future_yrs)
        combined.index.name = "Year"
        st.dataframe(combined.style.format("{:.3f}"), use_container_width=True)
        graph_caption(
            "This combined table summarises ARIMA predictions for all three indicators together. "
            "Use this to compare whether GDP, Inflation, and Unemployment are expected to rise or fall "
            "in the same years — a useful tool for cross-indicator policy planning."
        )


# ═════════════════════════════════════════════════════════
#  TAB 2 — REGRESSION MODELING
# ═════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        '<div class="section-header">🔮 Output 2 — Multivariate Regression Modeling</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="desc-box">
    <b>What this output does — Regression:</b><br>
    This module trains three regression models to <b>predict GDP Growth (%)</b> using 8 other
    economic indicators as input features — simulating how the economy's GDP responds to
    changes in inflation, unemployment, trade, and other factors.<br><br>
    <b>Features used as input:</b> GDP Deflator, Inflation, Unemployment, Population Growth,
    Population Total, Real Interest Rate, Government Debt, Trade.<br>
    <b>Target variable:</b> GDP Growth (annual %).<br>
    <b>Split:</b> {int((1-test_size)*100)}% Training / {int(test_size*100)}% Testing.<br><br>
    <b>Models trained:</b><br>
    &nbsp;&nbsp;• <b>Random Forest Regressor</b> — ensemble of decision trees, handles non-linear relationships.<br>
    &nbsp;&nbsp;• <b>XGBoost Regressor</b> — gradient boosting, high accuracy on structured data.<br>
    &nbsp;&nbsp;• <b>Linear Regression</b> — simple baseline, shows direct coefficient impact.<br><br>
    <b>Metrics used:</b> R² Score (higher = better), MAE and RMSE (lower = better).
    </div>
    """, unsafe_allow_html=True)

    X, y = prepare_regression_data(df)

    if len(X) < 10:
        st.warning("Not enough data rows for regression after dropping NaNs.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

        models_reg = {
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
        }
        if XGBOOST_AVAILABLE:
            models_reg["XGBoost"] = XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=42, eval_metric="rmse", verbosity=0)

        results_reg  = []
        model_objects = {}
        with st.spinner("Training regression models..."):
            for name, mdl in models_reg.items():
                res = evaluate_model(name, mdl, X_tr, X_te, y_tr, y_te)
                results_reg.append(res)
                model_objects[name] = res

        # ── Metrics Table ──────────────────────────────────
        st.markdown("#### 📊 Model Performance Comparison")
        metrics_df = pd.DataFrame([
            {"Model": r["Model"], "R² Score": r["R²"], "MAE": r["MAE"], "RMSE": r["RMSE"]}
            for r in results_reg
        ])
        st.dataframe(
            metrics_df.style
                .highlight_max(subset=["R² Score"], color="#c8e6c9")
                .highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9"),
            use_container_width=True
        )
        best_r = max(results_reg, key=lambda r: r["R²"])
        graph_caption(
            f"Best model: <b>{best_r['Model']}</b> — R²={best_r['R²']}, MAE={best_r['MAE']}%, "
            f"RMSE={best_r['RMSE']}%. R² closer to 1.0 = model explains more variance in GDP growth. "
            f"MAE = average prediction error in percentage points. RMSE penalises large errors more."
        )

        # ── Actual vs Predicted ────────────────────────────
        st.markdown("#### 📈 Actual vs Predicted — GDP Growth (%)")
        pred_cols = st.columns(len(results_reg))
        plot_colors = [C["blue"], C["red"], C["green"]]
        for i, res in enumerate(results_reg):
            with pred_cols[i]:
                fig = plot_actual_vs_predicted(
                    res["y_test"], res["y_pred"], res["Model"],
                    plot_colors[i % len(plot_colors)]
                )
                show_fig(fig)
                graph_caption(
                    f"<b>{res['Model']}</b>: Each dot = one test year. Dots on the dashed diagonal = perfect prediction. "
                    f"R²={res['R²']} — the model explains {res['R²']*100:.1f}% of GDP growth variation."
                )

        # ── Feature Importance ─────────────────────────────
        st.markdown("#### 🔍 Feature Importance — What Drives GDP Growth?")
        fi_models = ["Random Forest"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])
        fi_cols   = st.columns(len(fi_models))
        fi_colors = [C["blue"], C["red"]]
        for i, name in enumerate(fi_models):
            if name in model_objects:
                mdl      = model_objects[name]["model"]
                imp      = mdl.feature_importances_
                top_feat = list(X.columns)[np.argmax(imp)]
                with fi_cols[i]:
                    fig = plot_feature_importance(imp, list(X.columns),
                                                   f"{name} — Feature Importance", fi_colors[i])
                    show_fig(fig)
                    graph_caption(
                        f"<b>{name}</b>: Most influential feature is <b>{top_feat}</b> "
                        f"(score={max(imp):.3f}). Longer bars = stronger influence on GDP growth prediction."
                    )

        # ── Linear Regression Coefficients ────────────────
        st.markdown("#### 📋 Linear Regression — Coefficients")
        lr_mdl  = model_objects["Linear Regression"]["model"]
        coef_df = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": lr_mdl.coef_.round(4)
        }).sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True)
        top_c = coef_df.iloc[0]
        graph_caption(
            f"Coefficients show the direct linear impact on GDP growth. "
            f"Strongest predictor: <b>{top_c['Feature']}</b> (coefficient={top_c['Coefficient']}). "
            f"Positive coefficient = variable increases GDP; Negative = it reduces GDP."
        )


# ═════════════════════════════════════════════════════════
#  TAB 3 — ECONOMIC HEALTH INDEX
# ═════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        '<div class="section-header">💹 Output 3 — Economic Health Index (EHI)</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Economic Health Index:</b><br>
    This module creates a <b>custom composite Economic Health Index (EHI)</b> — a single numeric score
    per year that summarises the overall economic health of India using three key indicators.<br><br>
    <b>Formula:</b> <code>EHI = 0.4 × GDP Growth − 0.3 × Inflation − 0.3 × Unemployment</code><br>
    &nbsp;&nbsp;• GDP Growth contributes positively (40% weight) — higher growth = healthier economy.<br>
    &nbsp;&nbsp;• Inflation and Unemployment contribute negatively (30% each) — high values drag the score down.<br><br>
    <b>Normalisation:</b> Raw EHI is rescaled to a 0–1 range using Min-Max Scaling for easy year-to-year comparison.<br><br>
    <b>Categories assigned per year:</b><br>
    &nbsp;&nbsp;• 🟢 <b>Strong</b> — EHI ≥ 0.70 &nbsp;|&nbsp; 🔵 <b>Stable</b> — 0.45–0.70
    &nbsp;|&nbsp; 🟠 <b>Risky</b> — 0.25–0.45 &nbsp;|&nbsp; 🔴 <b>Recession</b> — &lt; 0.25<br><br>
    <b>Why a custom index?</b> A single composite score is easier to track and communicate than
    monitoring 3 separate indicators individually.
    </div>
    """, unsafe_allow_html=True)

    ehi_df = compute_ehi(df)

    c1, c2, c3, c4 = st.columns(4)
    latest_ehi = ehi_df["EHI"].iloc[-1]
    avg_ehi    = ehi_df["EHI"].mean()
    best_yr    = int(ehi_df.loc[ehi_df["EHI"].idxmax(), "Time"])
    worst_yr   = int(ehi_df.loc[ehi_df["EHI"].idxmin(), "Time"])
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{latest_ehi:.3f}</div><div class="metric-label">Latest EHI Score</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_ehi:.3f}</div><div class="metric-label">50-Year Average EHI</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{best_yr}</div><div class="metric-label">Best Economic Year</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{worst_yr}</div><div class="metric-label">Worst Economic Year</div></div>', unsafe_allow_html=True)

    st.markdown("#### 📈 EHI Trend — 50 Years")
    fig = plot_ehi_trend(ehi_df)
    show_fig(fig)
    latest_condition = classify_ehi(latest_ehi)
    graph_caption(
        f"Top chart: EHI score from {int(ehi_df['Time'].min())} to {int(ehi_df['Time'].max())}. "
        f"Current (latest) EHI = <b>{latest_ehi:.3f}</b> → economy classified as <b>{latest_condition}</b>. "
        f"Best year = <b>{best_yr}</b> (highest EHI), Worst year = <b>{worst_yr}</b> (lowest EHI). "
        f"Bottom bar: each year's colour shows its economic condition category at a glance."
    )

    st.markdown("#### 📋 Year-wise EHI Score Table")
    ehi_display = ehi_df[["Time", "GDP growth (annual %)",
                           "Inflation, consumer prices (annual %)",
                           "Unemployment, total", "EHI_raw", "EHI"]].copy()
    ehi_display["Condition"] = ehi_display["EHI"].apply(classify_ehi)
    ehi_display.columns = ["Year", "GDP Growth %", "Inflation %",
                            "Unemployment %", "EHI Raw", "EHI (0-1)", "Condition"]
    st.dataframe(
        ehi_display.style.format({
            "GDP Growth %": "{:.2f}", "Inflation %": "{:.2f}",
            "Unemployment %": "{:.2f}", "EHI Raw": "{:.3f}", "EHI (0-1)": "{:.3f}",
        }).background_gradient(subset=["EHI (0-1)"], cmap="RdYlGn"),
        use_container_width=True
    )
    graph_caption(
        "This table shows how the EHI score is calculated for each year. "
        "Green = healthier economy (high EHI), Red = weaker economy (low EHI). "
        "The 'Condition' column gives the final economic label assigned to that year."
    )


# ═════════════════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION
# ═════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(
        '<div class="section-header">🏷️ Output 4 — Economic Condition Classification</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Classification:</b><br>
    This module trains three <b>machine learning classifiers</b> to automatically label each year's
    economic condition as <b>Strong, Stable, Risky, or Recession</b>, based on the EHI score
    computed in Output 3.<br><br>
    <b>Input feature:</b> EHI Score (0–1). &nbsp;&nbsp;<b>Labels:</b> Strong | Stable | Risky | Recession.<br><br>
    <b>Models trained:</b><br>
    &nbsp;&nbsp;• <b>Decision Tree</b> — creates if-else rules based on EHI score thresholds.<br>
    &nbsp;&nbsp;• <b>Random Forest Classifier</b> — ensemble of decision trees for higher accuracy.<br>
    &nbsp;&nbsp;• <b>Logistic Regression</b> — probabilistic classification with linear decision boundaries.<br><br>
    <b>Evaluation:</b> Accuracy score, confusion matrix, and full classification report (precision, recall, F1).<br><br>
    <b>Note on small dataset:</b> With only 50 data points and 4 classes, stratified splitting is skipped
    (it causes errors when some classes have only 1 sample in test set). A random 80/20 split is used instead.
    </div>
    """, unsafe_allow_html=True)

    try:
        ehi_df_cls        = compute_ehi(df)
        X_cls, y_cls, le  = prepare_classification_data(ehi_df_cls)

        if len(X_cls) < 10:
            st.warning("Not enough data for classification.")
        else:
            # No stratify — avoids 'least populated class' error on small datasets
            X_tr_c, X_te_c, y_tr_c, y_te_c = safe_split(
                X_cls, y_cls, test_size=0.2, random_state=42)

            cls_models = {
                "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            }

            # Labels that actually exist in the full dataset
            all_label_indices = list(range(len(le.classes_)))

            cls_results = {}
            with st.spinner("Training classifiers..."):
                for name, mdl in cls_models.items():
                    mdl.fit(X_tr_c, y_tr_c)
                    y_pred_c = mdl.predict(X_te_c)
                    cls_results[name] = {
                        "model":  mdl,
                        "acc":    accuracy_score(y_te_c, y_pred_c),
                        # Force confusion matrix to always be 4x4 using all label indices
                        "cm":     confusion_matrix(
                                      y_te_c, y_pred_c,
                                      labels=all_label_indices),
                        "y_pred": y_pred_c,
                        "y_test": y_te_c,
                    }

            # ── Accuracy Cards ─────────────────────────────
            st.markdown("#### 🎯 Classifier Accuracy Scores")
            acc_cols = st.columns(3)
            for i, (name, res) in enumerate(cls_results.items()):
                with acc_cols[i]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{res["acc"]*100:.1f}%</div>'
                        f'<div class="metric-label">{name}</div></div>',
                        unsafe_allow_html=True
                    )

            best_cls_name = max(cls_results, key=lambda k: cls_results[k]["acc"])
            graph_caption(
                f"Best classifier: <b>{best_cls_name}</b> — accuracy = "
                f"<b>{cls_results[best_cls_name]['acc']*100:.1f}%</b>. "
                f"It correctly predicted the economic condition (Strong/Stable/Risky/Recession) "
                f"for {cls_results[best_cls_name]['acc']*100:.1f}% of the test years."
            )

            # ── Confusion Matrices ─────────────────────────
            st.markdown("#### 🔢 Confusion Matrices")
            cm_colors   = [C["blue"], C["green"], C["purple"]]
            cm_cols     = st.columns(3)
            classes_lbl = le.classes_   # always all 4 class names
            for i, (name, res) in enumerate(cls_results.items()):
                with cm_cols[i]:
                    st.markdown(f"**{name}**")
                    fig = plot_confusion_matrix(
                        res["cm"], classes_lbl, name, cm_colors[i])
                    show_fig(fig)

            graph_caption(
                "Each confusion matrix shows correct predictions (diagonal) vs "
                "misclassifications (off-diagonal). Rows = actual condition, "
                "Columns = predicted condition. Darker diagonal = better accuracy."
            )

            # ── Classification Reports ─────────────────────
            st.markdown("#### 📋 Detailed Classification Reports")
            for name, res in cls_results.items():
                with st.expander(f"📄 {name} — Full Report", expanded=False):
                    # Only report on labels present in test+pred to avoid size mismatch
                    present_labels = sorted(
                        set(res["y_test"].tolist()) | set(res["y_pred"].tolist()))
                    present_names  = [
                        le.classes_[i] for i in present_labels
                        if i < len(le.classes_)
                    ]
                    try:
                        report = classification_report(
                            res["y_test"], res["y_pred"],
                            labels=present_labels,
                            target_names=present_names,
                            zero_division=0
                        )
                    except Exception as rep_err:
                        report = (
                            f"Report could not be generated automatically.\n"
                            f"Reason: {rep_err}\n\n"
                            f"Classes seen in test set: "
                            f"{[le.classes_[i] for i in sorted(set(res['y_test'].tolist()))]}"
                        )
                    st.code(report)

            # ── Predicted labels for all years ────────────
            st.markdown("#### 🗂️ Predicted Economic Condition — All 50 Years")
            best_mdl   = cls_results[best_cls_name]["model"]
            all_pred   = best_mdl.predict(X_cls)
            all_labels = le.inverse_transform(all_pred)
            pred_df = pd.DataFrame({
                "Year":             ehi_df_cls["Time"].values,
                "EHI Score":        ehi_df_cls["EHI"].values.round(3),
                "Actual Condition": ehi_df_cls["EHI"].apply(classify_ehi).values,
                f"Predicted ({best_cls_name})": all_labels,
            })
            st.dataframe(pred_df, use_container_width=True)
            match_count = (
                pred_df["Actual Condition"] == pred_df[f"Predicted ({best_cls_name})"]
            ).sum()
            graph_caption(
                f"Using the best model (<b>{best_cls_name}</b>), {match_count} out of "
                f"{len(pred_df)} years were correctly classified. "
                f"Compare 'Actual Condition' (EHI threshold rule) with 'Predicted' (ML model) — "
                f"mismatches show years where the model's decision boundary slightly differed."
            )

    except Exception as tab4_err:
        st.error(f"Classification module encountered an error: {tab4_err}")
        st.info("All other output tabs (Risk Detection, Correlation) are unaffected.")


# ═════════════════════════════════════════════════════════
#  TAB 5 — RISK DETECTION
# ═════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(
        '<div class="section-header">⚠️ Output 5 — Risk Detection System</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Risk Detection:</b><br>
    This module implements a <b>Rule-Based Economic Risk Detection System</b> that scans the most
    recent economic data and raises alerts when critical thresholds are breached.<br><br>
    <b>Rules applied (Threshold Logic):</b><br>
    &nbsp;&nbsp;🔴 <b>High Inflation Risk</b> — Inflation &gt; 7%<br>
    &nbsp;&nbsp;🟠 <b>Employment Crisis</b> — Unemployment &gt; 8%<br>
    &nbsp;&nbsp;🔴 <b>Economic Slowdown</b> — GDP Growth &lt; 0% (negative growth)<br>
    &nbsp;&nbsp;🟠 <b>Consecutive GDP Decline</b> — GDP growth fell for 3 straight years<br>
    &nbsp;&nbsp;🟡 <b>Rising Unemployment Trend</b> — Unemployment rose for 3 straight years<br><br>
    <b>Additional outputs:</b><br>
    &nbsp;&nbsp;• A 50-year historical risk timeline showing when each threshold was breached.<br>
    &nbsp;&nbsp;• A year-by-year flag table marking each indicator's risk status.<br><br>
    <b>Why rule-based?</b> Economic risk detection doesn't always need heavy ML. Simple, transparent
    threshold rules are faster, more interpretable, and trusted by economists and policymakers.
    </div>
    """, unsafe_allow_html=True)

    alerts, latest_year = detect_risks(df)

    st.markdown(f"#### 🚨 Risk Alerts — Latest Year in Dataset: {latest_year}")
    for title, detail, cls in alerts:
        st.markdown(
            f'<div class="{cls}"><b>{title}</b><br>{detail}</div>',
            unsafe_allow_html=True
        )

    # ── Historical Risk Timeline ───────────────────────────
    st.markdown("#### 📊 Historical Risk Timeline — All 50 Years")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    risk_plot_cfg = [
        ("GDP growth (annual %)",                "GDP Growth (%)",    C["blue"],  None, 0),
        ("Inflation, consumer prices (annual %)", "Inflation (%)",    C["red"],   7,    1),
        ("Unemployment, total",                  "Unemployment (%)",  C["green"], 8,    2),
    ]
    for col, label, color, threshold, ax_idx in risk_plot_cfg:
        ax   = axes[ax_idx]
        vals = df[col]; yrs = df["Time"]
        ax.plot(yrs, vals, color=color, lw=2, marker="o", markersize=3)
        ax.fill_between(yrs, vals, 0, alpha=0.1, color=color)
        if threshold:
            ax.axhline(threshold, color=C["red"], ls="--", lw=1.3,
                       label=f"Risk Threshold ({threshold}%)")
            ax.fill_between(yrs, threshold, vals.clip(lower=threshold),
                            where=vals > threshold, alpha=0.25, color=C["red"])
        ax.set_ylabel(label, fontsize=9)
        if threshold: ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Year")
    axes[0].set_title("Economic Risk Indicators — 50-Year Historical Overview")
    fig.tight_layout()
    show_fig(fig)

    latest_row  = df.iloc[-1]
    gdp_val     = latest_row["GDP growth (annual %)"]
    inf_val     = latest_row["Inflation, consumer prices (annual %)"]
    unemp_val   = latest_row["Unemployment, total"]
    graph_caption(
        f"This chart shows all 3 risk indicators across 50 years. Red shaded zones = years where the "
        f"indicator breached the risk threshold. "
        f"In {latest_year}: GDP Growth = {gdp_val:.2f}%, "
        f"Inflation = {inf_val:.2f}% ({'⚠ above' if inf_val>7 else 'within'} 7% threshold), "
        f"Unemployment = {unemp_val:.2f}% ({'⚠ above' if unemp_val>8 else 'within'} 8% threshold)."
    )

    # ── Year-wise Risk Flag Table ──────────────────────────
    st.markdown("#### 📋 Year-wise Risk Flag Table")
    risk_df = df[["Time", "GDP growth (annual %)",
                  "Inflation, consumer prices (annual %)",
                  "Unemployment, total"]].copy()
    risk_df["Inflation Risk"]    = risk_df["Inflation, consumer prices (annual %)"].apply(
        lambda x: "🔴 HIGH" if x > 7 else "🟢 OK")
    risk_df["Unemployment Risk"] = risk_df["Unemployment, total"].apply(
        lambda x: "🟠 CRISIS" if x > 8 else "🟢 OK")
    risk_df["GDP Risk"]          = risk_df["GDP growth (annual %)"].apply(
        lambda x: "🔴 SLOWDOWN" if x < 0 else "🟢 OK")
    risk_df.columns = ["Year", "GDP Growth %", "Inflation %", "Unemployment %",
                       "Inflation Risk", "Unemployment Risk", "GDP Risk"]
    st.dataframe(risk_df, use_container_width=True)

    high_inf_yrs  = risk_df[risk_df["Inflation Risk"]    == "🔴 HIGH"]["Year"].tolist()
    crisis_yrs    = risk_df[risk_df["Unemployment Risk"] == "🟠 CRISIS"]["Year"].tolist()
    slowdown_yrs  = risk_df[risk_df["GDP Risk"]          == "🔴 SLOWDOWN"]["Year"].tolist()
    graph_caption(
        f"Risk flags across 50 years: "
        f"High Inflation years (>7%): {high_inf_yrs if high_inf_yrs else 'None'}. "
        f"Unemployment Crisis years (>8%): {crisis_yrs if crisis_yrs else 'None'}. "
        f"GDP Slowdown years (<0%): {slowdown_yrs if slowdown_yrs else 'None'}."
    )


# ═════════════════════════════════════════════════════════
#  TAB 6 — CORRELATION & INSIGHTS
# ═════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        '<div class="section-header">🔗 Correlation Analysis & Economic Insights</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Correlation & Insights:</b><br>
    This module provides a <b>deep statistical analysis</b> of how India's economic indicators
    relate to each other, and auto-generates key insights from 50 years of data.<br><br>
    <b>Outputs in this section:</b><br>
    &nbsp;&nbsp;• <b>Correlation Heatmap</b> — Pearson correlation between all indicator pairs (range: −1 to +1).
    Blue = positive correlation, Red = negative correlation.<br>
    &nbsp;&nbsp;• <b>Multi-indicator Trend Plots</b> — 50-year line charts for GDP, Inflation, Unemployment, Trade.<br>
    &nbsp;&nbsp;• <b>Automated Insights</b> — key data-driven findings such as peak years, trend direction,
    and strongest economic relationships.<br><br>
    <b>Why correlation?</b> If inflation consistently rises before GDP falls, inflation becomes a
    leading indicator — helping policymakers act before a slowdown occurs.
    </div>
    """, unsafe_allow_html=True)

    # ── Heatmap ───────────────────────────────────────────
    st.markdown("#### 🌡️ Correlation Heatmap — All Economic Indicators")
    fig = plot_correlation_heatmap(df)
    show_fig(fig)

    corr_gdp_inf   = df[["GDP growth (annual %)", "Inflation, consumer prices (annual %)"]].corr().iloc[0, 1]
    corr_gdp_unemp = df[["GDP growth (annual %)", "Unemployment, total"]].corr().iloc[0, 1]
    graph_caption(
        f"Each cell shows the Pearson correlation between two indicators. "
        f"GDP Growth vs Inflation = {corr_gdp_inf:.3f} "
        f"({'positively' if corr_gdp_inf > 0 else 'negatively'} correlated). "
        f"GDP Growth vs Unemployment = {corr_gdp_unemp:.3f} "
        f"({'positively' if corr_gdp_unemp > 0 else 'negatively'} correlated). "
        f"Values near ±1 = strong relationship; near 0 = weak/no relationship."
    )

    # ── Multi-indicator Trend Plots ────────────────────────
    st.markdown("#### 📈 Multi-Indicator Economic Trends (50 Years)")
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    trend_cfg = [
        ("GDP growth (annual %)",                "GDP Growth (%)",    C["blue"],   axes2[0, 0]),
        ("Inflation, consumer prices (annual %)", "Inflation (%)",    C["red"],    axes2[0, 1]),
        ("Unemployment, total",                  "Unemployment (%)",  C["green"],  axes2[1, 0]),
        ("Trade",                                "Trade (% of GDP)",  C["purple"], axes2[1, 1]),
    ]
    for col, label, color, ax in trend_cfg:
        if col in df.columns:
            ax.plot(df["Time"], df[col], color=color, lw=2)
            ax.fill_between(df["Time"], df[col], alpha=0.12, color=color)
            ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
            ax.grid(True, alpha=0.2)
    fig2.suptitle("India — Key Economic Indicator Trends (50 Years)",
                  fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()
    show_fig(fig2)

    gdp_peak_yr  = int(df.loc[df["GDP growth (annual %)"].idxmax(), "Time"])
    gdp_peak_val = df["GDP growth (annual %)"].max()
    inf_peak_yr  = int(df.loc[df["Inflation, consumer prices (annual %)"].idxmax(), "Time"])
    inf_peak_val = df["Inflation, consumer prices (annual %)"].max()
    graph_caption(
        f"These 4 charts show how each indicator changed over 50 years. "
        f"GDP Growth peaked at <b>{gdp_peak_val:.2f}%</b> in <b>{gdp_peak_yr}</b>. "
        f"Inflation peaked at <b>{inf_peak_val:.2f}%</b> in <b>{inf_peak_yr}</b>. "
        f"Trade shows India's increasing global integration over the decades."
    )

    # ── Automated Insights ─────────────────────────────────
    st.markdown("#### 💡 Automated Economic Insights (Data-Driven)")
    insights = []

    gdp_recent = df["GDP growth (annual %)"].iloc[-5:].mean()
    if gdp_recent > 6:
        insights.append(("🟢", "Strong Recent GDP Momentum",
                          f"Average GDP growth over last 5 years: {gdp_recent:.2f}%"))
    elif gdp_recent > 3:
        insights.append(("🟡", "Moderate GDP Growth",
                          f"Average GDP growth over last 5 years: {gdp_recent:.2f}%"))
    else:
        insights.append(("🔴", "Weak GDP Growth",
                          f"Average GDP growth over last 5 years: {gdp_recent:.2f}%"))

    insights.append(("📊", "GDP–Inflation Correlation",
                      f"Pearson r = {corr_gdp_inf:.3f} "
                      f"({'positive' if corr_gdp_inf > 0 else 'negative'} relationship)"))
    insights.append(("📊", "GDP–Unemployment Correlation",
                      f"Pearson r = {corr_gdp_unemp:.3f} "
                      f"({'positive' if corr_gdp_unemp > 0 else 'negative'} relationship)"))
    insights.append(("🏆", "Best GDP Growth Year",
                      f"{gdp_peak_yr} — GDP grew at {gdp_peak_val:.2f}%"))
    insights.append(("📉", "Worst GDP Growth Year",
                      f"{int(df.loc[df['GDP growth (annual %)'].idxmin(), 'Time'])} — "
                      f"GDP was {df['GDP growth (annual %)'].min():.2f}%"))
    insights.append(("🔥", "Peak Inflation Year",
                      f"{inf_peak_yr} — Inflation reached {inf_peak_val:.2f}%"))
    insights.append(("👥", "Average Population Growth",
                      f"{df['Population growth (annual %)'].mean():.2f}% per year over 50 years"))

    for icon, title, detail in insights:
        st.markdown(f"""
        <div style="background:#f8f9ff; border:1px solid #c5cae9; border-radius:8px;
                    padding:0.8rem 1.2rem; margin:0.4rem 0;">
            <span style="font-size:1.3rem">{icon}</span>
            <b style="color:#1a237e; margin-left:0.5rem;">{title}</b>
            <br><span style="color:#546e7a; margin-left:2rem;">{detail}</span>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style='text-align:center; color:#9e9e9e; font-size:0.88rem; padding:1rem 0;'>
    <b>Economic Intelligence Prediction & Risk Analysis System</b><br>
    India — 50-Year Economic Dataset | Built with Streamlit · Scikit-learn · Statsmodels · Matplotlib
</div>
""", unsafe_allow_html=True)
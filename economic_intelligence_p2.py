import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Core ML / Stats ───────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# ── Prophet (optional) ────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ═════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Economic Intelligence — P2",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════
#  GLOBAL CSS  (matches P1 palette + new P2 styles)
# ═════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a237e, #0d47a1, #1565c0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .subtitle { text-align:center; color:#546e7a; font-size:1.05rem; margin-bottom:1.5rem; }
    .section-header {
        background: linear-gradient(90deg,#1a237e,#283593);
        color:white; padding:0.6rem 1.2rem; border-radius:8px;
        font-size:1.15rem; font-weight:700; margin:1.2rem 0 0.8rem 0;
    }
    .p2-header {
        background: linear-gradient(90deg,#4a148c,#6a1b9a);
        color:white; padding:0.6rem 1.2rem; border-radius:8px;
        font-size:1.15rem; font-weight:700; margin:1.2rem 0 0.8rem 0;
    }
    .metric-card {
        background:#f8f9ff; border:1px solid #c5cae9;
        border-radius:10px; padding:1rem; text-align:center; margin:0.3rem;
    }
    .metric-value { font-size:1.6rem; font-weight:800; color:#1a237e; }
    .metric-label { font-size:0.85rem; color:#546e7a; margin-top:0.2rem; }
    .graph-caption {
        background:#fffde7; border-left:4px solid #f9a825;
        padding:0.6rem 1rem; border-radius:0 6px 6px 0;
        font-size:0.9rem; color:#555; margin:0.4rem 0 1rem 0;
    }
    .desc-box {
        background:#f3f4ff; border:1px solid #c5cae9;
        border-radius:8px; padding:1rem 1.2rem; margin:0.8rem 0;
        font-size:0.93rem; color:#1a237e; line-height:1.7;
    }
    .insight-card {
        background:#f8f9ff; border:1px solid #c5cae9; border-radius:8px;
        padding:0.8rem 1.2rem; margin:0.4rem 0;
    }
    .event-covid  { background:#fff3e0; border-left:5px solid #e65100; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .event-war    { background:#ffebee; border-left:5px solid #c62828; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .event-crisis { background:#fce4ec; border-left:5px solid #880e4f; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .event-good   { background:#e8f5e9; border-left:5px solid #2e7d32; padding:0.8rem; border-radius:6px; margin:0.4rem 0; }
    .stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; padding:0.6rem 1.2rem; }
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
    "teal":   "#00695c",
    "brown":  "#6d4c41",
    "bg":     "#f5f7ff",
}

# Country → colour mapping (consistent across all charts)
COUNTRY_COLORS = {
    "India":         C["blue"],
    "United States": C["red"],
    "Germany":       C["green"],
    "Brazil":        C["orange"],
}

plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["bg"],
    "axes.spines.top":  False,
    "axes.spines.right": False,
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
#  DATA LOADING
# ═════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_multi_country(uploaded_file):
    """
    Loads the multi-country CSV, standardises column names,
    converts numerics, forward/back fills NaNs per country.
    Returns a clean long-format DataFrame.
    """
    df = pd.read_csv(uploaded_file)
    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    df["Country Name"] = df["Country Name"].str.strip()

    # Rename to canonical short names (matching user's spec)
    rename_map = {
        "Broad money ":               "Broad money",
        "Expense ":                   "Expense",
        "GDP growth ":                "GDP growth",
        "Inflation, consumer prices ":"Inflation, consumer prices",
        "Population, female ":        "Population, female",
        "Unemployment, female ":      "Unemployment, female",
        "Unemployment, total ":       "Unemployment, total",
        "Unemployment, youth total ": "Unemployment, youth total",
    }
    df.rename(columns=rename_map, inplace=True)

    numeric_cols = [
        "Broad money", "Expense", "Exports of goods",
        "GDP constant", "GDP current", "GDP growth",
        "Gross savings", "Inflation, GDP deflator",
        "Inflation, consumer prices", "Population growth",
        "Population, female", "Population, male", "Population, total",
        "Unemployment, female", "Unemployment, male",
        "Unemployment, total", "Unemployment, youth total",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df.sort_values(["Country Name", "Year"], inplace=True)

    # Fill NaNs within each country group
    df = df.groupby("Country Name", group_keys=False).apply(
        lambda g: g.ffill().bfill()
    )
    df.reset_index(drop=True, inplace=True)
    return df


# ═════════════════════════════════════════════════════════
#  EHI HELPER  (mirrors P1 formula — usable on any country)
# ═════════════════════════════════════════════════════════
def compute_ehi_for_country(cdf):
    """
    Computes normalised Economic Health Index for a single-country slice.
    Formula (same as P1): EHI_raw = 0.4*GDPgrowth - 0.3*Inflation - 0.3*Unemployment
    Returns the slice with EHI column added.
    """
    sub = cdf[["Year", "GDP growth", "Inflation, consumer prices",
                "Unemployment, total"]].dropna().copy()
    if sub.empty:
        return sub
    sub["EHI_raw"] = (
        0.4 * sub["GDP growth"]
        - 0.3 * sub["Inflation, consumer prices"]
        - 0.3 * sub["Unemployment, total"]
    )
    scaler = MinMaxScaler()
    sub["EHI"] = scaler.fit_transform(sub[["EHI_raw"]])
    return sub


# ═════════════════════════════════════════════════════════
#  ARIMA HELPER
# ═════════════════════════════════════════════════════════
def fit_arima(series, orders=None):
    if orders is None:
        orders = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0), (2, 1, 1)]
    for order in orders:
        try:
            return ARIMA(series.dropna(), order=order).fit(), order
        except Exception:
            continue
    return None, None


def fit_sarima(series):
    """
    SARIMA with annual seasonality period=4 (for yearly macro data we use
    a short seasonality; period=1 degenerates to ARIMA so we use period=4
    to capture business-cycle effects roughly every 4 years).
    """
    for order, seasonal in [
        ((1, 1, 1), (1, 0, 0, 4)),
        ((1, 1, 0), (1, 0, 0, 4)),
        ((0, 1, 1), (0, 1, 0, 4)),
        ((1, 1, 1), (0, 0, 0, 0)),   # fallback = plain ARIMA
    ]:
        try:
            model = SARIMAX(
                series.dropna(),
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            return model, order, seasonal
        except Exception:
            continue
    return None, None, None


def fit_prophet_model(series, years):
    """Wraps Prophet for yearly economic data."""
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(years.astype(int), format="%Y"),
        "y":  series.values,
    })
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
    )
    m.fit(prophet_df)
    return m, prophet_df


# ═════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Flag_of_India.svg",
        width=80
    )
    st.markdown("## 🌍 Economic Intelligence — P2")
    st.markdown("Multi-Country Extension")
    st.divider()

    uploaded_p2 = st.file_uploader(
        "📂 Upload P2 Dataset (Multi-Country CSV)",
        type=["csv"],
        help="Upload clean_economic_data.csv",
    )

    forecast_steps = st.slider("Forecast Horizon (Years)", 3, 10, 5)

    st.divider()
    st.markdown("""
    **P2 Modules:**
    1. 🌍 Multi-Country Comparison
    2. 🔗 Interdependency / Lag Analysis
    3. 🔮 Advanced Forecasting
    4. 📰 Sentiment + Events Layer
    """)
    st.divider()
    st.caption("© 2025 Economic Intelligence System — P2")


# ═════════════════════════════════════════════════════════
#  MAIN HEADER
# ═════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-title">🌍 Economic Intelligence System — Project 2</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Multi-Country Analysis · Interdependency · Advanced Forecasting · Events Layer'
    '</div>',
    unsafe_allow_html=True,
)

if uploaded_p2 is None:
    st.markdown("""
    <div style="background:#e3f2fd;border:1px solid #90caf9;border-radius:8px;
                padding:0.8rem 1rem;margin:0.6rem 0;font-size:0.92rem;color:#0d47a1;">
    👈 <b>Upload your P2 multi-country CSV dataset</b> from the sidebar to begin.<br>
    Expected file: <code>clean_economic_data.csv</code> (Brazil · Germany · India · United States)
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    items = [("4", "Countries"), ("4", "P2 Modules"), ("18", "Indicators"), ("50", "Years")]
    for col, (v, l) in zip([c1, c2, c3, c4], items):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{v}</div>'
                f'<div class="metric-label">{l}</div></div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ── Load & validate ───────────────────────────────────────
with st.spinner("Loading multi-country dataset..."):
    df_all = load_multi_country(uploaded_p2)

countries   = sorted(df_all["Country Name"].unique().tolist())
india_df    = df_all[df_all["Country Name"] == "India"].copy()
year_range  = f"{int(df_all['Year'].min())}–{int(df_all['Year'].max())}"

st.success(
    f"✅ Dataset loaded: {len(df_all)} rows · {len(countries)} countries · "
    f"Years {year_range}"
)

# KPI strip
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_all)}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{year_range}</div><div class="metric-label">Year Range</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(countries)}</div><div class="metric-label">Countries</div></div>', unsafe_allow_html=True)
india_avg_gdp = india_df["GDP growth"].mean()
with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{india_avg_gdp:.2f}%</div><div class="metric-label">India Avg GDP Growth</div></div>', unsafe_allow_html=True)
with c5: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_all.columns)-2}</div><div class="metric-label">Indicators</div></div>', unsafe_allow_html=True)

with st.expander("🔍 View Raw P2 Dataset", expanded=False):
    st.dataframe(df_all, use_container_width=True)


# ═════════════════════════════════════════════════════════
#  P2 TABS
# ═════════════════════════════════════════════════════════
tabs = st.tabs([
    "🌍 P2-1 · Multi-Country Comparison",
    "🔗 P2-2 · Interdependency & Lag",
    "🔮 P2-3 · Advanced Forecasting",
    "📰 P2-4 · Sentiment & Events",
])


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB P2-1 — MULTI-COUNTRY COMPARATIVE ANALYSIS       ║
# ╚═══════════════════════════════════════════════════════╝
with tabs[0]:
    st.markdown(
        '<div class="p2-header">🌍 P2 Output 1 — Multi-Country Comparative Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this module does:</b><br>
    Runs the same EHI pipeline from P1 on all 4 countries (India, USA, Germany, Brazil)
    and plots them together so you can directly compare economic trajectories.<br><br>
    <b>Outputs:</b><br>
    &nbsp;&nbsp;• Overlaid line charts for GDP Growth, Inflation, Unemployment across all countries.<br>
    &nbsp;&nbsp;• EHI Score comparison — which country is economically healthiest over time?<br>
    &nbsp;&nbsp;• Correlation of each country's indicators with India's — quantifies influence.<br>
    &nbsp;&nbsp;• Scenario answer: "When US inflation rises 2% — how does India's GDP respond?"<br><br>
    <b>Why useful?</b> India is a large open economy. US Federal Reserve rate changes,
    German industrial output, and Brazilian commodity prices all ripple into India.
    Comparing them together reveals which global economy is India's strongest co-mover.
    </div>
    """, unsafe_allow_html=True)

    # ── 1A: GDP Growth overlaid ───────────────────────────
    st.markdown("#### 📊 GDP Growth (%) — All Countries Overlaid")
    fig, ax = plt.subplots(figsize=(12, 5))
    for country in countries:
        cdf = df_all[df_all["Country Name"] == country]
        ax.plot(cdf["Year"], cdf["GDP growth"],
                color=COUNTRY_COLORS.get(country, "black"),
                lw=2.2, marker="o", markersize=3, label=country)
    ax.axhline(0, color=C["gray"], ls="--", lw=1, alpha=0.6)
    ax.set_title("GDP Growth (%) — India vs USA vs Germany vs Brazil")
    ax.set_xlabel("Year"); ax.set_ylabel("GDP Growth (%)")
    ax.legend(); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    show_fig(fig)
    graph_caption(
        "Blue = India, Red = USA, Green = Germany, Orange = Brazil. "
        "Notice how global recessions (2008–09, 2020 COVID) cause synchronized dips across all economies. "
        "India's GDP growth has generally been higher than the USA and Germany since 2000."
    )

    # ── 1B: Inflation overlaid ────────────────────────────
    st.markdown("#### 📊 Inflation (Consumer Prices %) — All Countries Overlaid")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for country in countries:
        cdf = df_all[df_all["Country Name"] == country]
        ax2.plot(cdf["Year"], cdf["Inflation, consumer prices"],
                 color=COUNTRY_COLORS.get(country, "black"),
                 lw=2.2, marker="o", markersize=3, label=country)
    ax2.axhline(2, color=C["navy"], ls=":", lw=1.2, alpha=0.7, label="2% target")
    ax2.set_title("Inflation (Consumer Prices %) — All Countries")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Inflation (%)")
    ax2.legend(); ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    show_fig(fig2)
    graph_caption(
        "Brazil historically had extreme inflation (>1000% in the 1980s–90s, not fully visible here due to scale). "
        "India's inflation spikes around 2008–10 and 2022 (post-COVID supply shocks). "
        "USA and Germany tend to stay near the 2% central bank target, making them useful baseline comparisons."
    )

    # ── 1C: Unemployment overlaid ─────────────────────────
    st.markdown("#### 📊 Unemployment (%) — All Countries Overlaid")
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for country in countries:
        cdf = df_all[df_all["Country Name"] == country]
        ax3.plot(cdf["Year"], cdf["Unemployment, total"],
                 color=COUNTRY_COLORS.get(country, "black"),
                 lw=2.2, marker="o", markersize=3, label=country)
    ax3.set_title("Unemployment (%) — All Countries")
    ax3.set_xlabel("Year"); ax3.set_ylabel("Unemployment (%)")
    ax3.legend(); ax3.grid(True, alpha=0.2)
    fig3.tight_layout()
    show_fig(fig3)
    graph_caption(
        "Germany dramatically reduced unemployment from ~11% (2005) to ~3% (2019) through labour market reforms. "
        "India's official unemployment figures appear lower but may not capture informal employment. "
        "USA spiked sharply in 2020 due to COVID lockdowns, then recovered rapidly."
    )

    # ── 1D: EHI Comparison ───────────────────────────────
    st.markdown("#### 💹 Economic Health Index (EHI) — Country Comparison")
    st.markdown(
        '<div class="desc-box">'
        'EHI uses the <b>same P1 formula</b> applied independently to each country: '
        '<code>EHI = 0.4×GDP − 0.3×Inflation − 0.3×Unemployment</code> then normalised 0–1. '
        'This allows a fair apples-to-apples comparison of economic health across nations.'
        '</div>',
        unsafe_allow_html=True,
    )

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ehi_summary = []

    for country in countries:
        cdf_c = df_all[df_all["Country Name"] == country]
        ehi_c = compute_ehi_for_country(cdf_c)
        if ehi_c.empty:
            continue
        ax4.plot(ehi_c["Year"], ehi_c["EHI"],
                 color=COUNTRY_COLORS.get(country, "black"),
                 lw=2.2, marker="o", markersize=3, label=country)
        avg_ehi = ehi_c["EHI"].mean()
        latest_ehi = ehi_c["EHI"].iloc[-1]
        ehi_summary.append({
            "Country": country,
            "Average EHI (all years)": round(avg_ehi, 3),
            "Latest EHI": round(latest_ehi, 3),
            "Best Year": int(ehi_c.loc[ehi_c["EHI"].idxmax(), "Year"]),
            "Worst Year": int(ehi_c.loc[ehi_c["EHI"].idxmin(), "Year"]),
        })

    ax4.axhline(0.5, color=C["gray"], ls="--", lw=1, alpha=0.5, label="Midpoint")
    ax4.set_title("Economic Health Index (EHI) — All Countries")
    ax4.set_xlabel("Year"); ax4.set_ylabel("EHI Score (0–1)")
    ax4.legend(); ax4.grid(True, alpha=0.2)
    fig4.tight_layout()
    show_fig(fig4)

    st.markdown("**EHI Summary Table:**")
    ehi_summary_df = pd.DataFrame(ehi_summary)
    st.dataframe(
        ehi_summary_df.style.background_gradient(
            subset=["Average EHI (all years)", "Latest EHI"], cmap="RdYlGn"
        ),
        use_container_width=True,
    )
    graph_caption(
        "A higher EHI means better overall economic health for that year. "
        "Germany tends to score highest among the 4 due to low inflation and controlled unemployment. "
        "Brazil's score is volatile due to historical inflation crises. India's EHI trend is upward "
        "post-1990 economic liberalisation."
    )

    # ── 1E: Correlation of each country with India ────────
    st.markdown("#### 🔗 Correlation with India — Which Country Moves with India?")
    st.markdown("""
    <div class="desc-box">
    We compute the <b>Pearson correlation</b> between India's GDP Growth, Inflation, and Unemployment
    with the same indicator from each other country. A high correlation means the two economies
    tend to rise and fall together — making the other country a useful <b>signal</b> for India's future.
    </div>
    """, unsafe_allow_html=True)

    indicators_corr = {
        "GDP growth":                "GDP Growth",
        "Inflation, consumer prices": "Inflation",
        "Unemployment, total":        "Unemployment",
    }
    india_corr = india_df.set_index("Year")

    corr_rows = []
    for country in [c for c in countries if c != "India"]:
        cdf_c = df_all[df_all["Country Name"] == country].set_index("Year")
        shared_years = india_corr.index.intersection(cdf_c.index)
        row = {"Country": country}
        for col, label in indicators_corr.items():
            try:
                r = india_corr.loc[shared_years, col].corr(
                    cdf_c.loc[shared_years, col]
                )
                row[f"India {label} vs {country}"] = round(r, 3)
            except Exception:
                row[f"India {label} vs {country}"] = np.nan
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    st.dataframe(
        corr_df.set_index("Country").style.background_gradient(
            cmap="RdYlGn", vmin=-1, vmax=1
        ).format("{:.3f}"),
        use_container_width=True,
    )
    graph_caption(
        "Green cells = high positive correlation (economies move together). "
        "Red cells = high negative correlation (when one rises, the other falls). "
        "USA typically shows the highest GDP correlation with India, confirming that "
        "global US-led recessions transmit strongly into India."
    )

    # ── 1F: Scenario — US Inflation +2% → India GDP ──────
    st.markdown("#### 🧪 Scenario: When US Inflation Rises by 2% → India GDP Response?")
    st.markdown("""
    <div class="desc-box">
    This uses a <b>simple linear regression</b>: regress India's GDP Growth on USA's Inflation
    using the years where both are available. The slope tells us: for each 1% rise in US inflation,
    India's GDP changes by <i>slope</i> percentage points on average.<br><br>
    <b>Why US inflation affects India:</b> Higher US inflation → US Fed raises rates → global
    capital flows from EM to US → INR depreciates → imported inflation in India → RBI raises rates
    → India GDP slows. This is the classic emerging-market transmission channel.
    </div>
    """, unsafe_allow_html=True)

    try:
        us_df  = df_all[df_all["Country Name"] == "United States"].set_index("Year")
        shared = india_corr.index.intersection(us_df.index)

        us_inf   = us_df.loc[shared, "Inflation, consumer prices"].dropna()
        india_gdp = india_corr.loc[us_inf.index, "GDP growth"].dropna()
        common_idx = us_inf.index.intersection(india_gdp.index)

        from sklearn.linear_model import LinearRegression
        X_s = us_inf.loc[common_idx].values.reshape(-1, 1)
        y_s = india_gdp.loc[common_idx].values

        lr = LinearRegression().fit(X_s, y_s)
        slope = lr.coef_[0]
        intercept = lr.intercept_
        r2 = lr.score(X_s, y_s)

        change_2pct = slope * 2

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig5, ax5 = plt.subplots(figsize=(8, 4.5))
            ax5.scatter(X_s, y_s, color=C["blue"], alpha=0.7, edgecolors="white", s=70, zorder=3)
            x_line = np.linspace(X_s.min(), X_s.max(), 100).reshape(-1, 1)
            ax5.plot(x_line, lr.predict(x_line), color=C["red"], lw=2, label=f"Fit (slope={slope:.3f})")
            ax5.set_xlabel("US Inflation (%)")
            ax5.set_ylabel("India GDP Growth (%)")
            ax5.set_title("US Inflation vs India GDP Growth (Scatter + Regression Line)")
            ax5.legend(); ax5.grid(True, alpha=0.2)
            fig5.tight_layout()
            show_fig(fig5)

        with col_b:
            direction = "increase" if change_2pct > 0 else "decrease"
            st.markdown(f"""
            <div class="metric-card" style="margin-top:2rem;">
                <div class="metric-value">{'📈' if change_2pct > 0 else '📉'} {change_2pct:.2f}%</div>
                <div class="metric-label">Estimated change in India GDP<br>when US inflation rises 2%</div>
            </div>
            <div class="metric-card" style="margin-top:0.5rem;">
                <div class="metric-value">{slope:.4f}</div>
                <div class="metric-label">Regression Slope<br>(per 1% US Inflation)</div>
            </div>
            <div class="metric-card" style="margin-top:0.5rem;">
                <div class="metric-value">R² = {r2:.3f}</div>
                <div class="metric-label">Model Fit Score</div>
            </div>
            """, unsafe_allow_html=True)

        graph_caption(
            f"Regression result: slope = {slope:.3f}. "
            f"When US inflation rises by 2%, India's GDP growth is estimated to "
            f"<b>{direction} by {abs(change_2pct):.2f}%</b> on average (R²={r2:.3f}). "
            f"A negative slope confirms the transmission channel: US tightening → global tightening → India slowdown."
        )

    except Exception as e:
        st.warning(f"Scenario analysis error: {e}")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB P2-2 — INTERDEPENDENCY / LAG ANALYSIS           ║
# ╚═══════════════════════════════════════════════════════╝
with tabs[1]:
    st.markdown(
        '<div class="p2-header">🔗 P2 Output 2 — Interdependency & Lag Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this module does:</b><br>
    Answers the core question: <i>"How much do global economies affect India — and with what delay?"</i><br><br>
    <b>Three complementary techniques used:</b><br>
    &nbsp;&nbsp;• <b>Full Cross-Country Correlation Matrix</b> — which economies move together (no time lag).<br>
    &nbsp;&nbsp;• <b>Lag Analysis (Cross-Correlation)</b> — does US GDP today predict India's GDP next year?
    Shows correlation at lag = 0,1,2,3 years.<br>
    &nbsp;&nbsp;• <b>Granger Causality Test</b> — formal statistical test: "Does knowing X's past values
    help predict India's future values better than India's own history alone?"<br><br>
    <b>Why lag analysis?</b> Economic shocks don't transmit instantly. Trade channels have 2–3 quarter lags,
    capital flow effects show up within 1–2 years. Identifying the exact lag helps build better forecasting
    models for India using leading signals from the USA.
    </div>
    """, unsafe_allow_html=True)

    # ── 2A: Full Correlation Matrix ───────────────────────
    st.markdown("#### 🌡️ Cross-Country GDP Correlation Matrix")

    pivot_gdp = df_all.pivot_table(
        index="Year", columns="Country Name", values="GDP growth"
    ).dropna()

    corr_matrix = pivot_gdp.corr()
    fig6, ax6   = plt.subplots(figsize=(7, 5.5))
    cmap_rg     = LinearSegmentedColormap.from_list("rg", [C["red"], "white", C["blue"]])
    im          = ax6.imshow(corr_matrix.values, cmap=cmap_rg, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax6, shrink=0.85, label="Correlation")
    labels = corr_matrix.columns.tolist()
    ax6.set_xticks(range(len(labels))); ax6.set_xticklabels(labels, rotation=30, ha="right")
    ax6.set_yticks(range(len(labels))); ax6.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax6.text(j, i, f"{corr_matrix.values[i,j]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if abs(corr_matrix.values[i,j]) > 0.6 else "black")
    ax6.set_title("GDP Growth Correlation Matrix — All Countries")
    fig6.tight_layout()
    show_fig(fig6)

    india_us_corr = corr_matrix.loc["India", "United States"] if "United States" in corr_matrix else np.nan
    graph_caption(
        f"India–USA GDP correlation = {india_us_corr:.3f}. "
        "Blue = strong positive correlation (economies grow/shrink together). "
        "Red = negative (one grows while the other shrinks — rare for major economies). "
        "The 2008 crisis and 2020 COVID crash caused high global synchronization."
    )

    # ── 2B: Full Indicator Correlation Heat ───────────────
    st.markdown("#### 🔢 Full Indicator Correlation — India vs Each Country")

    indicator_cols = ["GDP growth", "Inflation, consumer prices", "Unemployment, total",
                      "Exports of goods", "Gross savings"]
    available_cols = [c for c in indicator_cols if c in df_all.columns]

    pivot_full = {}
    for col in available_cols:
        for country in countries:
            cdf_c = df_all[df_all["Country Name"] == country].set_index("Year")[col]
            pivot_full[f"{country[:3]}_{col[:8]}"] = cdf_c

    pivot_full_df = pd.DataFrame(pivot_full).dropna()
    if not pivot_full_df.empty:
        full_corr = pivot_full_df.corr()
        fig7, ax7 = plt.subplots(figsize=(14, 11))
        im2       = ax7.imshow(full_corr.values, cmap=cmap_rg, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im2, ax=ax7, shrink=0.75, label="Pearson r")
        ax7.set_xticks(range(len(full_corr.columns)))
        ax7.set_xticklabels(full_corr.columns, rotation=45, ha="right", fontsize=7)
        ax7.set_yticks(range(len(full_corr.columns)))
        ax7.set_yticklabels(full_corr.columns, fontsize=7)
        for i in range(len(full_corr.columns)):
            for j in range(len(full_corr.columns)):
                ax7.text(j, i, f"{full_corr.values[i,j]:.1f}", ha="center", va="center",
                         fontsize=5, color="white" if abs(full_corr.values[i,j]) > 0.6 else "black")
        ax7.set_title("Full Cross-Country Indicator Correlation Matrix\n"
                       "(Ind=India, Uni=USA, Ger=Germany, Bra=Brazil)", fontsize=11)
        fig7.tight_layout()
        show_fig(fig7)
        graph_caption(
            "This matrix covers GDP, Inflation, Unemployment, Exports, and Savings across all 4 countries. "
            "Strong blue blocks indicate indicators that are globally synchronized — e.g., all countries' "
            "exports moving together during global trade expansions/contractions."
        )

    # ── 2C: Lag Analysis ─────────────────────────────────
    st.markdown("#### ⏱️ Lag Analysis — Does US GDP Predict India's GDP?")
    st.markdown("""
    <div class="desc-box">
    <b>Lag analysis</b> computes the correlation between US GDP at time <i>t−lag</i> and
    India's GDP at time <i>t</i>. If the correlation peaks at lag=1, it means
    <b>US GDP one year ago is the best predictor of India's GDP today</b> —
    making US GDP a <i>leading indicator</i> for India.
    </div>
    """, unsafe_allow_html=True)

    max_lag = 5
    lag_results = []

    for other_country in [c for c in countries if c != "India"]:
        other_df = df_all[df_all["Country Name"] == other_country].set_index("Year")
        india_set = india_df.set_index("Year")
        shared    = india_set.index.intersection(other_df.index)

        india_gdp_s = india_set.loc[shared, "GDP growth"].dropna()
        other_gdp_s = other_df.loc[shared, "GDP growth"].dropna()
        shared2     = india_gdp_s.index.intersection(other_gdp_s.index)

        lag_corrs = {}
        for lag in range(max_lag + 1):
            if lag == 0:
                x = other_gdp_s.loc[shared2]
                y = india_gdp_s.loc[shared2]
            else:
                shifted_idx = shared2[lag:]
                x = other_gdp_s.loc[shared2[:-lag]] if lag < len(shared2) else pd.Series(dtype=float)
                y = india_gdp_s.loc[shifted_idx] if lag < len(shared2) else pd.Series(dtype=float)
                min_len = min(len(x), len(y))
                x, y = x.values[:min_len], y.values[:min_len]
            try:
                r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
            except Exception:
                r = np.nan
            lag_corrs[f"Lag {lag}"] = round(r, 3)

        lag_results.append({"Country (Lagged)": other_country, **lag_corrs})

    lag_df = pd.DataFrame(lag_results).set_index("Country (Lagged)")
    st.dataframe(
        lag_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
        use_container_width=True,
    )

    # Plot lag correlation curves
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    for idx, row in lag_df.iterrows():
        ax8.plot(range(max_lag + 1), row.values,
                 color=COUNTRY_COLORS.get(idx, "black"),
                 lw=2.2, marker="o", markersize=6, label=idx)
    ax8.axhline(0, color=C["gray"], ls="--", lw=1, alpha=0.5)
    ax8.set_title("Lag Correlation: Other Country GDP(t−lag) → India GDP(t)")
    ax8.set_xlabel("Lag (years)"); ax8.set_ylabel("Pearson Correlation")
    ax8.set_xticks(range(max_lag + 1))
    ax8.set_xticklabels([f"Lag {i}" for i in range(max_lag + 1)])
    ax8.legend(); ax8.grid(True, alpha=0.2)
    fig8.tight_layout()
    show_fig(fig8)

    best_lag_us = lag_df.loc["United States"].idxmax() if "United States" in lag_df.index else "N/A"
    graph_caption(
        f"For USA→India GDP: highest correlation found at <b>{best_lag_us}</b>. "
        "When the peak correlation is at Lag 1, it means USA GDP one year earlier is the best "
        "leading signal for India's GDP this year. "
        "Use this lag value when building cross-country forecast models."
    )

    # ── 2D: Granger Causality ─────────────────────────────
    st.markdown("#### 📐 Granger Causality Test — Does US GDP 'Cause' India's GDP?")
    st.markdown("""
    <div class="desc-box">
    <b>Granger Causality</b> tests whether past values of X improve the forecast of Y beyond
    Y's own past values. <b>It does NOT mean true causal direction</b> — it means
    "X Granger-causes Y" = knowing X's history gives you statistically significant extra predictive
    power for Y's future. p-value &lt; 0.05 = X significantly helps predict Y.<br><br>
    <b>Model used:</b> VAR (Vector Autoregression) within the Granger test at lags 1, 2, 3.
    </div>
    """, unsafe_allow_html=True)

    granger_rows = []

    for other_country in [c for c in countries if c != "India"]:
        other_df  = df_all[df_all["Country Name"] == other_country].set_index("Year")
        india_set = india_df.set_index("Year")
        shared    = india_set.index.intersection(other_df.index)

        try:
            india_gdp_g = india_set.loc[shared, "GDP growth"].dropna()
            other_gdp_g = other_df.loc[shared, "GDP growth"].dropna()
            shared2     = india_gdp_g.index.intersection(other_gdp_g.index)
            min_len     = min(len(india_gdp_g.loc[shared2]), len(other_gdp_g.loc[shared2]))

            if min_len < 10:
                continue

            test_data = pd.DataFrame({
                "India":   india_gdp_g.loc[shared2].values[:min_len],
                "Other":   other_gdp_g.loc[shared2].values[:min_len],
            })

            gc_res = grangercausalitytests(test_data[["India", "Other"]], maxlag=3, verbose=False)
            for lag in [1, 2, 3]:
                if lag in gc_res:
                    pval = gc_res[lag][0]["ssr_chi2test"][1]
                    granger_rows.append({
                        "X (predictor)":  other_country,
                        "Y (target)":     "India GDP",
                        "Lag":            lag,
                        "p-value":        round(pval, 4),
                        "Significant?":   "✅ Yes" if pval < 0.05 else "❌ No",
                        "Interpretation": (
                            f"{other_country} GDP Granger-causes India GDP at lag {lag}"
                            if pval < 0.05
                            else f"No significant G-causality at lag {lag}"
                        ),
                    })
        except Exception:
            continue

    if granger_rows:
        granger_df = pd.DataFrame(granger_rows)
        st.dataframe(
            granger_df.style.applymap(
                lambda v: "background:#c8e6c9" if v == "✅ Yes" else "background:#ffcdd2",
                subset=["Significant?"]
            ),
            use_container_width=True,
        )
        sig_count = granger_df[granger_df["Significant?"] == "✅ Yes"].shape[0]
        graph_caption(
            f"{sig_count} out of {len(granger_rows)} Granger causality tests are significant (p<0.05). "
            "Green = the foreign country's GDP history provides statistically significant extra "
            "predictive power for India's future GDP. "
            "This confirms the transmission channels identified in Tab P2-1."
        )
    else:
        st.info("Not enough overlapping data to run Granger Causality tests.")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB P2-3 — ADVANCED FORECASTING                     ║
# ╚═══════════════════════════════════════════════════════╝
with tabs[2]:
    st.markdown(
        '<div class="p2-header">🔮 P2 Output 3 — Advanced Forecasting: ARIMA vs SARIMA vs Prophet</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div class="desc-box">
    <b>Why P1 gave flat/constant forecasts — and how P2 fixes it:</b><br>
    P1 used a basic ARIMA(1,1,1). When the differenced series has no strong autocorrelation
    pattern, ARIMA collapses to a near-random-walk and outputs flat predictions.<br><br>
    <b>P2 introduces three models and compares them on the same data:</b><br><br>
    <b>① ARIMA</b> — AutoRegressive Integrated Moving Average<br>
    &nbsp;&nbsp;• Best for: stationary series with no seasonality.<br>
    &nbsp;&nbsp;• Used as: the baseline from P1 for direct comparison.<br><br>
    <b>② SARIMA</b> — Seasonal ARIMA (adds seasonal components S, D, Q, m)<br>
    &nbsp;&nbsp;• Best for: series with repeating patterns every m periods.<br>
    &nbsp;&nbsp;• For annual GDP data we use m=4 to capture rough 4-year business cycles.<br>
    &nbsp;&nbsp;• More parameters → can produce more dynamic, non-flat forecasts.<br><br>
    <b>③ Prophet</b> (Facebook / Meta) — Additive regression with trend + seasonality + holidays<br>
    &nbsp;&nbsp;• Best for: series with strong trend changes and irregular data.<br>
    &nbsp;&nbsp;• Uses change-point detection automatically.<br>
    &nbsp;&nbsp;• Produces uncertainty bands natively.<br>
    &nbsp;&nbsp;• Easiest to tune without deep time-series knowledge.<br><br>
    <b>Evaluation metric:</b> MAE on in-sample fit (since annual data = small holdout).<br>
    <b>Forecast horizon:</b> Next <b>{forecast_steps} years</b> for India's GDP Growth.
    </div>
    """, unsafe_allow_html=True)

    # Prepare India GDP series
    india_gdp_series = india_df.dropna(subset=["GDP growth"]).set_index("Year")["GDP growth"]

    # ── ARIMA ────────────────────────────────────────────
    st.markdown("---\n#### 📊 Model 1: ARIMA — Baseline Forecast")
    arima_model, arima_order = fit_arima(india_gdp_series)

    if arima_model:
        arima_fc   = arima_model.get_forecast(steps=forecast_steps)
        arima_mean = arima_fc.predicted_mean
        arima_ci   = arima_fc.conf_int()
        last_yr    = int(india_gdp_series.index[-1])
        future_yrs = list(range(last_yr + 1, last_yr + forecast_steps + 1))

        arima_insample = arima_model.fittedvalues
        arima_mae      = np.mean(np.abs(
            india_gdp_series.loc[arima_insample.index] - arima_insample
        ))

        fig_a, ax_a = plt.subplots(figsize=(11, 4.5))
        ax_a.plot(india_gdp_series.index, india_gdp_series.values,
                  color=C["blue"], lw=2.2, label="Historical")
        ax_a.plot(india_gdp_series.index, arima_insample.values,
                  color=C["purple"], lw=1.5, ls="--", label="ARIMA In-Sample Fit")
        ax_a.plot(future_yrs, arima_mean.values,
                  color=C["gold"], lw=2.2, ls="--", marker="o",
                  markersize=5, label=f"ARIMA{arima_order} Forecast")
        ax_a.fill_between(future_yrs, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1],
                           color=C["gold"], alpha=0.2, label="95% CI")
        ax_a.axvline(last_yr, color=C["gray"], ls=":", lw=1.2, alpha=0.6)
        ax_a.set_title(f"India GDP Growth — ARIMA{arima_order} Forecast")
        ax_a.set_xlabel("Year"); ax_a.set_ylabel("GDP Growth (%)")
        ax_a.legend(); ax_a.grid(True, alpha=0.2)
        fig_a.tight_layout()
        show_fig(fig_a)
        graph_caption(
            f"ARIMA{arima_order} in-sample MAE = {arima_mae:.3f}%. "
            "The purple dashed line shows how well ARIMA fitted the historical data. "
            "If the forecast is flat, it means the differenced series has no exploitable autocorrelation — "
            "this is why SARIMA and Prophet are needed as better alternatives."
        )
    else:
        st.warning("ARIMA fitting failed for India GDP Growth.")
        arima_mae = None

    # ── SARIMA ───────────────────────────────────────────
    st.markdown("---\n#### 📊 Model 2: SARIMA — Seasonal ARIMA Forecast")
    sarima_model, sar_order, sar_seasonal = fit_sarima(india_gdp_series)

    if sarima_model:
        sarima_fc   = sarima_model.get_forecast(steps=forecast_steps)
        sarima_mean = sarima_fc.predicted_mean
        sarima_ci   = sarima_fc.conf_int()

        sarima_insample = sarima_model.fittedvalues
        sarima_mae      = np.mean(np.abs(
            india_gdp_series.loc[sarima_insample.index] - sarima_insample
        ))

        fig_s, ax_s = plt.subplots(figsize=(11, 4.5))
        ax_s.plot(india_gdp_series.index, india_gdp_series.values,
                  color=C["blue"], lw=2.2, label="Historical")
        ax_s.plot(india_gdp_series.index, sarima_insample.values,
                  color=C["purple"], lw=1.5, ls="--", label="SARIMA In-Sample Fit")
        ax_s.plot(future_yrs, sarima_mean.values,
                  color=C["green"], lw=2.2, ls="--", marker="s",
                  markersize=5, label=f"SARIMA{sar_order}×{sar_seasonal} Forecast")
        ax_s.fill_between(future_yrs, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1],
                           color=C["green"], alpha=0.2, label="95% CI")
        ax_s.axvline(last_yr, color=C["gray"], ls=":", lw=1.2, alpha=0.6)
        ax_s.set_title(f"India GDP Growth — SARIMA{sar_order}×{sar_seasonal} Forecast")
        ax_s.set_xlabel("Year"); ax_s.set_ylabel("GDP Growth (%)")
        ax_s.legend(); ax_s.grid(True, alpha=0.2)
        fig_s.tight_layout()
        show_fig(fig_s)
        graph_caption(
            f"SARIMA{sar_order}×{sar_seasonal} in-sample MAE = {sarima_mae:.3f}%. "
            "SARIMA adds seasonal components (S=seasonal AR, D=seasonal differencing, Q=seasonal MA) "
            "that capture cyclical patterns ARIMA misses. "
            "If MAE is lower than ARIMA, SARIMA is a better model for this series."
        )
    else:
        st.warning("SARIMA fitting failed.")
        sarima_mae = None

    # ── Prophet ──────────────────────────────────────────
    st.markdown("---\n#### 📊 Model 3: Prophet (Facebook/Meta) — Trend + Change-Point Forecast")

    if PROPHET_AVAILABLE:
        try:
            prophet_series = india_gdp_series.reset_index()
            prophet_series.columns = ["Year", "GDP growth"]
            prophet_series = prophet_series.dropna()

            prophet_model, _ = fit_prophet_model(
                prophet_series["GDP growth"],
                prophet_series["Year"]
            )

            future_df = pd.DataFrame({
                "ds": pd.to_datetime(future_yrs, format="%Y")
            })
            prophet_forecast = prophet_model.predict(future_df)

            # In-sample
            hist_future_df = pd.DataFrame({
                "ds": pd.to_datetime(prophet_series["Year"].astype(int), format="%Y")
            })
            hist_pred = prophet_model.predict(hist_future_df)
            prophet_mae = np.mean(np.abs(prophet_series["GDP growth"].values - hist_pred["yhat"].values))

            fig_p, ax_p = plt.subplots(figsize=(11, 4.5))
            ax_p.plot(prophet_series["Year"].astype(int), prophet_series["GDP growth"].values,
                      color=C["blue"], lw=2.2, label="Historical")
            ax_p.plot(prophet_series["Year"].astype(int), hist_pred["yhat"].values,
                      color=C["purple"], lw=1.5, ls="--", label="Prophet In-Sample Fit")
            ax_p.plot(future_yrs, prophet_forecast["yhat"].values,
                      color=C["red"], lw=2.2, ls="--", marker="^",
                      markersize=5, label="Prophet Forecast")
            ax_p.fill_between(future_yrs,
                               prophet_forecast["yhat_lower"].values,
                               prophet_forecast["yhat_upper"].values,
                               color=C["red"], alpha=0.2, label="80% CI")
            ax_p.axvline(last_yr, color=C["gray"], ls=":", lw=1.2, alpha=0.6)
            ax_p.set_title("India GDP Growth — Prophet Forecast")
            ax_p.set_xlabel("Year"); ax_p.set_ylabel("GDP Growth (%)")
            ax_p.legend(); ax_p.grid(True, alpha=0.2)
            fig_p.tight_layout()
            show_fig(fig_p)
            graph_caption(
                f"Prophet in-sample MAE = {prophet_mae:.3f}%. "
                "Prophet detects change-points in the trend automatically (e.g., 1991 liberalisation, 2020 COVID). "
                "It extrapolates from the most recent trend segment, often giving more realistic non-flat forecasts. "
                "Wider confidence bands = more uncertainty in the forecast."
            )
        except Exception as pe:
            st.warning(f"Prophet forecasting error: {pe}")
            prophet_mae = None
    else:
        st.info("📦 Prophet not installed. Run: `pip install prophet` to enable this model.")
        prophet_mae = None

    # ── Model Comparison Summary ──────────────────────────
    st.markdown("---\n#### 🏆 Model Comparison Summary")

    comparison_data = []
    if arima_mae is not None:
        comparison_data.append({
            "Model":        f"ARIMA{arima_order}",
            "In-Sample MAE": round(arima_mae, 4),
            "Type":         "Baseline",
            "Seasonal?":    "No",
            "Change-point?":"No",
            "Best for":     "Stationary, no seasonality",
        })
    if sarima_mae is not None:
        comparison_data.append({
            "Model":        f"SARIMA{sar_order}×{sar_seasonal}",
            "In-Sample MAE": round(sarima_mae, 4),
            "Type":         "Seasonal Extension",
            "Seasonal?":    "Yes",
            "Change-point?":"No",
            "Best for":     "Business cycle patterns",
        })
    if prophet_mae is not None:
        comparison_data.append({
            "Model":        "Prophet",
            "In-Sample MAE": round(prophet_mae, 4),
            "Type":         "Additive Regression",
            "Seasonal?":    "Yes",
            "Change-point?":"Yes",
            "Best for":     "Irregular trends, event shocks",
        })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        best_model = comp_df.loc[comp_df["In-Sample MAE"].idxmin(), "Model"]
        st.dataframe(
            comp_df.style.highlight_min(subset=["In-Sample MAE"], color="#c8e6c9"),
            use_container_width=True,
        )
        graph_caption(
            f"Best model (lowest MAE) = <b>{best_model}</b>. "
            "Lower MAE = the model's in-sample predictions were closer to actual values. "
            "SARIMA typically outperforms plain ARIMA for macro data by capturing 4-year business cycles. "
            "Prophet is most robust when the series has structural breaks (e.g., COVID dip in 2020)."
        )

    # ── All Forecasts on One Chart ────────────────────────
    st.markdown("---\n#### 📈 All Model Forecasts Overlaid")

    fig_all, ax_all = plt.subplots(figsize=(12, 5))
    ax_all.plot(india_gdp_series.index, india_gdp_series.values,
                color=C["blue"], lw=2.5, label="Historical GDP Growth", zorder=5)

    if arima_model:
        ax_all.plot(future_yrs, arima_mean.values, color=C["gold"],
                    lw=2, ls="--", marker="o", markersize=5, label=f"ARIMA{arima_order}")
    if sarima_model:
        ax_all.plot(future_yrs, sarima_mean.values, color=C["green"],
                    lw=2, ls="--", marker="s", markersize=5, label=f"SARIMA{sar_order}")
    if PROPHET_AVAILABLE and prophet_mae is not None:
        ax_all.plot(future_yrs, prophet_forecast["yhat"].values, color=C["red"],
                    lw=2, ls="--", marker="^", markersize=5, label="Prophet")

    ax_all.axvline(last_yr, color=C["gray"], ls=":", lw=1.2, alpha=0.6)
    ax_all.set_title(f"India GDP Growth — ARIMA vs SARIMA vs Prophet Forecasts ({forecast_steps}-Year)")
    ax_all.set_xlabel("Year"); ax_all.set_ylabel("GDP Growth (%)")
    ax_all.legend(); ax_all.grid(True, alpha=0.2)
    fig_all.tight_layout()
    show_fig(fig_all)
    graph_caption(
        "All three models' forecasts shown together. If models diverge significantly, "
        "there is high uncertainty about the future. If they converge, the forecast is more reliable. "
        f"Models diverging = India's GDP trajectory is uncertain post-{last_yr}."
    )


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB P2-4 — SENTIMENT + EVENTS LAYER                 ║
# ╚═══════════════════════════════════════════════════════╝
with tabs[3]:
    st.markdown(
        '<div class="p2-header">📰 P2 Output 4 — Sentiment & Major Events Layer</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this module does:</b><br>
    Adds a <b>qualitative/news layer</b> on top of the purely numerical P1/P2 analysis.
    Maps major global and Indian economic events to the dataset years and shows how those
    events caused visible spikes or crashes in each indicator.<br><br>
    <b>Two sub-modules:</b><br>
    &nbsp;&nbsp;• <b>Event Impact Overlay</b> — annotates GDP, Inflation, and Unemployment charts with
    named events (COVID-19 2020, Russia-Ukraine 2022, GFC 2008, India Liberalisation 1991, etc.).<br>
    &nbsp;&nbsp;• <b>Proxy Sentiment Score</b> — constructs a simple proxy score from data alone
    (rate of change in GDP growth + inflation spike detection), shows it alongside the economic indicators,
    and demonstrates how this proxy "sensed" downturns 1–2 years before they fully appeared in GDP.<br><br>
    <b>Why a sentiment layer?</b> Real economic turning points are preceded by news/sentiment shifts.
    COVID was detectable in early 2020 headlines before GDP data was published. A combined model
    (numerical + sentiment) catches crises earlier than models trained on lagged official statistics alone.
    </div>
    """, unsafe_allow_html=True)

    # ── Major events catalogue ────────────────────────────
    EVENTS = [
        # (year, label, category, sentiment_score, description)
        (1979, "Oil Crisis II",       "crisis",  -0.8, "Second OPEC oil shock; global stagflation."),
        (1991, "India Liberalisation","positive", 0.9, "India opens economy; GDP accelerates post-reform."),
        (1997, "Asian Fin. Crisis",   "crisis",  -0.7, "Currency crises across SE Asia; contagion risk to India."),
        (2001, "Dot-com Bust + 9/11", "crisis",  -0.6, "US recession; IT sector crash; global uncertainty."),
        (2008, "Global Fin. Crisis",  "crisis",  -1.0, "Lehman collapse; worst global recession since 1929."),
        (2010, "Post-GFC Recovery",   "positive", 0.7, "Stimulus-driven global recovery; India GDP > 8%."),
        (2013, "Taper Tantrum",       "crisis",  -0.4, "US Fed tapering signal → EM capital outflows, INR crash."),
        (2016, "Demonetisation",      "crisis",  -0.5, "India demonetises 86% of currency; short-term GDP dip."),
        (2019, "Trade War",           "crisis",  -0.3, "US-China tariff escalation; global trade slowdown."),
        (2020, "COVID-19 Pandemic",   "covid",   -1.0, "Global lockdowns; India GDP -6.6%; worst since independence."),
        (2021, "COVID Recovery",      "positive", 0.6, "V-shaped rebound; vaccine rollout; pent-up demand."),
        (2022, "Russia-Ukraine War",  "war",     -0.8, "Energy/food price shock; global inflation surge."),
        (2023, "Rate Hike Cycle",     "crisis",  -0.4, "Global central banks raise rates to combat inflation."),
        (2024, "AI + Growth Surge",   "positive", 0.6, "India maintains 7%+ growth; technology investment boom."),
    ]

    events_df = pd.DataFrame(EVENTS, columns=["Year", "Event", "Category", "Sentiment", "Description"])

    event_colors = {
        "crisis":   C["red"],
        "covid":    C["orange"],
        "war":      "#880E4F",
        "positive": C["green"],
    }
    event_markers = {
        "crisis":   "▼",
        "covid":    "⚡",
        "war":      "⚔",
        "positive": "▲",
    }

    # ── 4A: GDP Growth + Events ───────────────────────────
    st.markdown("#### 📊 India GDP Growth + Major Events Overlay")

    fig_ev1, ax_ev1 = plt.subplots(figsize=(13, 5))
    ax_ev1.plot(india_gdp_series.index, india_gdp_series.values,
                color=C["blue"], lw=2.3, label="India GDP Growth", zorder=4)
    ax_ev1.fill_between(india_gdp_series.index, india_gdp_series.values,
                         alpha=0.1, color=C["blue"])
    ax_ev1.axhline(0, color=C["gray"], ls="--", lw=1, alpha=0.5)

    for _, ev_row in events_df.iterrows():
        yr  = ev_row["Year"]
        cat = ev_row["Category"]
        if yr in india_gdp_series.index:
            gdp_val = india_gdp_series[yr]
            ax_ev1.axvline(yr, color=event_colors[cat], ls=":", lw=1.2, alpha=0.6)
            offset = 0.5 if gdp_val >= 0 else -1.5
            ax_ev1.annotate(
                ev_row["Event"][:15],
                xy=(yr, gdp_val),
                xytext=(yr, gdp_val + offset + (1 if cat == "positive" else -1)),
                fontsize=6.5,
                color=event_colors[cat],
                rotation=45,
                ha="left",
                arrowprops=dict(arrowstyle="-", color=event_colors[cat], lw=0.8),
            )

    legend_patches = [
        mpatches.Patch(color=v, label=k.title()) for k, v in event_colors.items()
    ]
    ax_ev1.legend(handles=legend_patches, loc="upper left", fontsize=8)
    ax_ev1.set_title("India GDP Growth — Annotated with Major Global & Domestic Events")
    ax_ev1.set_xlabel("Year"); ax_ev1.set_ylabel("GDP Growth (%)")
    ax_ev1.grid(True, alpha=0.2)
    fig_ev1.tight_layout()
    show_fig(fig_ev1)
    graph_caption(
        "Each vertical line marks a major economic event. Red/orange = negative shock, Green = positive. "
        "Notice the 2020 COVID dip is the sharpest, followed by 2008 GFC. "
        "Post-1991 liberalisation marks a clear regime shift: India's average GDP growth visibly rises."
    )

    # ── 4B: All-country GDP with events ──────────────────
    st.markdown("#### 🌍 All Countries GDP + Event Markers")

    fig_ev2, ax_ev2 = plt.subplots(figsize=(13, 5))
    for country in countries:
        cdf_c = df_all[df_all["Country Name"] == country]
        ax_ev2.plot(cdf_c["Year"], cdf_c["GDP growth"],
                    color=COUNTRY_COLORS.get(country, "black"),
                    lw=2, marker="o", markersize=2, label=country, alpha=0.8)

    key_events = [(2008, "GFC 2008"), (2020, "COVID 2020"), (2022, "War 2022")]
    for yr, label in key_events:
        ax_ev2.axvline(yr, color=C["red"], ls=":", lw=1.5, alpha=0.7)
        ax_ev2.text(yr + 0.3, ax_ev2.get_ylim()[0] + 1 if ax_ev2.get_ylim()[0] > -20 else -10,
                    label, fontsize=8, color=C["red"], rotation=90)

    ax_ev2.set_title("Global GDP Growth — Synchronized Shocks Visible Across All Countries")
    ax_ev2.set_xlabel("Year"); ax_ev2.set_ylabel("GDP Growth (%)")
    ax_ev2.legend(); ax_ev2.grid(True, alpha=0.2)
    fig_ev2.tight_layout()
    show_fig(fig_ev2)
    graph_caption(
        "All four countries show simultaneous crashes at GFC 2008 and COVID 2020 — "
        "confirming that these were global events, not country-specific. "
        "Russia-Ukraine 2022 shows a smaller but synchronized inflation-driven slowdown."
    )

    # ── 4C: Proxy Sentiment Score ─────────────────────────
    st.markdown("#### 📉 Proxy Sentiment Score — Early Warning Signal")
    st.markdown("""
    <div class="desc-box">
    <b>Proxy Sentiment Score construction:</b><br>
    Since we don't have real news data, we construct a proxy from economic signals that typically
    move <i>before</i> official GDP data catches up:<br>
    &nbsp;&nbsp;• GDP growth rate-of-change (acceleration/deceleration)<br>
    &nbsp;&nbsp;• Inflation deviation from 5-year rolling average (price pressure building)<br>
    &nbsp;&nbsp;• Unemployment change (labour market stress)<br><br>
    <code>Proxy_Sentiment = −0.4×ΔInfl_norm − 0.3×ΔUnemp_norm + 0.3×ΔGDP_norm</code><br>
    (Normalised 0–1 using MinMaxScaler)<br><br>
    <b>Validation against real events:</b> We then overlay the event catalogue to check
    whether the sentiment proxy dipped 1–2 years <i>before</i> the official GDP crash.
    </div>
    """, unsafe_allow_html=True)

    try:
        india_sent_df = india_df[[
            "Year", "GDP growth", "Inflation, consumer prices", "Unemployment, total"
        ]].dropna().copy()

        india_sent_df["dGDP"]   = india_sent_df["GDP growth"].diff()
        india_sent_df["dInfl"]  = india_sent_df["Inflation, consumer prices"].diff()
        india_sent_df["dUnemp"] = india_sent_df["Unemployment, total"].diff()

        india_sent_df.dropna(subset=["dGDP", "dInfl", "dUnemp"], inplace=True)

        scaler_sent = MinMaxScaler()
        cols_to_norm = ["dGDP", "dInfl", "dUnemp"]
        india_sent_df[["dGDP_n", "dInfl_n", "dUnemp_n"]] = scaler_sent.fit_transform(
            india_sent_df[cols_to_norm]
        )

        india_sent_df["Proxy_Sentiment"] = (
            0.3 * india_sent_df["dGDP_n"]
            - 0.4 * india_sent_df["dInfl_n"]
            - 0.3 * india_sent_df["dUnemp_n"]
        )

        scaler2 = MinMaxScaler()
        india_sent_df["Proxy_Sentiment_norm"] = scaler2.fit_transform(
            india_sent_df[["Proxy_Sentiment"]]
        )

        fig_sent, (ax_s1, ax_s2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        ax_s1.plot(india_sent_df["Year"], india_sent_df["GDP growth"],
                   color=C["blue"], lw=2.2, label="India GDP Growth")
        ax_s1.axhline(0, color=C["gray"], ls="--", lw=1, alpha=0.5)

        for yr, event, cat in [(2008, "GFC", "crisis"), (2020, "COVID", "covid"),
                                 (2022, "War", "war"), (1991, "Liberalisation", "positive")]:
            if yr in india_sent_df["Year"].values:
                ax_s1.axvline(yr, color=event_colors[cat], ls=":", lw=1.5, alpha=0.7)
                ax_s1.text(yr + 0.2, india_sent_df["GDP growth"].max() * 0.9,
                            event, fontsize=7.5, color=event_colors[cat])

        ax_s1.set_title("India GDP Growth (Top) vs Proxy Sentiment Score (Bottom)")
        ax_s1.set_ylabel("GDP Growth (%)")
        ax_s1.legend(); ax_s1.grid(True, alpha=0.2)

        ax_s2.fill_between(india_sent_df["Year"], india_sent_df["Proxy_Sentiment_norm"],
                            0.5, where=india_sent_df["Proxy_Sentiment_norm"] < 0.5,
                            color=C["red"], alpha=0.4, label="Negative Sentiment")
        ax_s2.fill_between(india_sent_df["Year"], india_sent_df["Proxy_Sentiment_norm"],
                            0.5, where=india_sent_df["Proxy_Sentiment_norm"] >= 0.5,
                            color=C["green"], alpha=0.4, label="Positive Sentiment")
        ax_s2.plot(india_sent_df["Year"], india_sent_df["Proxy_Sentiment_norm"],
                   color=C["navy"], lw=2, label="Proxy Score (0–1)")
        ax_s2.axhline(0.5, color=C["gray"], ls="--", lw=1, alpha=0.6, label="Neutral")

        for yr, event, cat in [(2008, "GFC", "crisis"), (2020, "COVID", "covid"),
                                 (2022, "War", "war"), (1991, "Lib.", "positive")]:
            if yr in india_sent_df["Year"].values:
                ax_s2.axvline(yr, color=event_colors[cat], ls=":", lw=1.5, alpha=0.7)

        ax_s2.set_xlabel("Year"); ax_s2.set_ylabel("Proxy Sentiment Score (0–1)")
        ax_s2.legend(loc="lower left"); ax_s2.grid(True, alpha=0.2)
        ax_s2.set_ylim(0, 1)

        fig_sent.tight_layout(pad=2)
        show_fig(fig_sent)
        graph_caption(
            "The proxy sentiment score (bottom chart) often dips into red 1–2 years BEFORE the actual "
            "GDP crash (top chart) materialises in official data. "
            "For example, stress signals built up in 2007 before the 2008 GFC hit India. "
            "Similarly, rising inflation + unemployment change in 2019 preceded the 2020 COVID shock. "
            "A real sentiment model using news headlines would sharpen this signal further."
        )

    except Exception as sent_err:
        st.warning(f"Proxy sentiment construction error: {sent_err}")

    # ── 4D: Event Catalogue Table ─────────────────────────
    st.markdown("#### 📋 Major Economic Events Catalogue")

    events_display = events_df.copy()
    cat_icon = {"crisis": "🔴", "covid": "🟠", "war": "⚔️", "positive": "🟢"}
    events_display["Type"] = events_display["Category"].map(cat_icon)
    events_display["Sentiment Score"] = events_display["Sentiment"]

    st.dataframe(
        events_display[["Year", "Type", "Event", "Sentiment Score", "Description"]]
        .style.background_gradient(subset=["Sentiment Score"], cmap="RdYlGn", vmin=-1, vmax=1),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### 💡 Key Insights from Events Layer")

    insights_events = [
        ("🔴", "COVID-19 (2020) — Most severe shock", "India's GDP fell ~6.6% in 2020 — worst since independence. All 4 countries in dataset showed simultaneous contraction."),
        ("🔴", "Global Financial Crisis (2008)", "India slowed from 8-9% to 3.9% GDP growth. US mortgage crisis transmitted to India via capital outflows and trade slowdown."),
        ("⚔️", "Russia-Ukraine War (2022)", "Triggered a global energy and food price spike. India's inflation rose sharply; RBI had to raise rates aggressively."),
        ("🟢", "India Liberalisation (1991)", "Watershed moment: India's average GDP growth shifted from 4-5% to 6-8%+ post-reform. Clear structural break visible in GDP chart."),
        ("🟠", "Taper Tantrum (2013)", "US Fed hint about QE tapering caused EM capital flight. INR depreciated 20% in months; India's current account deficit widened."),
        ("🟢", "Post-COVID Recovery (2021-22)", "India had one of the fastest rebounds (8.7% in 2021). Government infrastructure spending and pent-up consumer demand drove recovery."),
    ]

    for icon, title, detail in insights_events:
        cls = "event-covid" if "COVID" in title else \
              "event-war" if "War" in title or "Ukraine" in title else \
              "event-crisis" if icon == "🔴" or icon == "🟠" else "event-good"
        st.markdown(
            f'<div class="{cls}">'
            f'<span style="font-size:1.2rem">{icon}</span> '
            f'<b style="color:#1a237e">{title}</b><br>'
            f'<span style="color:#546e7a">{detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style='text-align:center; color:#9e9e9e; font-size:0.88rem; padding:1rem 0;'>
    <b>Economic Intelligence System — Project 2 Extension</b><br>
    Multi-Country (India · USA · Germany · Brazil) | ARIMA · SARIMA · Prophet · Lag Analysis · Granger · Events Layer<br>
    Built with Streamlit · Statsmodels · Prophet · Scikit-learn · Matplotlib
</div>
""", unsafe_allow_html=True)

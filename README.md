# 📊 Economic Intelligence Prediction & Risk Analysis System

> **PMF Project 1 + Project 2** — India 50-Year Economic Analysis + Multi-Country Comparative Platform  
> Built with Streamlit · Scikit-learn · Statsmodels · Prophet · Matplotlib


## 🌐 Overview

This is a two-part data science + machine learning project built as a fully interactive **Streamlit web application** for economic analysis and forecasting.

**Project 1** focuses on **India's 50-year economic history** (1974–2024) — forecasting key indicators, predicting GDP growth using ML regression models, classifying economic conditions, and detecting risk.

**Project 2** extends the platform to **four countries (India, USA, Germany, Brazil)** — comparing economic health across nations, quantifying interdependencies using lag analysis and Granger Causality, introducing advanced forecasting models (SARIMA + Prophet), and adding a qualitative events/sentiment layer.

---

## 📁 Project Structure

```
economic-intelligence-system/
│
├── economic_intelligence_app.py     # Project 1 — India 50-Year Analysis
├── economic_intelligence_p2.py      # Project 2 — Multi-Country Extension
│
├── data/
│   ├── PMF_50yrs_data.csv           # P1 dataset — India (upload in app)
│   └── clean_economic_data.csv      # P2 dataset — 4 countries (upload in app)
│
├── docs/
│   └── model_reference_guide.docx   # Full model explanation document
│
└── README.md
```

---

## ✨ Features

### Project 1 — India 50-Year Analysis

| Tab | Module | What it does |
|-----|--------|-------------|
| 📈 Output 1 | **ARIMA Forecasting** | Forecasts GDP Growth, Inflation, and Unemployment for the next N years using ARIMA with ADF stationarity tests |
| 🔮 Output 2 | **Regression Modeling** | Predicts GDP growth using 8 economic features — Random Forest, XGBoost, and Linear Regression compared side by side |
| 💹 Output 3 | **Economic Health Index** | Custom composite EHI score (0–1) per year using the formula: `0.4×GDP − 0.3×Inflation − 0.3×Unemployment` |
| 🏷️ Output 4 | **Classification** | Classifies each year as Strong / Stable / Risky / Recession using Decision Tree, Random Forest, and Logistic Regression |
| ⚠️ Output 5 | **Risk Detection** | Rule-based threshold alert system with a 50-year historical risk timeline |
| 🔗 Output 6 | **Correlation & Insights** | Pearson correlation heatmap + multi-indicator trend plots + automated data-driven insights |

### Project 2 — Multi-Country Extension

| Tab | Module | What it does |
|-----|--------|-------------|
| 🌍 P2-1 | **Multi-Country Comparison** | GDP, Inflation, Unemployment, and EHI overlaid across India, USA, Germany, Brazil. Includes US inflation → India GDP scenario analysis |
| 🔗 P2-2 | **Interdependency & Lag Analysis** | Cross-country correlation matrix, lag analysis (does US GDP today predict India's next year?), Granger Causality test |
| 🔮 P2-3 | **Advanced Forecasting** | ARIMA vs SARIMA vs Prophet — side-by-side comparison with MAE metric and all forecasts overlaid on a single chart |
| 📰 P2-4 | **Sentiment & Events Layer** | 14 major global events annotated on charts + a proxy sentiment early-warning score built from rate-of-change signals |

---

## 📊 Dataset

### Project 1 Dataset
- **File:** `PMF_50yrs_data.csv`
- **Coverage:** India, ~1974–2024 (50 years)
- **Key columns:** GDP growth (annual %), Inflation CPI, Unemployment total, Real interest rate, Government debt, Trade (% of GDP), Population, and more

### Project 2 Dataset
- **File:** `clean_economic_data.csv`
- **Countries:** Brazil, Germany, India, United States
- **Coverage:** 1976–2025 (50 years × 4 countries = 200 rows)
- **Columns (19 total):**

| Column | Description |
|--------|-------------|
| `Country Name` | One of: Brazil, Germany, India, United States |
| `Year` | 1976–2025 |
| `Broad money` | Broad money supply (% of GDP) |
| `Expense` | Government expense (% of GDP) |
| `Exports of goods` | Exports (% of GDP) |
| `GDP constant` | GDP in constant 2015 US$ |
| `GDP current` | GDP in current US$ |
| `GDP growth` | Annual GDP growth rate (%) |
| `Gross savings` | Gross savings (% of GDP) |
| `Inflation, GDP deflator` | GDP deflator inflation (%) |
| `Inflation, consumer prices` | CPI Inflation (%) |
| `Population growth` | Annual population growth (%) |
| `Population, female` | Female population (% of total) |
| `Population, male` | Male population (% of total) |
| `Population, total` | Total population |
| `Unemployment, female` | Female unemployment (% of female labour force) |
| `Unemployment, male` | Male unemployment (% of male labour force) |
| `Unemployment, total` | Total unemployment (% of total labour force) |
| `Unemployment, youth total` | Youth unemployment (% of ages 15–24) |

> **Source:** World Bank Open Data

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Web App** | `streamlit` |
| **Data Processing** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` (Random Forest, XGBoost, Linear/Logistic Regression, Decision Tree) |
| **Time Series** | `statsmodels` (ARIMA, SARIMA, ADF Test, Granger Causality), `prophet` |
| **Visualisation** | `matplotlib` |
| **Preprocessing** | `sklearn.preprocessing` (MinMaxScaler, LabelEncoder) |

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sufia-Sayyed/economic-intelligence-system.git
cd economic-intelligence-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib scikit-learn statsmodels prophet
```

> **Note:** XGBoost is optional but recommended for Output 2.
> ```bash
> pip install xgboost
> ```

> **Prophet installation note:** On some systems, Prophet requires additional build tools.
> ```bash
> pip install prophet
> # If that fails on Windows:
> conda install -c conda-forge prophet
> ```

---

## ▶️ How to Run

### Run Project 1 (India Analysis)

```bash
streamlit run economic_intelligence_app.py
```

### Run Project 2 (Multi-Country Extension)

```bash
streamlit run economic_intelligence_p2.py
```

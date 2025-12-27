# 🏦 RatingLens: Corporate Credit Rating Analysis

RatingLens is an interactive **Streamlit** application that predicts **corporate default risk** using 16 fundamental financial ratios.  
It leverages an **XGBoost classifier** as the primary Probability of Default (PD) engine and benchmarks **13 alternative machine learning models**.

The project converts raw financial statements into a transparent, quantitative **credit risk framework** suitable for:
- Credit & equity analysts  
- Risk managers  
- Finance / data science recruiters  

---

## ✨ Core Features

| Feature | Description | Output |
|------|------------|--------|
| **XGBoost PD Engine** | Learns non-linear relationships between 16 financial ratios | Probability of Default (0–100%) |
| **Risk Band Calibration** | Converts PD into intuitive risk buckets | Low / Medium / High |
| **Financial Ratio KPIs** | Displays key leverage, liquidity, profitability, efficiency ratios | KPI cards per company |
| **Feature Importance** | Explains drivers of default risk | Top-driver bar chart |
| **13-Model Benchmarking** | Compares ML algorithms on the same dataset | AUC, Accuracy, Precision, Recall |
| **Educational Explainability** | Explains ML logic & financial formulas | Expandable theory sections |

---

## 🧱 Data & Label Engineering

**Dataset:**  
`data/corporateCreditRatingWithFinancialRatios.csv`

### Target Variable

The raw dataset contains an agency-provided column:

- `Binary Rating`
  - `1` → Safe / Investment Grade  
  - `0` → Risky / Junk  

To model **Probability of Default (PD)**, RatingLens flips the label:

```math
Default =
\begin{cases}
1 & \text{if Binary Rating} = 0 \\
0 & \text{if Binary Rating} = 1
\end{cases}
```

This produces a clean binary outcome:
- **1 = Default-prone**
- **0 = Financially safe**

### Identifiers

- `Company_ID`  
  - Uses **Ticker** if available, else **Corporation name**
- `Display_Name`  
  - `"Corporation (Ticker)"`  
  - Used for Streamlit dropdown selection

---

## 📊 Feature Set (16 Financial Ratios)

Defined in `config.py` and enforced via `data_engine.py`.

### Liquidity
- `Current Ratio`

### Leverage
- `Long-term Debt / Capital`
- `Debt/Equity Ratio`

### Profitability & Margins
- `Gross Margin`
- `Operating Margin`
- `EBIT Margin`
- `EBITDA Margin`
- `Pre-Tax Profit Margin`
- `Net Profit Margin`

### Efficiency
- `Asset Turnover`

### Returns
- `ROE - Return On Equity`
- `Return On Tangible Equity`
- `ROA - Return On Assets`
- `ROI - Return On Investment`

### Cash Flow
- `Operating Cash Flow Per Share`
- `Free Cash Flow Per Share`

**Preprocessing rules:**
- All features coerced to numeric
- Missing / invalid values filled with `0.0`
- Ensures model stability and consistent inference

---

## 📐 Key Financial Formulas

### 1. Leverage (Solvency)

```math
Debt\text{-}to\text{-}Equity =
\frac{Total\ Liabilities}{Shareholders'\ Equity}
```

- Higher leverage increases sensitivity to earnings volatility
- Excessive debt → higher default risk

---

### 2. Liquidity (Short-Term Health)

```math
Current\ Ratio =
\frac{Current\ Assets}{Current\ Liabilities}
```

- `< 1.0` indicates potential liquidity stress
- Important for near-term solvency

---

### 3. Profitability & Efficiency

```math
ROA =
\frac{Net\ Income}{Total\ Assets}
```

```math
Operating\ Margin =
\frac{Operating\ Income\ (EBIT)}{Revenue}
```

- Capture efficiency of asset and revenue utilization
- Strong profitability offsets leverage risk

---

## 🤖 XGBoost Credit Risk Engine

### Modeling Pipeline

1. User selects a company via dropdown  
2. App extracts its 16 financial ratios  
3. Features standardized using `StandardScaler`  
4. Input passed to trained `XGBClassifier`  
5. Raw score mapped to **Probability of Default (PD)**  
6. PD converted into calibrated risk bands  

---

### Objective Function

XGBoost minimizes a **regularized loss**:

```math
\mathcal{L}(\phi) =
\sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)
```

Where:
- `l(·)` = classification loss (log-loss)
- `Ω(f_k)` = regularization term to control model complexity

---

### Additive Tree Ensemble

```math
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
```

Final prediction mapped to PD using sigmoid:

```math
P(Default \mid x_i) =
\frac{1}{1 + e^{-\hat{y}_i}}
```

---

### Risk Band Calibration

```math
Risk =
\begin{cases}
Low\ Risk & PD \le 10\% \\
Medium\ Risk & 10\% < PD \le 40\% \\
High\ Risk & PD > 40\%
\end{cases}
```

Displayed as:
- Gauge chart (green / amber / red)
- Textual risk classification
- External rating comparison

---

### Feature Importance

If available:
- Uses `feature_importances_`
- Displays **Top 10 drivers** of default risk
- Visualized as horizontal bar chart

---

## 📊 Dashboard Structure

### 🔍 Tab 1 — Credit Default Analysis (XGBoost)

**Outputs**
- Probability of Default gauge
- Risk classification label
- External agency rating

**KPIs**
- Debt/Equity
- Current Ratio
- Operating Margin
- ROA

**Explainability**
- Feature importance chart
- XGBoost theory
- Financial ratio explanations

---

### ⚙️ Tab 2 — Model Benchmarking (13 Models)

**Algorithms**
- Tree & Ensemble: XGBoost, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, Decision Tree
- Linear / Probabilistic: Logistic Regression, GaussianNB, LDA, QDA
- Distance / Margin / Neural: KNN, SVC, MLP

**Metrics**
- AUC (primary)
- Accuracy
- Precision
- Recall

**Visuals**
- AUC comparison bar chart
- Styled leaderboard table

---

## 🚀 Quick Start

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Rating-Lens.git
cd RatingLens
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

Open browser at:  
`http://localhost:8501`

---

## 📦 Example requirements.txt

```text
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
```

---

## 🎓 Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy
- **Core Model:** XGBoost
- **Benchmarking:** scikit-learn
- **Scaling:** StandardScaler

---

## ⚖️ Disclaimer

This project is intended **strictly for educational and research purposes**.

It does **not** constitute:
- Investment advice
- Lending decisions
- Regulatory guidance

Real-world credit risk assessment must incorporate:
- Macroeconomic variables
- Qualitative judgment
- Regulatory and policy constraints

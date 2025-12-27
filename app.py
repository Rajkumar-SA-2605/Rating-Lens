import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Modular Imports
from src.config import DATA_PATH
from src.data_engine import load_and_clean_data
from src.risk_model import CreditRiskModel, ModelBenchmarker

st.set_page_config(page_title="RatingLens: Corporate Credit Rating Analysis", page_icon="🏦", layout="wide")

@st.cache_data
def get_data():
    return load_and_clean_data(DATA_PATH)

@st.cache_resource
def get_risk_model(df):
    model = CreditRiskModel()
    model.train(df)
    return model

def main():
    st.title("🏦 RatingLens: Corporate Credit Rating Analysis")
    st.markdown("""
    **Project Objective:** To develop a robust, AI-driven framework for predicting corporate credit default risk. 
    By analyzing 16 distinct financial ratios, this system replaces subjective human judgment with quantitative probability assessments.
    """)

    data = get_data()
    if data is None:
        st.error(f"Failed to load data from {DATA_PATH}.")
        st.stop()
        
    risk_model = get_risk_model(data)

    tab1, tab2 = st.tabs(["🔍 Credit Default Analysis (XGBoost)", "⚙️ Model Benchmarking (Comparative Study)"])

    # --- TAB 1: XGBOOST ANALYSIS ---
    with tab1:
        st.subheader("Single Entity Risk Assessment")
        all_companies = data['Display_Name'].unique()
        selected_company_name = st.selectbox("Select Corporation:", all_companies)
        
        if selected_company_name:
            row = data[data['Display_Name'] == selected_company_name].iloc[0]
            input_df = pd.DataFrame([row])
            
            pd_val = risk_model.predict_pd(input_df)[0]
            risk_label = risk_model.calibrate_risk_level(pd_val)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Risk Probability (PD)", f"{pd_val:.2%}")
                st.metric("Implied Risk Level", risk_label)
                st.info(f"Actual Agency Rating: **{row.get('Rating', 'N/A')}**")

            with c2:
                # GAUGE CHART
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pd_val * 100,
                    title = {'text': "Risk Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if pd_val > 0.4 else "orange" if pd_val > 0.1 else "green"},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 40], 'color': "navajowhite"},
                            {'range': [40, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 40}
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=50))
                st.plotly_chart(fig, width='stretch')

            st.divider()
            st.subheader("Key Financial Drivers")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Debt/Equity", f"{row['Debt/Equity Ratio']:.2f}", help="Total Debt / Total Equity. > 2.0 is typically high.")
            k2.metric("Current Ratio", f"{row['Current Ratio']:.2f}", help="Current Assets / Current Liabilities. < 1.0 indicates liquidity stress.")
            k3.metric("Operating Margin", f"{row['Operating Margin']:.1f}%", help="Profit from core business operations.")
            k4.metric("ROA", f"{row['ROA - Return On Assets']:.1f}%", help="Efficiency of asset utilization.")
            
            st.divider()
            
            # --- FEATURE IMPORTANCE ---
            st.subheader("📊 Risk Driver Analysis")
            importance_df = risk_model.get_feature_importance()
            if importance_df is not None:
                fig_imp = px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                st.plotly_chart(fig_imp, width='stretch')

            # --- DEEP DIVE SECTION (NEW) ---
            st.divider()
            st.subheader("📘 Project Deep Dive: How XGBoost Works")
            
            with st.expander("1. XGBoost Model Architecture & Formulas", expanded=True):
                st.markdown("""
                The **Extreme Gradient Boosting (XGBoost)** model is an ensemble of decision trees. It does not use a single linear formula (like $y = mx+b$). Instead, it uses an additive strategy where new trees correct the errors of previous trees.
                
                **The Mathematical Objective:**
                The model minimizes a regularized objective function:
                """)
                st.latex(r'''\mathcal{L}(\phi) = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k)''')
                st.markdown("""
                Where:
                - $l$ is the **Loss Function** (measuring the difference between prediction $\hat{y}$ and actual $y$).
                - $\Omega$ is the **Regularization Term** (penalizes complex trees to prevent overfitting).
                
                **Prediction Logic:**
                The final prediction is the sum of scores from $K$ trees:
                """)
                st.latex(r'''\hat{y}_i = \sum_{k=1}^K f_k(x_i), \quad f_k \in \mathcal{F}''')
                st.markdown("""
                Finally, this raw sum is converted into a probability (0% to 100%) using the **Sigmoid Function**:
                """)
                st.latex(r'''P(Default) = \frac{1}{1 + e^{-\hat{y}_i}}''')

            with st.expander("2. Project Objectives & Effectiveness"):
                st.markdown("""
                **Objectives:**
                1.  **Automate Credit Rating:** Remove human bias from the initial screening process.
                2.  **Early Warning System:** Detect subtle combinations of ratios (e.g., deteriorating Cash Flow + Rising Debt) that signal distress before a default occurs.
                
                **Why this is effective:**
                *   **Non-Linearity:** Traditional models assume relationships are straight lines. XGBoost understands that "Debt is bad" only if "Cash Flow is low." If Cash Flow is high, high Debt might actually be good (leverage for growth).
                *   **Robustness:** It handles missing data and outliers better than standard regression.
                """)

            with st.expander("3. Fundamental Financial Formulas Used"):
                c_f1, c_f2 = st.columns(2)
                with c_f1:
                    st.markdown("**Leverage (Solvency)**")
                    st.latex(r'''\text{Debt/Equity} = \frac{\text{Total Liabilities}}{\text{Shareholders' Equity}}''')
                    st.caption("Measures the degree to which a company is financing its operations through debt.")
                    
                    st.markdown("**Liquidity (Short-term Health)**")
                    st.latex(r'''\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}''')
                    st.caption("Ability to pay off short-term obligations with short-term assets.")
                
                with c_f2:
                    st.markdown("**Profitability (Efficiency)**")
                    st.latex(r'''\text{ROA} = \frac{\text{Net Income}}{\text{Total Assets}}''')
                    st.caption("Indicator of how profitable a company is relative to its total assets.")
                    
                    st.markdown("**Operating Performance**")
                    st.latex(r'''\text{Operating Margin} = \frac{\text{Operating Income (EBIT)}}{\text{Revenue}}''')
                    st.caption("Profit after variable costs but before interest or tax.")

    # --- TAB 2: BENCHMARKING ---
    with tab2:
        st.subheader("🏆 Extensive Model Performance Benchmark")
        st.markdown("""
        This section compares **13 different Machine Learning Algorithms** to identify the most accurate classifier for this dataset. 
        We evaluate them based on **AUC (Area Under Curve)**, which measures the ability to distinguish between defaulters and non-defaulters.
        """)
        
        if st.button("🚀 Run Benchmark Experiment"):
            with st.spinner("Training 13 models... (This may take 15-30 seconds)"):
                benchmarker = ModelBenchmarker()
                results_df = benchmarker.run_benchmark(data)
                
                st.success("Benchmark Complete!")
                best_model = results_df.iloc[0]
                st.metric("Top Performing Model", f"{best_model['Model']} (AUC: {best_model['AUC Score']:.3f})")
                
                fig_bar = px.bar(results_df, x='Model', y='AUC Score', color='Model', text_auto='.3f', title="AUC Score by Algorithm")
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, width='stretch')
                
                st.subheader("Detailed Metrics Table")
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), width='stretch')

                st.divider()
                st.subheader("🎓 Educational: How Each Model Works")
                
                # ENSEMBLE MODELS
                with st.expander("🌲 Ensemble & Tree-Based Models (Generally Best Performers)", expanded=True):
                    st.markdown("""
                    *   **XGBoost:** "The Specialist." Builds trees sequentially. Each new tree focuses specifically on fixing the errors made by the previous trees. Very accurate but can overfit.
                    *   **Random Forest:** "The Democracy." Builds hundreds of independent trees in parallel. Each tree gets a vote, and the majority decision wins. Reduces variance and is very stable.
                    *   **Gradient Boosting:** Similar to XGBoost but uses a standard gradient descent algorithm. Good for structured data.
                    *   **Extra Trees:** "Extremely Randomized Trees." Similar to Random Forest but chooses split points randomly. Often faster and sometimes more robust to noise.
                    *   **AdaBoost:** "Adaptive Boosting." Increases the weight of misclassified data points so the next classifier is forced to focus on the hard cases.
                    *   **Decision Tree:** The simplest unit. Asks a sequence of Yes/No questions to split data. Easy to interpret but prone to bias/overfitting on its own.
                    """)
                
                # LINEAR & PROBABILISTIC
                with st.expander("📈 Linear & Probabilistic Models"):
                    st.markdown("""
                    *   **Logistic Regression:** The "Baseline." Estimates the probability of default using a linear combination of features passed through a sigmoid function. Simple, fast, and interpretable.
                    *   **Gaussian Naive Bayes:** Assumes all features are independent (e.g., Debt has no relation to Profit). Fast, but this assumption is rarely true in finance.
                    *   **LDA (Linear Discriminant Analysis):** Projects data onto a lower-dimensional space to maximize the separation between the two classes (Default vs. Non-Default).
                    *   **QDA (Quadratic Discriminant Analysis):** Similar to LDA but allows for curved boundaries between classes, making it more flexible.
                    """)
                
                # DISTANCE & NEURAL
                with st.expander("🧠 Distance & Neural Models"):
                    st.markdown("""
                    *   **KNN (K-Nearest Neighbors):** "Guilty by Association." Finds the 5 most similar companies in the dataset. If the majority of them defaulted, it predicts this one will too.
                    *   **SVC (Support Vector Classifier):** Finds the optimal "hyperplane" (boundary) that separates the two classes with the widest possible margin. Great for high-dimensional spaces.
                    *   **Neural Network (MLP):** "Deep Learning Lite." A Multi-Layer Perceptron that mimics the human brain using layers of interconnected nodes. Can learn extremely complex, non-linear patterns but requires lots of data.
                    """)

if __name__ == "__main__":
    main()

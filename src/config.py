import os

DATA_PATH = os.path.join("data", "corporateCreditRatingWithFinancialRatios.csv")

FEATURES = [
    'Current Ratio', 'Long-term Debt / Capital', 'Debt/Equity Ratio', 
    'Gross Margin', 'Operating Margin', 'EBIT Margin', 'EBITDA Margin', 
    'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover', 
    'ROE - Return On Equity', 'Return On Tangible Equity', 
    'ROA - Return On Assets', 'ROI - Return On Investment', 
    'Operating Cash Flow Per Share', 'Free Cash Flow Per Share'
]

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42
}

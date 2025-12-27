import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Classifiers (Your original list)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from src.config import FEATURES, XGB_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskModel:
    def __init__(self):
        self.model = XGBClassifier(**XGB_PARAMS)
        self.scaler = StandardScaler()
        self.features = FEATURES
        self.is_fitted = False

    def train(self, df: pd.DataFrame):
        X = df[self.features]
        y = df['Default']
        
        # Scale on full training set
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            # Fallback for UI flow if train wasn't called explicitly
            return np.array([0.5]) 
            
        X = df[self.features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'Feature': self.features,
                'Importance': self.model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        return None

    @staticmethod
    def calibrate_risk_level(pd: float) -> str:
        if pd <= 0.10: return "Low Risk (Investment Grade)"
        elif pd <= 0.40: return "Medium Risk"
        else: return "High Risk (Non-Investment Grade)"

class ModelBenchmarker:
    def __init__(self):
        # YOUR ORIGINAL 13 MODELS
        self.models = {
            "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVC (Prob)": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Gaussian NB": GaussianNB(),
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
            "Neural Network (MLP)": MLPClassifier(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        self.features = FEATURES

    def run_benchmark(self, df: pd.DataFrame):
        X = df[self.features]
        y = df['Default']
        
        # Split -> Scale to prevent leakage
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        results = []
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = 0.5
                
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC Score": auc,
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0)
                })
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                
        return pd.DataFrame(results).sort_values(by="AUC Score", ascending=False)

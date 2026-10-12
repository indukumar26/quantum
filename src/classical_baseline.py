"""
Classical Baseline Models Module
================================
Trains and evaluates classical ML baselines on identical biomedical features:
  - Logistic Regression (Primary Baseline with class_weight='balanced')
  - Support Vector Classifier (RBF kernel, balanced)
  - Random Forest Classifier (balanced)

Computes diagnostic performance metrics:
  - Recall (Primary Metric for clinical arrhythmia screening)
  - F1-Score
  - Accuracy
  - Precision
  - ROC-AUC
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

class ClassicalBaselines:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Logistic Regression": LogisticRegression(
                class_weight='balanced',
                C=1.0,
                max_iter=1000,
                random_state=random_state
            ),
            "Support Vector Classifier (RBF)": SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                max_depth=5,
                random_state=random_state
            )
        }
        self.fitted_models = {}

    def fit_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Fits all baseline models and computes clinical classification metrics."""
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
            
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
                
            acc = float(accuracy_score(y_test, y_pred))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            results[name] = {
                "accuracy": acc,
                "recall": rec,          # Primary Metric
                "f1_score": f1,
                "precision": prec,
                "roc_auc": auc,
                "confusion_matrix": cm,
                "y_pred": y_pred,
                "y_prob": y_prob
            }
        return results

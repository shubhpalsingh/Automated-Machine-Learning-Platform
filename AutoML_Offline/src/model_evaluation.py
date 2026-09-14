"""
model_evaluation.py

Evaluates trained machine learning models and compares performance.
Supports both classification and regression metrics.
"""

import pandas as pd

# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Regression metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_classification_models(models, X_test, y_test):
    """
    Evaluate classification models using common metrics.
    """

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    return pd.DataFrame(results)


def evaluate_regression_models(models, X_test, y_test):
    """
    Evaluate regression models using regression metrics.
    """

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results.append({
            "Model": name,
            "R2 Score": r2,
            "MAE": mae,
            "MSE": mse
        })

    return pd.DataFrame(results)
"""
model_training.py

Handles training of machine learning models.
Supports both classification and regression problems.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import pandas as pd

def split_dataset(X, y, problem_type, test_size=0.2, random_state=42):
    y = pd.Series(y).squeeze()

    stratify_y = None
    if problem_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify_y = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    return X_train, X_test, y_train, y_test

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', probability=True),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200),
    }

    return models


def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=200),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVM": SVR(kernel='rbf'),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200),
    }

    return models

def scale_data(X_train, X_test):
    """
    Scale data using StandardScaler (for models that need it).
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def train_models(models, X_train, y_train):
    """
    Train multiple models.

    Parameters
    ----------
    models : dict
        Dictionary of models
    X_train : DataFrame
    y_train : Series

    Returns
    -------
    dict
        Trained models
    """

    trained_models = {} 

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models
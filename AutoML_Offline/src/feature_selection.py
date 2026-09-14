"""
feature_selection.py

Handles selection of feature columns (X) and target column (y)
for machine learning models.
"""

import pandas as pd


def get_available_columns(df: pd.DataFrame) -> list:
    """
    Returns all dataset columns.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list
    """

    return df.columns.tolist()


def validate_target_column(df: pd.DataFrame, target_column: str):
    """
    Validates that the target column exists in dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    target_column : str

    Raises
    ------
    ValueError
    """

    if target_column not in df.columns:
        raise ValueError("Selected target column does not exist.")


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list:
    """
    Returns list of feature columns excluding target.

    Parameters
    ----------
    df : pandas.DataFrame
    target_column : str

    Returns
    -------
    list
    """

    return [col for col in df.columns if col != target_column]


def split_features_target(df: pd.DataFrame, feature_columns: list, target_column: str):
    """
    Splits dataset into features (X) and target (y).

    Parameters
    ----------
    df : pandas.DataFrame
    feature_columns : list
    target_column : str

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series
    """

    if not feature_columns:
        raise ValueError("No feature columns selected.")

    if target_column in feature_columns:
        raise ValueError("Target column cannot be included in feature columns.")

    X = df[feature_columns]
    y = df[target_column]

    return X, y


def detect_problem_type(y: pd.Series) -> str:
    """
    Detect whether problem is classification or regression.

    Parameters
    ----------
    y : pandas.Series

    Returns
    -------
    str
        'classification' or 'regression'
    """

    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    else:
        return "regression"
"""
data_cleaning.py

Provides functions for handling missing values
and basic dataset cleaning operations.
"""

import pandas as pd


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing missing values per column.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_values": df.isnull().sum(),
        "missing_percentage": (df.isnull().sum() / len(df)) * 100
    })

    return missing_report


def fill_missing_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with mean.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    return df_clean


def fill_missing_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with median.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    return df_clean


def fill_missing_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in categorical columns using mode.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df_clean = df.copy()

    categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        mode_value = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_value, inplace=True)

    return df_clean


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing missing values.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    return df.dropna()


def drop_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns with too many missing values.

    Parameters
    ----------
    df : pandas.DataFrame
    threshold : float
        Maximum allowed missing ratio (default 50%)

    Returns
    -------
    pandas.DataFrame
    """

    df_clean = df.copy()

    missing_ratio = df_clean.isnull().mean()

    columns_to_drop = missing_ratio[missing_ratio > threshold].index

    df_clean.drop(columns=columns_to_drop, inplace=True)

    return df_clean
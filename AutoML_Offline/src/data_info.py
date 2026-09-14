"""
data_info.py

Provides functions to extract dataset information
for exploratory data analysis (EDA).
"""

import pandas as pd


def get_dataset_shape(df: pd.DataFrame) -> dict:
    """
    Returns dataset shape information.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Number of rows and columns
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }


def get_column_data_types(df: pd.DataFrame) -> dict:
    """
    Returns column data types.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Column names and their data types
    """

    return df.dtypes.astype(str).to_dict()


def get_missing_values(df: pd.DataFrame) -> dict:
    """
    Returns missing value counts per column.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Missing value count for each column
    """

    return df.isnull().sum().to_dict()


def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns statistical summary of numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Descriptive statistics
    """

    return df.describe()


def categorize_columns(df: pd.DataFrame) -> dict:
    """
    Categorize dataset columns into numeric and categorical.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Numeric and categorical column lists
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    }


def get_unique_values(df: pd.DataFrame, column: str) -> list:
    """
    Returns unique values of a specific column.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str

    Returns
    -------
    list
    """

    if column not in df.columns:
        raise ValueError("Column not found in dataset.")

    return df[column].dropna().unique().tolist()
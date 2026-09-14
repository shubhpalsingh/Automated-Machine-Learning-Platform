"""
data_loader.py

Handles loading datasets (CSV / Excel) for the AutoML application.
"""

import pandas as pd


def load_dataset(uploaded_file):
    """
    Load dataset from uploaded file.

    Parameters
    ----------
    uploaded_file : file-like object
        File uploaded through Streamlit uploader.

    Returns
    -------
    df : pandas.DataFrame
        Loaded dataset.
    file_type : str
        Type of file loaded (csv or excel).
    """

    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    file_name = uploaded_file.name.lower()

    # Detect file type
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        file_type = "csv"

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        file_type = "excel"

    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    return df, file_type


def validate_dataset(df):
    """
    Perform basic dataset validation.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Dataset summary information.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }

    return summary


def preview_dataset(df, n=5):
    """
    Return preview rows of dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    n : int
        Number of rows to preview.

    Returns
    -------
    pandas.DataFrame
    """
    return df.head(n)
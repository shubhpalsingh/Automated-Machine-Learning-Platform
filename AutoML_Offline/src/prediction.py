"""
prediction.py

Handles prediction using trained machine learning models.
"""

import pandas as pd


def prepare_input_data(feature_columns, input_values):
    """
    Convert user input into DataFrame format required by ML model.

    Parameters
    ----------
    feature_columns : list
        List of feature names used for training
    input_values : dict
        Dictionary of user input values

    Returns
    -------
    pandas.DataFrame
    """

    input_df = pd.DataFrame([input_values], columns=feature_columns)

    return input_df


def make_prediction(model, input_df):
    """
    Generate prediction using trained model.

    Parameters
    ----------
    model : trained ML model
    input_df : pandas.DataFrame

    Returns
    -------
    prediction
    """

    prediction = model.predict(input_df)

    return prediction[0]


def get_prediction_probability(model, input_df):
    """
    Get prediction probability for classification models.

    Parameters
    ----------
    model : trained ML model
    input_df : pandas.DataFrame

    Returns
    -------
    probability : list or None
    """

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)
        return prob[0]

    return None
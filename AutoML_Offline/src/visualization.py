"""
visualization.py

Provides functions to create visualizations for dataset exploration.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_bar(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Bar Plot - {col}")
    return fig


def plot_scatter(df, x, y):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Scatter Plot ({x} vs {y})")
    return fig


def plot_pie(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title(f"Pie Chart - {col}")
    return fig


def plot_histogram(df, col):
    fig, ax = plt.subplots()
    df[col].hist(ax=ax)
    ax.set_title(f"Histogram - {col}")
    return fig


def plot_heatmap(df):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def plot_box(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Box Plot - {col}")
    return fig


def plot_violin(df, col):
    fig, ax = plt.subplots()
    sns.violinplot(y=df[col], ax=ax)
    ax.set_title(f"Violin Plot - {col}")
    return fig


def plot_3d(df, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title("3D Plot")
    return fig

def plot_histogram(df: pd.DataFrame, column: str):
    """
    Plot histogram for a numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
    """

    if column not in df.columns:
        raise ValueError("Column not found in dataset.")

    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    return plt


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plot correlation heatmap for numeric columns.
    """

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numeric columns for correlation heatmap.")

    plt.figure(figsize=(8,6))
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    return plt


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Scatter plot between two numeric columns.
    """

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Selected columns not found in dataset.")

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f"{x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    return plt


def plot_bar_chart(df: pd.DataFrame, column: str):
    """
    Bar chart for categorical column counts.
    """

    if column not in df.columns:
        raise ValueError("Column not found.")

    plt.figure(figsize=(6,4))
    df[column].value_counts().plot(kind="bar")

    plt.title(f"Category Count: {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()

    return plt


def get_numeric_columns(df: pd.DataFrame):
    """
    Return numeric columns list.
    """

    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame):
    """
    Return categorical columns list.
    """

    return df.select_dtypes(include=["object", "category"]).columns.tolist()
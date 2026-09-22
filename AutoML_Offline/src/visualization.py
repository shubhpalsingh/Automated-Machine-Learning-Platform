"""
visualization.py

Provides functions to create visualizations for dataset exploration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CATEGORIES = 15
NUMERIC_BINS = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _prepare_group_column(df, column):
    """
    Prepare a column for grouped visualizations.

    - Numeric columns with many unique values are converted
      into bins.
    - Categorical columns with too many categories are reduced
      to the most frequent categories + 'Other'.
    """

    data = df.copy()

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    # --------------------------------------------------------
    # Numeric column
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(data[column]):

        unique_values = data[column].nunique()

        # Too many numeric values -> create bins
        if unique_values > MAX_CATEGORIES:

            data["_group_column"] = pd.cut(
                data[column],
                bins=NUMERIC_BINS,
                precision=2
            )

        else:
            data["_group_column"] = data[column]

    # --------------------------------------------------------
    # Categorical column
    # --------------------------------------------------------

    else:

        value_counts = data[column].value_counts()

        if len(value_counts) > MAX_CATEGORIES:

            top_categories = value_counts.nlargest(
                MAX_CATEGORIES
            ).index

            data["_group_column"] = data[column].where(
                data[column].isin(top_categories),
                "Other"
            )

        else:
            data["_group_column"] = data[column]

    return data


# ============================================================
# BAR PLOT
# ============================================================

def plot_bar(df, x_col, y_col):
    """
    Create a bar plot.

    X-axis:
        Categorical column or binned numeric column.

    Y-axis:
        Numeric column.

    For numeric X columns with many unique values,
    values are automatically grouped into bins.
    """

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found.")

    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found.")

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError("Y-axis must be a numeric column.")

    data = _prepare_group_column(df, x_col)

    grouped = (
        data.groupby(
            "_group_column",
            observed=True
        )[y_col]
        .mean()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    grouped.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(f"Average {y_col}")
    ax.set_title(f"{y_col} by {x_col}")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    return fig


# ============================================================
# SCATTER PLOT
# ============================================================

def plot_scatter(df, x_col, y_col):
    """
    Create a scatter plot between two numeric columns.
    """

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Selected columns not found in dataset.")

    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise ValueError("X-axis must be numeric.")

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError("Y-axis must be numeric.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df[x_col],
        df[y_col],
        alpha=0.6
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")

    plt.tight_layout()

    return fig


# ============================================================
# PIE CHART
# ============================================================

def plot_pie(df, col):
    """
    Create a pie chart.

    If a column has many categories, only the most frequent
    categories are shown and the remaining values are grouped
    into 'Other'.

    Numeric columns with many unique values are automatically
    binned.
    """

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    data = _prepare_group_column(df, col)

    counts = data["_group_column"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))

    counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")
    ax.set_title(f"Distribution of {col}")

    plt.tight_layout()

    return fig


# ============================================================
# HISTOGRAM
# ============================================================

def plot_histogram(df, column):
    """
    Create a histogram for a numeric column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("Histogram requires a numeric column.")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


# ============================================================
# HEATMAP
# ============================================================

def plot_heatmap(df):
    """
    Create a correlation heatmap using numeric columns only.
    """

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two numeric columns are required "
            "for a correlation heatmap."
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()

    return fig


# ============================================================
# BOX PLOT
# ============================================================

def plot_box(df, x_col, y_col):
    """
    Create a box plot.

    X-axis:
        Categorical column or binned numeric column.

    Y-axis:
        Numeric column.
    """

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found.")

    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found.")

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError("Y-axis must be numeric.")

    data = _prepare_group_column(df, x_col)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=data,
        x="_group_column",
        y=y_col,
        ax=ax
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {x_col}")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    return fig


# ============================================================
# VIOLIN PLOT
# ============================================================

def plot_violin(df, x_col, y_col):
    """
    Create a violin plot.

    X-axis:
        Categorical column or binned numeric column.

    Y-axis:
        Numeric column.
    """

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found.")

    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found.")

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError("Y-axis must be numeric.")

    data = _prepare_group_column(df, x_col)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(
        data=data,
        x="_group_column",
        y=y_col,
        ax=ax
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {x_col}")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    return fig


# ============================================================
# 3D PLOT
# ============================================================

def plot_3d(df, x, y, z):
    """
    Create a 3D scatter plot.

    X, Y and Z must all be numeric.
    """

    for column in [x, y, z]:

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} must be numeric for a 3D plot."
            )

    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        df[x],
        df[y],
        df[z],
        alpha=0.6
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    ax.set_title(
        f"3D Plot: {x}, {y}, {z}"
    )

    plt.tight_layout()

    return fig


# ============================================================
# COLUMN HELPERS
# ============================================================

def get_numeric_columns(df):
    """
    Return numeric columns.
    """

    return df.select_dtypes(
        include=["number"]
    ).columns.tolist()


def get_categorical_columns(df):
    """
    Return categorical columns.
    """

    return df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

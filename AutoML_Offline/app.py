
import streamlit as st

# Import modules
from src.data_loader import load_dataset, validate_dataset, preview_dataset
from src.data_info import (
    get_dataset_shape,
    get_column_data_types,
    get_missing_values,
    get_statistical_summary,
)
from src.data_cleaning import (
    get_missing_value_report,
    fill_missing_mean,
    fill_missing_median,
)
from src.feature_selection import (
    get_available_columns,
    split_features_target,
    detect_problem_type,
)
from src.visualization import (
    plot_bar,
    plot_scatter,
    plot_pie,
    plot_histogram,
    plot_heatmap,
    plot_box,
    plot_violin,
    plot_3d,
)
from src.model_training import (
    split_dataset,
    get_classification_models,
    get_regression_models,
    train_models,
)
from src.model_evaluation import (
    evaluate_classification_models,
    evaluate_regression_models,
)
from src.prediction import prepare_input_data, make_prediction

st.markdown("""
<style>

/* Sidebar button style */
section[data-testid="stSidebar"] button {
    width: 100%;
    padding: 12px;
    margin-bottom: 8px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    background-color: #FA7807;
    border: none;
    transition: all 0.25s ease;
}

/* Hover animation */
section[data-testid="stSidebar"] button:hover {
    background-color: #4CAF50;
    color: white;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# Page title
st.title("Automated Machine Learning Platform")


# ==============================
# SIDEBAR NAVIGATION
# ==============================

st.sidebar.title("Pipeline Steps")

# Default step
if "step" not in st.session_state:
    st.session_state.step = "Upload Dataset"

# Sidebar buttons
if st.sidebar.button("Upload Dataset"):
    st.session_state.step = "Upload Dataset"

if st.sidebar.button("Dataset Information"):
    st.session_state.step = "Dataset Information"

if st.sidebar.button("Data Cleaning"):
    st.session_state.step = "Data Cleaning"

if st.sidebar.button("Feature Selection"):
    st.session_state.step = "Feature Selection"

if st.sidebar.button("Data Visualization"):
    st.session_state.step = "Data Visualization"

if st.sidebar.button("Model Training"):
    st.session_state.step = "Model Training"

if st.sidebar.button("Model Evaluation"):
    st.session_state.step = "Model Evaluation"

if st.sidebar.button("Prediction"):
    st.session_state.step = "Prediction"

step = st.session_state.step


# ==============================
# 1 DATA UPLOAD
# ==============================

if step == "Upload Dataset":

    st.header("1. Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", type=["csv", "xlsx"]
    )

    if uploaded_file:

        df, file_type = load_dataset(uploaded_file)

        st.session_state["df"] = df

        st.success("Dataset Loaded Successfully")

        st.dataframe(preview_dataset(df))


# ==============================
# 2 DATASET INFORMATION
# ==============================

elif step == "Dataset Information":

    st.header("2. Dataset Information")

    if "df" not in st.session_state:

        st.warning("Please upload dataset first")

    else:

        df = st.session_state["df"]

        shape = get_dataset_shape(df)

        st.write("Rows:", shape["rows"])
        st.write("Columns:", shape["columns"])

        st.subheader("Dataset Preview")
        st.dataframe(preview_dataset(df))

        st.subheader("Column Data Types")
        st.json(get_column_data_types(df))

        st.subheader("Missing Values")
        st.json(get_missing_values(df))

        st.subheader("Statistical Summary")
        st.dataframe(get_statistical_summary(df))


# ==============================
# 3 DATA CLEANING
# ==============================

elif step == "Data Cleaning":

    st.header("3. Data Cleaning")

    if "df" not in st.session_state:

        st.warning("Upload dataset first")

    else:

        df = st.session_state["df"]

        st.subheader("Missing Value Report")
        st.dataframe(get_missing_value_report(df))

        clean_option = st.selectbox(
            "Choose missing value handling method",
            ["None", "Fill with Mean", "Fill with Median"],
        )

        if st.button("Apply Cleaning"):

            if clean_option == "Fill with Mean":

                df = fill_missing_mean(df)

            elif clean_option == "Fill with Median":

                df = fill_missing_median(df)

            st.session_state["df"] = df

            st.success("Cleaning applied successfully")


# ==============================
# 4 FEATURE SELECTION
# ==============================

elif step == "Feature Selection":

    st.header("4. Feature Selection")

    if "df" not in st.session_state:

        st.warning("Upload dataset first")

    else:

        df = st.session_state["df"]

        columns = get_available_columns(df)

        target_column = st.selectbox("Select Target Column", columns)

        feature_columns = st.multiselect(
            "Select Feature Columns",
            [col for col in columns if col != target_column],
        )

        if feature_columns:

            X, y = split_features_target(df, feature_columns, target_column)

            # Encode categorical features
            X = X.copy()
            for col in X.select_dtypes(include=["object"]).columns:
                X[col] = X[col].astype("category").cat.codes

            problem_type = detect_problem_type(y)

            st.success(f"Detected Problem Type: {problem_type}")

            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["feature_columns"] = feature_columns
            st.session_state["problem_type"] = problem_type


# ==============================
# 5 DATA VISUALIZATION
# ==============================

elif step == "Data Visualization":

    st.header("5. Data Visualization")

    if "df" not in st.session_state:
        st.warning("Upload dataset first")

    else:
        df = st.session_state["df"]

        # Separate column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_cols = df.columns.tolist()

        plot_type = st.selectbox(
            "Select Plot Type",
            [
                "Bar Graph",
                "Scatter Plot",
                "Pie Chart",
                "Histogram",
                "Heatmap",
                "Box Plot",
                "Violin Plot",
                "3D Plot",
            ],
        )

        # -------------------------
        # Dynamic Column Selection
        # -------------------------

        if plot_type == "Bar Graph":
            col = st.selectbox("Select Column", all_cols)
            fig = plot_bar(df, col)
            st.pyplot(fig)

        elif plot_type == "Scatter Plot":
            x = st.selectbox("X Axis", numeric_cols)
            y = st.selectbox("Y Axis", numeric_cols)
            fig = plot_scatter(df, x, y)
            st.pyplot(fig)

        elif plot_type == "Pie Chart":
            col = st.selectbox("Select Column", all_cols)
            fig = plot_pie(df, col)
            st.pyplot(fig)

        elif plot_type == "Histogram":
            col = st.selectbox("Select Column", numeric_cols)
            fig = plot_histogram(df, col)
            st.pyplot(fig)

        elif plot_type == "Heatmap":
            fig = plot_heatmap(df[numeric_cols])
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            col = st.selectbox("Select Column", numeric_cols)
            fig = plot_box(df, col)
            st.pyplot(fig)

        elif plot_type == "Violin Plot":
            col = st.selectbox("Select Column", numeric_cols)
            fig = plot_violin(df, col)
            st.pyplot(fig)

        elif plot_type == "3D Plot":
            x = st.selectbox("X Axis", numeric_cols)
            y = st.selectbox("Y Axis", numeric_cols)
            z = st.selectbox("Z Axis", numeric_cols)
            fig = plot_3d(df, x, y, z)
            st.pyplot(fig)


# ==============================
# 6 MODEL TRAINING
# ==============================

elif step == "Model Training":

    st.header("6. Model Training")

    if "X" not in st.session_state:

        st.warning("Complete Feature Selection first")

    else:

        X = st.session_state["X"]
        y = st.session_state["y"]
        problem_type = st.session_state["problem_type"]

        X_train, X_test, y_train, y_test = split_dataset(X, y, problem_type)

        if problem_type == "classification":

            models = get_classification_models()

        else:

            models = get_regression_models()

        if st.button("Start Model Training"):

            progress = st.progress(0)

            with st.spinner("Training models..."):

                trained_models = train_models(models, X_train, y_train)

                progress.progress(100)

            st.success("Models trained successfully")

            st.session_state["trained_models"] = trained_models
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test


# ==============================
# 7 MODEL EVALUATION
# ==============================

elif step == "Model Evaluation":

    st.header("7. Model Evaluation")

    if "trained_models" not in st.session_state:

        st.warning("Train models first")

    else:

        trained_models = st.session_state["trained_models"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        problem_type = st.session_state["problem_type"]

        if problem_type == "classification":

            results = evaluate_classification_models(
                trained_models, X_test, y_test
            )

        else:

            results = evaluate_regression_models(
                trained_models, X_test, y_test
            )

        st.dataframe(results)


# ==============================
# 8 PREDICTION
# ==============================

elif step == "Prediction":

    st.header("8. Prediction")

    if "trained_models" not in st.session_state:

        st.warning("Train models first")

    else:

        trained_models = st.session_state["trained_models"]
        feature_columns = st.session_state["feature_columns"]

        model_names = list(trained_models.keys())

        selected_model_name = st.selectbox(
            "Select Model for Prediction", model_names
        )

        selected_model = trained_models[selected_model_name]

        st.subheader("Enter Feature Values")

        input_values = {}

        for col in feature_columns:

            value = st.number_input(f"Enter value for {col}")

            input_values[col] = value

        if st.button("Predict"):

            input_df = prepare_input_data(feature_columns, input_values)

            prediction = make_prediction(selected_model, input_df)

            st.success(f"Prediction Result: {prediction}")

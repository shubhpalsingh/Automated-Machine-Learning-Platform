import streamlit as st
import pandas as pd
import numpy as np
import io

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
    plot_bar, plot_scatter, plot_pie, plot_histogram,
    plot_heatmap, plot_box, plot_violin, plot_3d,
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


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================
# PIPELINE STEPS DEFINITION & PREREQUISITES
# ==============================
STEPS = [
    {"label": "Upload", "icon": "📁", "key": "Upload Dataset"},
    {"label": "Info", "icon": "ℹ️", "key": "Dataset Information"},
    {"label": "Cleaning", "icon": "🧹", "key": "Data Cleaning"},
    {"label": "Features", "icon": "🎯", "key": "Feature Selection"},
    {"label": "Visualize", "icon": "📈", "key": "Data Visualization"},
    {"label": "Training", "icon": "🚀", "key": "Model Training"},
    {"label": "Evaluate", "icon": "📊", "key": "Model Evaluation"},
    {"label": "Predict", "icon": "🔮", "key": "Prediction"},
]


# ==============================
# SESSION STATE INITIALIZATION
# ==============================
if "step_index" not in st.session_state:
    st.session_state.step_index = 0
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None


# ==============================
# VALIDATION ENGINE
# ==============================
def is_step_complete(step_key):
    """Determines if a step's requirements have been completed successfully."""
    if step_key == "Upload Dataset":
        return st.session_state.df is not None
    elif step_key in ["Dataset Information", "Data Cleaning", "Data Visualization"]:
        return st.session_state.df is not None
    elif step_key == "Feature Selection":
        return "X" in st.session_state and "y" in st.session_state
    elif step_key == "Model Training":
        return "trained_models" in st.session_state
    elif step_key in ["Model Evaluation", "Prediction"]:
        return "trained_models" in st.session_state
    return False


def is_step_accessible(index):
    """Enforces logical progression. Users cannot skip incomplete work."""
    if index == 0:
        return True
    
    # Prerequisite map
    prereqs = {
        1: 0,  # Info requires Upload done
        2: 0,  # Cleaning requires Upload done
        3: 0,  # Features requires Upload done
        4: 0,  # Visualization requires Upload done
        5: 3,  # Training requires Feature Selection done
        6: 5,  # Evaluation requires Training done
        7: 5,  # Prediction requires Training done
    }
    
    prereq_idx = prereqs.get(index, 0)
    prereq_key = STEPS[prereq_idx]["key"]
    return is_step_complete(prereq_key)


# ==============================
# CUSTOM STYLING (MODERN DARK TECH)
# ==============================
st.markdown("""
<style>
    /* Global layout adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Gradient Typography */
    .main-title {
        background: linear-gradient(90deg, #FA7807 0%, #FF9A3C 50%, #FFD09B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 850;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #8C92AC;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    /* ==========================================
       TOP PROGRESS PIPELINE (EXACT ALIGNMENT)
       ========================================== */
    .pipeline-outer-container {
        background: #111217;
        padding: 24px 20px 16px 20px;
        border-radius: 16px;
        border: 1px solid #23252F;
        margin-bottom: 35px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        position: relative;
    }

    /* Background track (Full Width) */
    .pipeline-track-bg {
        position: absolute;
        top: 20px;
        left: 6.25%; /* Aligned perfectly to step 1 center */
        right: 6.25%;
        height: 4px;
        background-color: #23252F;
        z-index: 1;
        border-radius: 2px;
    }

    /* Active progress filling track */
    .pipeline-track-fill {
        position: absolute;
        top: 20px;
        left: 6.25%;
        height: 4px;
        background: linear-gradient(90deg, #4CAF50 0%, #FA7807 100%);
        z-index: 2;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 2px;
        box-shadow: 0 0 12px rgba(250, 120, 7, 0.4);
    }

    /* Nodes styling */
    .pipeline-node {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 3;
        text-align: center;
    }

    .node-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: bold;
        background: #181920;
        border: 3px solid #2D313F;
        color: #5C627A;
        transition: all 0.4s ease;
    }

    /* Node states */
    .node-circle.completed {
        background: #102A18;
        border-color: #4CAF50;
        color: #4CAF50;
    }

    .node-circle.active {
        background: #2E1B0E;
        border-color: #FA7807;
        color: #FA7807;
        transform: scale(1.12);
        box-shadow: 0 0 20px rgba(250, 120, 7, 0.6);
    }

    .node-label {
        font-size: 11px;
        font-weight: 600;
        color: #5C627A;
        margin-top: 8px;
        transition: color 0.3s ease;
    }

    .node-label.completed {
        color: #4CAF50;
    }

    .node-label.active {
        color: #FA7807;
        font-weight: 750;
    }

    /* ==========================================
       SIDEBAR & WIDGET STYLING
       ========================================== */
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 750;
        color: #FA7807;
        padding-bottom: 8px;
        border-bottom: 2px solid #23252F;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        padding: 8px 12px;
        font-size: 13.5px;
        font-weight: 600;
        border-radius: 8px;
        background-color: #181920;
        color: #A3A9C2;
        border: 1px solid #23252F;
        text-align: left;
        transition: all 0.2s ease;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: #FA7807;
        color: #FA7807;
        transform: translateX(3px);
    }

    /* ==========================================
       NEXT/PREV CONTROLLER BAR
       ========================================== */
    .control-panel {
        background: #111217;
        padding: 20px;
        border-radius: 12px;
        border-top: 3px solid #FA7807;
        margin-top: 50px;
    }

    div[data-testid="column"]:nth-child(3) .stButton button {
        background: linear-gradient(135deg, #FA7807 0%, #FF9A3C 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 12px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(250, 120, 7, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="column"]:nth-child(3) .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(250, 120, 7, 0.5) !important;
    }

    div[data-testid="column"]:nth-child(1) .stButton button {
        background: #1E2029 !important;
        color: #D1D5DB !important;
        border: 1px solid #2D313F !important;
        padding: 12px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="column"]:nth-child(1) .stButton button:hover {
        background: #252834 !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# PIPELINE STEP RENDERER (FIXED)
# ==============================
def render_pipeline_bar(current_idx):
    """Renders the top stepper with static step names and a dynamic progression fill."""
    total_steps = len(STEPS)
    
    # Calculate exact progress center percent
    fill_percentage = (current_idx / (total_steps - 1)) * 100 if total_steps > 1 else 0

    # Start Container
    html = '<div class="pipeline-outer-container"><div class="pipeline-container">'
    html += '<div class="pipeline-track-bg"></div>'
    html += f'<div class="pipeline-track-fill" style="width: {fill_percentage * 0.875}%;"></div>'

    # Build individual Nodes on a single line to prevent Streamlit parsing errors
    for i, step in enumerate(STEPS):
        completed = is_step_complete(step["key"])
        is_active = (i == current_idx)
        
        if is_active:
            circle_class = "active"
            label_class = "active"
            content = step["icon"]
        elif completed:
            circle_class = "completed"
            label_class = "completed"
            content = "✓"
        else:
            circle_class = ""
            label_class = ""
            content = step["icon"]

        html += f'<div class="pipeline-node">'
        html += f'<div class="node-circle {circle_class}">{content}</div>'
        html += f'<div class="node-label {label_class}">{step["label"]}</div>'
        html += f'</div>'
    
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


# ==============================
# BOTTOM STEP CONTROLLER
# ==============================
def render_bottom_navigation(current_idx):
    """Render Next / Previous controls based on prerequisites validation."""
    st.markdown("<br><hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_idx > 0:
            prev_label = f"⬅️ Back to {STEPS[current_idx - 1]['label']}"
            if st.button(prev_label, key="prev_step_btn", use_container_width=True):
                st.session_state.step_index = current_idx - 1
                st.rerun()

    with col2:
        current_key = STEPS[current_idx]["key"]
        st.markdown(
            f"<div style='text-align:center; padding-top:12px; color:#5C627A; font-size:14px;'>"
            f"Currently Active: <strong style='color:#FA7807;'>Step {current_idx + 1} - {current_key}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col3:
        if current_idx < len(STEPS) - 1:
            next_label = f"Go to {STEPS[current_idx + 1]['label']} ➡️"
            can_go_next = is_step_complete(current_key)
            
            if st.button(next_label, key="next_step_btn", use_container_width=True, disabled=not can_go_next):
                st.session_state.step_index = current_idx + 1
                st.rerun()
        else:
            st.button("✨ Complete Pipeline", key="end_pipeline_btn", use_container_width=True, disabled=True)

    # Prompt action-required if they are stuck
    if current_idx < len(STEPS) - 1 and not is_step_complete(STEPS[current_idx]["key"]):
        st.warning(f"👉 Please complete the current action on this page to unlock the next phase: **{STEPS[current_idx + 1]['label']}**")


# ==============================
# RENDER LAYOUT HEADERS
# ==============================
st.markdown('<div class="main-title">🤖 Automated Machine Learning Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An advanced user-guided machine learning pipeline</div>', unsafe_allow_html=True)

# Render the dynamic horizontal stepper
render_pipeline_bar(st.session_state.step_index)


# ==============================
# SIDEBAR CONTROL
# ==============================
st.sidebar.markdown('<div class="sidebar-title">🧭 Quick Jump</div>', unsafe_allow_html=True)

for i, step in enumerate(STEPS):
    completed = is_step_complete(step["key"])
    accessible = is_step_accessible(i)
    
    status_icon = "✅ " if completed else "🔒 " if not accessible else "⚪ "
    active_marker = "▶ " if i == st.session_state.step_index else ""
    
    btn_label = f"{active_marker}{status_icon}{step['icon']} {step['key']}"
    
    if st.sidebar.button(btn_label, key=f"sidebar_nav_{step['key']}", disabled=not accessible):
        st.session_state.step_index = i
        st.rerun()

st.sidebar.markdown("---")

if st.session_state.df is not None:
    st.sidebar.markdown("### 📌 Active Data Frame")
    st.sidebar.info(f"**Rows:** {st.session_state.df.shape[0]:,}\n\n**Columns:** {st.session_state.df.shape[1]}")

    if st.sidebar.button("🔄 Reset Project"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# Extract operational active step
active_step_key = STEPS[st.session_state.step_index]["key"]


# ==============================
# 1. DATA UPLOAD
# ==============================
if active_step_key == "Upload Dataset":
    st.header("📁 Upload Your Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"],
            help="Supported formats: CSV (.csv), Excel (.xlsx)"
        )

    with col2:
        st.markdown("### 💡 Setup Tips")
        st.markdown("""
        - Clean column names with headers work best
        - The target feature column should ideally not contain empty rows
        - Keep file sizes reasonable for fast performance
        """)

    if uploaded_file:
        try:
            df, file_type = load_dataset(uploaded_file)
            st.session_state["df"] = df
            st.session_state["original_df"] = df.copy()

            st.success(f"✅ Dataset loaded successfully! ({file_type.upper()})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Rows Included", f"{df.shape[0]:,}")
            c2.metric("📋 Total Columns", df.shape[1])
            c3.metric("❓ Missing Cells", int(df.isnull().sum().sum()))
            c4.metric("💾 Memory Footprint", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

            st.subheader("👀 Dataset Preview")
            st.dataframe(preview_dataset(df), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


# ==============================
# 2. DATASET INFORMATION
# ==============================
elif active_step_key == "Dataset Information":
    st.header("ℹ️ Dataset Information")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first")
    else:
        df = st.session_state["df"]
        shape = get_dataset_shape(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{shape['rows']:,}")
        c2.metric("Columns", shape['columns'])
        c3.metric("Numeric Fields", len(df.select_dtypes(include=['number']).columns))
        c4.metric("Categorical Fields", len(df.select_dtypes(include=['object', 'category']).columns))

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Quick Preview", "🔤 Data Schema types", "❓ Unassigned Nulls", "📊 Descriptive Statistics"])

        with tab1:
            n_rows = st.slider("Rows to display", 5, min(100, len(df)), 10)
            st.dataframe(df.head(n_rows), use_container_width=True)

        with tab2:
            dtypes_df = pd.DataFrame({
                "Column Name": df.columns,
                "Structure Type": [str(dt) for dt in df.dtypes],
                "Cardinality (Unique)": [df[col].nunique() for col in df.columns],
                "First Non-Null Sample": [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A" for col in df.columns]
            })
            st.dataframe(dtypes_df, use_container_width=True)

        with tab3:
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({
                "Variable Label": missing.index,
                "Null Quantity": missing.values,
                "Proportion %": (missing.values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df["Null Quantity"] > 0]

            if len(missing_df) == 0:
                st.success("🎉 Excellent! Zero missing attributes detected inside this dataset.")
            else:
                st.dataframe(missing_df, use_container_width=True)

        with tab4:
            st.dataframe(get_statistical_summary(df), use_container_width=True)


# ==============================
# 3. DATA CLEANING
# ==============================
elif active_step_key == "Data Cleaning":
    st.header("🧹 Data Cleaning Operations")

    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset first")
    else:
        df = st.session_state["df"]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Missing Value Analysis")
            report = get_missing_value_report(df)
            st.dataframe(report, use_container_width=True)

        with col2:
            st.subheader("🛠️ Cleansing Workshop")

            clean_option = st.selectbox(
                "Null Value Strategy",
                ["None", "Fill with Mean", "Fill with Median", "Drop Null Rows", "Drop Columns"],
            )

            drop_duplicates = st.checkbox("Scrub duplicate rows")

            if st.button("✨ Apply Cleaning Configuration", type="primary"):
                original_shape = df.shape

                if clean_option == "Fill with Mean":
                    df = fill_missing_mean(df)
                elif clean_option == "Fill with Median":
                    df = fill_missing_median(df)
                elif clean_option == "Drop Null Rows":
                    df = df.dropna()
                elif clean_option == "Drop Columns":
                    df = df.dropna(axis=1)

                if drop_duplicates:
                    df = df.drop_duplicates()

                st.session_state["df"] = df
                st.success(f"✅ Operations successfully updated: {original_shape} ➔ {df.shape}")

            if st.button("↩️ Revert to Raw Original"):
                if st.session_state.original_df is not None:
                    st.session_state["df"] = st.session_state.original_df.copy()
                    st.success("Successfully restored raw data!")
                    st.rerun()

        st.subheader("📊 Current DataFrame Preview")
        st.dataframe(df.head(10), use_container_width=True)


# ==============================
# 4. FEATURE SELECTION
# ==============================
elif active_step_key == "Feature Selection":
    st.header("🎯 Feature Selection Setup")

    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset first")
    else:
        df = st.session_state["df"]
        columns = get_available_columns(df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Selection: Dependent Variable")
            target_column = st.selectbox(
                "Select Target Column (y)",
                columns,
                help="This is the specific variable targeted for predictive learning"
            )

        with col2:
            st.subheader("📊 Features: Independent Variables")
            available_features = [col for col in columns if col != target_column]

            select_all = st.checkbox("All Available Features")
            default_features = available_features if select_all else []

            feature_columns = st.multiselect(
                "Pick Features (X)",
                available_features,
                default=default_features,
            )

        if feature_columns:
            X, y = split_features_target(df, feature_columns, target_column)

            X = X.copy()
            encoded_cols = []
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = X[col].astype("category").cat.codes
                encoded_cols.append(col)

            problem_type = detect_problem_type(y)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Configured Target", target_column)
            c2.metric("📊 Selected Dimensions", len(feature_columns))
            c3.metric("🧠 ML Pipeline Mode", problem_type.capitalize())

            if encoded_cols:
                st.info(f"ℹ️ Automatic label encoding mapping applied to: {', '.join(encoded_cols)}")

            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["feature_columns"] = feature_columns
            st.session_state["target_column"] = target_column
            st.session_state["problem_type"] = problem_type

            st.success(f"✅ Variables locked. Proceed to training {problem_type} estimators.")

            with st.expander("🔍 Deep Preview features matrix (X)"):
                st.dataframe(X.head(), use_container_width=True)
            with st.expander("🔍 Deep Preview targets array (y)"):
                st.dataframe(y.head(), use_container_width=True)
        else:
            st.warning("Select at least one feature column to train.")


# ==============================
# 5. DATA VISUALIZATION
# ==============================
elif active_step_key == "Data Visualization":
    st.header("📈 Data Visualization Explorer")

    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset first")
    else:
        df = st.session_state["df"]

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        all_cols = df.columns.tolist()
        pie_cols = [col for col in all_cols if df[col].nunique(dropna=True) <= 15]

        plot_options = {
            "📊 Bar Graph": "Bar Graph",
            "🔵 Scatter Plot": "Scatter Plot",
            "🥧 Pie Chart": "Pie Chart",
            "📉 Histogram": "Histogram",
            "🔥 Heatmap Correlation": "Heatmap",
            "📦 Box Whisker Plot": "Box Plot",
            "🎻 Violin Distribution": "Violin Plot",
            "🧊 3D Dimensional Plot": "3D Plot",
        }

        col1, col2 = st.columns([1, 3])

        with col1:
            plot_label = st.radio("Chart Engine Selection", list(plot_options.keys()))
            plot_type = plot_options[plot_label]

        with col2:
            try:
                if plot_type == "Bar Graph":
                    if not numeric_cols:
                        st.warning("Requires numeric value mappings")
                    else:
                        x = st.selectbox("X Dimension Axis", all_cols, key="bar_x")
                        y_opts = [c for c in numeric_cols if c != x]
                        if y_opts:
                            y = st.selectbox("Y Quantitative Axis", y_opts, key="bar_y")
                            st.pyplot(plot_bar(df, x, y))

                elif plot_type == "Scatter Plot":
                    if len(numeric_cols) < 2:
                        st.warning("Requires at least 2 quantitative values.")
                    else:
                        c1, c2 = st.columns(2)
                        x = c1.selectbox("X Continuous Axis", numeric_cols, key="scatter_x")
                        y = c2.selectbox("Y Continuous Axis", [c for c in numeric_cols if c != x], key="scatter_y")
                        st.pyplot(plot_scatter(df, x, y))

                elif plot_type == "Pie Chart":
                    if not pie_cols:
                        st.warning("No categorical dimensions fit (<15 unique parameters)")
                    else:
                        col = st.selectbox("Categorical Target Label", pie_cols, key="pie_col")
                        st.pyplot(plot_pie(df, col))

                elif plot_type == "Histogram":
                    if not numeric_cols:
                        st.warning("Numeric bins selection required")
                    else:
                        col = st.selectbox("Variable", numeric_cols, key="hist_col")
                        st.pyplot(plot_histogram(df, col))

                elif plot_type == "Heatmap":
                    if len(numeric_cols) < 2:
                        st.warning("Requires 2+ numeric dimensions")
                    else:
                        st.pyplot(plot_heatmap(df))

                elif plot_type == "Box Plot":
                    if not numeric_cols:
                        st.warning("Numeric variable mapping required")
                    else:
                        c1, c2 = st.columns(2)
                        x = c1.selectbox("Label Categories", all_cols, key="box_x")
                        y_opts = [c for c in numeric_cols if c != x]
                        if y_opts:
                            y = c2.selectbox("Numeric Variables", y_opts, key="box_y")
                            st.pyplot(plot_box(df, x, y))

                elif plot_type == "Violin Plot":
                    if not numeric_cols:
                        st.warning("Continuous variables required")
                    else:
                        c1, c2 = st.columns(2)
                        x = c1.selectbox("Category Dimension", all_cols, key="violin_x")
                        y_opts = [c for c in numeric_cols if c != x]
                        if y_opts:
                            y = c2.selectbox("Continuous Distribution Values", y_opts, key="violin_y")
                            st.pyplot(plot_violin(df, x, y))

                elif plot_type == "3D Plot":
                    if len(numeric_cols) < 3:
                        st.warning("Requires 3 dimensional quantitative inputs")
                    else:
                        c1, c2, c3 = st.columns(3)
                        x = c1.selectbox("X Space Axis", numeric_cols, key="3d_x")
                        y = c2.selectbox("Y Space Axis", [c for c in numeric_cols if c != x], key="3d_y")
                        z = c3.selectbox("Z Space Axis", [c for c in numeric_cols if c not in [x, y]], key="3d_z")
                        st.pyplot(plot_3d(df, x, y, z))
            except Exception as e:
                st.error(f"Plot builder returned error: {str(e)}")


# ==============================
# 6. MODEL TRAINING
# ==============================
elif active_step_key == "Model Training":
    st.header("🚀 Model Training Suite")

    if "X" not in st.session_state:
        st.warning("⚠️ Complete Feature Selection setup first")
    else:
        X = st.session_state["X"]
        y = st.session_state["y"]
        problem_type = st.session_state["problem_type"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Pipeline Mode", problem_type.capitalize())
        col2.metric("Dataset Rows", len(X))
        col3.metric("Dataset Columns", X.shape[1])

        st.subheader("⚙️ Config Estimator Algorithms")

        if problem_type == "classification":
            models = get_classification_models()
        else:
            models = get_regression_models()

        selected_models = st.multiselect(
            "Select models to run",
            list(models.keys()),
            default=list(models.keys()),
        )

        if st.button("🚀 Train Selection", type="primary", use_container_width=True):
            if not selected_models:
                st.error("Select at least 1 machine learning model to start.")
            else:
                filtered_models = {k: models[k] for k in selected_models}
                X_train, X_test, y_train, y_test = split_dataset(X, y, problem_type)

                progress = st.progress(0)
                status = st.empty()

                with st.spinner("Training estimators..."):
                    status.info(f"🔄 Currently processing {len(filtered_models)} algorithms...")
                    trained_models = train_models(filtered_models, X_train, y_train)
                    progress.progress(100)

                status.empty()
                st.success(f"✅ Successfully trained {len(trained_models)} algorithms!")
                st.balloons()

                st.session_state["trained_models"] = trained_models
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test


# ==============================
# 7. MODEL EVALUATION
# ==============================
elif active_step_key == "Model Evaluation":
    st.header("📊 Performance Evaluations")

    if "trained_models" not in st.session_state:
        st.warning("⚠️ Train models first")
    else:
        trained_models = st.session_state["trained_models"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        problem_type = st.session_state["problem_type"]

        if problem_type == "classification":
            results = evaluate_classification_models(trained_models, X_test, y_test)
        else:
            results = evaluate_regression_models(trained_models, X_test, y_test)

        st.subheader("🏆 Comparative Performance Leaderboard")
        st.dataframe(results, use_container_width=True)

        try:
            if problem_type == "classification":
                best_metric = "Accuracy" if "Accuracy" in results.columns else results.columns[1]
                best_idx = results[best_metric].idxmax()
            else:
                if "R2 Score" in results.columns:
                    best_idx = results["R2 Score"].idxmax()
                elif "RMSE" in results.columns:
                    best_idx = results["RMSE"].idxmin()
                else:
                    best_idx = results.index[0]

            best_model = results.loc[best_idx]
            st.success(f"🥇 **Lead Estimator:** {best_model.iloc[0] if 'Model' in results.columns else best_idx}")
        except Exception:
            pass

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Performance CSV Metrics", csv, "model_results.csv", "text/csv")


# ==============================
# 8. PREDICTION
# ==============================
elif active_step_key == "Prediction":
    st.header("🔮 Forward Predictive Engine")

    if "trained_models" not in st.session_state:
        st.warning("⚠️ Train models first")
    else:
        trained_models = st.session_state["trained_models"]
        feature_columns = st.session_state["feature_columns"]
        X = st.session_state["X"]

        col1, col2 = st.columns([1, 1])

        with col1:
            selected_model_name = st.selectbox("🤖 Active Model Instance", list(trained_models.keys()))

        with col2:
            input_mode = st.radio("Forward Input Method", ["Manual Features Entry", "Import CSV File"], horizontal=True)

        selected_model = trained_models[selected_model_name]

        if input_mode == "Manual Features Entry":
            st.subheader("✍️ Specify Feature Variables")

            input_values = {}
            cols_per_row = 3
            for i in range(0, len(feature_columns), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(feature_columns[i:i+cols_per_row]):
                    with cols[j]:
                        default_val = float(X[col_name].mean()) if col_name in X.columns else 0.0
                        input_values[col_name] = st.number_input(
                            f"{col_name}",
                            value=default_val,
                            help=f"Mean bounds: Min: {X[col_name].min():.2f}, Max: {X[col_name].max():.2f}" if col_name in X.columns else ""
                        )

            if st.button("🔮 Perform Prediction Action", type="primary", use_container_width=True):
                try:
                    input_df = prepare_input_data(feature_columns, input_values)
                    prediction = make_prediction(selected_model, input_df)

                    st.markdown("---")
                    st.markdown("### 🎯 Inference Output")
                    result_val = prediction[0] if hasattr(prediction, '__len__') else prediction
                    st.success(f"### **{result_val}**")
                except Exception as e:
                    st.error(f"Inference run failed: {str(e)}")

        else:
            st.subheader("📁 Import Processing Batch CSV")
            st.info(f"Target format required: **{', '.join(feature_columns)}**")

            pred_file = st.file_uploader("Upload Batch File", type=["csv"], key="batch_pred_upload")

            if pred_file:
                pred_df = pd.read_csv(pred_file)
                missing = [c for c in feature_columns if c not in pred_df.columns]

                if missing:
                    st.error(f"Missing Columns required inside batches: {missing}")
                else:
                    pred_df_features = pred_df[feature_columns].copy()
                    for col in pred_df_features.select_dtypes(include=["object"]).columns:
                        pred_df_features[col] = pred_df_features[col].astype("category").cat.codes

                    if st.button("🔮 Process Batch Inferences", type="primary"):
                        predictions = selected_model.predict(pred_df_features)
                        pred_df["Predictions_Output"] = predictions

                        st.subheader("Prediction Outputs Matrix")
                        st.dataframe(pred_df, use_container_width=True)

                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Export Finished Batch CSV", csv, "batch_predictions.csv", "text/csv")


# ==============================
# RENDER NAVIGATION FOOTER CONTROLLER
# ==============================
render_bottom_navigation(st.session_state.step_index)

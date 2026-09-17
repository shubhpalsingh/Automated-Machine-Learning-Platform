ONLINE LINK - https://offline-automated-machine-learning-platform-ng6hojp8zzpjt6ocdk.streamlit.app

# 🤖  Automated Machine Learning Platform

An **offline Automated Machine Learning (AutoML) platform** built with **Python and Streamlit** that simplifies the machine learning workflow from dataset upload to model prediction.

The platform allows users to upload a CSV or Excel dataset, inspect and clean the data, visualize relationships, select features and target variables, train multiple machine learning models, compare their performance, and use the best-performing model for predictions — all through an interactive web interface.

The project is designed to make basic machine learning workflows more accessible while keeping the entire process **local and offline**.

---

## 🚀 Features

### 📂 Dataset Upload

* Upload **CSV and Excel datasets**
* Automatically inspect dataset structure
* Display number of rows and columns
* View column names and data types
* Preview dataset contents

### 🔍 Dataset Analysis

* Identify missing values
* Analyze numerical and categorical columns
* Generate basic dataset statistics
* Inspect data distributions

### 🧹 Data Cleaning

* Detect missing values
* Handle missing values using:

  * Mean
  * Median
* Prepare datasets for machine learning

### 🎯 Feature & Target Selection

* Select the target variable
* Select relevant input features
* Support numerical and categorical data where applicable

### 📊 Data Visualization

Generate visualizations to understand the dataset and relationships between variables.

Depending on the selected dataset, the platform can be used to explore:

* Feature distributions
* Correlations
* Relationships between variables
* Target distributions

### 🧠 Automated Model Training

The platform automatically trains multiple machine learning models based on the selected problem type.

#### Classification Models

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Support Vector Machine (SVM)
* Gradient Boosting
* AdaBoost

#### Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* K-Nearest Neighbors Regressor
* Support Vector Regression (SVR)

### ⚖️ Model Comparison

Models are trained and evaluated automatically so their performance can be compared.

The platform provides a model-performance comparison that helps users identify the most suitable model for their dataset.

### 📈 Model Evaluation

The evaluation process uses appropriate metrics depending on the machine learning task.

For classification:

* Accuracy
* Classification performance comparison

For regression:

* Regression performance metrics

### 🔮 Prediction

After evaluating the trained models, users can select a model and provide input values to generate predictions.

---

## 🏗️ Application Workflow

```text
            ┌─────────────────┐
            │  Upload Dataset │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │ Dataset Analysis│
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │  Data Cleaning  │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │ Feature/Target  │
            │    Selection    │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │  Visualization  │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │ Model Training  │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │ Model Evaluation│
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │ Model Selection │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │    Prediction   │
            └─────────────────┘
```

---

## 🛠️ Tech Stack

### Programming Language

* **Python**

### Framework

* **Streamlit**

### Machine Learning

* **Scikit-learn**

### Data Processing

* **Pandas**
* **NumPy**

### Visualization

* **Matplotlib**
* **Seaborn**

### Development Environment

* Jupyter Notebook
* VS Code

---

## 📁 Project Structure

```text
Offline-AutoML/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_dataset.csv
│
├── models/
│   └── ...
│
└── assets/
    └── screenshots/
```

> The exact structure may differ depending on the current implementation.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Offline-AutoML.git
```

### 2. Navigate to the project directory

```bash
cd Offline-AutoML
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💻 How to Use

### Step 1 — Upload Dataset

Upload a `.csv` or `.xlsx` dataset through the Streamlit interface.

### Step 2 — Inspect Dataset

Review:

* Dataset dimensions
* Column names
* Data types
* Missing values
* Statistical information

### Step 3 — Clean Data

Choose the appropriate missing-value handling method and prepare the dataset.

### Step 4 — Select Features

Select the input features and identify the target column.

### Step 5 — Visualize

Explore the dataset using the available visualization tools before training models.

### Step 6 — Train Models

Select the machine learning task:

```text
Classification
       or
Regression
```

The platform trains multiple supported models automatically.

### Step 7 — Compare Results

Compare model performance and identify the best-performing model.

### Step 8 — Make Predictions

Use the selected model to generate predictions using new input values.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Simplify the machine learning workflow for beginners.
* Automate repetitive model-training tasks.
* Allow users to compare multiple ML algorithms quickly.
* Provide an interactive interface for dataset analysis.
* Demonstrate practical implementation of machine learning concepts.
* Build an **offline alternative to basic AutoML workflows** without requiring an external ML platform or cloud service.

---

## 🔮 Future Improvements

Potential improvements include:

* Automatic categorical feature encoding
* Automatic feature scaling
* Advanced missing-value strategies
* Hyperparameter tuning
* Cross-validation
* Feature importance analysis
* ROC-AUC and confusion matrix visualization
* Precision, recall and F1-score
* Automated feature selection
* Ensemble model support
* Model export and import
* Prediction history
* Downloadable trained models
* Automated ML reports
* Docker support
* More advanced AutoML algorithms

---

## ⚠️ Limitations

This project is intended primarily as an educational and practical AutoML application.

The automatically selected model is **not guaranteed to be the best model for every dataset**. Model performance depends on factors such as:

* Dataset quality
* Feature selection
* Data preprocessing
* Class imbalance
* Dataset size
* Train/test split
* Hyperparameters

For production ML systems, additional validation, tuning, monitoring, and domain-specific analysis would be required.

---

## 📚 What I Learned

Through this project, I gained practical experience with:

* Machine learning workflows
* Data preprocessing
* Exploratory data analysis
* Feature and target selection
* Classification and regression
* Model training
* Model evaluation
* Comparing ML algorithms
* Building interactive ML applications
* Streamlit application development
* Structuring an end-to-end ML project

---

## 👨‍💻 Author

**Shubh**

Built as a practical machine learning project to explore **Automated Machine Learning, data preprocessing, model evaluation, and deployment-oriented ML workflows**.

---

## ⭐ Future Vision

The long-term goal is to evolve this project into a more complete AutoML system capable of automatically handling:

```text
Raw Dataset
     ↓
Data Profiling
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Model Selection
     ↓
Hyperparameter Optimization
     ↓
Cross Validation
     ↓
Model Evaluation
     ↓
Best Model
     ↓
Prediction
```

This would transform the current application from a **multi-model ML dashboard into a more comprehensive automated machine learning pipeline**.

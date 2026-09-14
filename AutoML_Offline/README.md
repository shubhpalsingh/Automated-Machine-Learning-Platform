#  Offline Automated Machine Learning (AutoML) Platform
##  Overview
This project is an **Offline Automated Machine Learning (AutoML) platform** designed to simplify the complete machine learning workflow for users without requiring deep technical expertise.
The system allows users to:
* Upload datasets
* Perform data analysis and cleaning
* Visualize data
* Train multiple machine learning models
* Compare performance
* Generate predictions
All operations are performed locally (offline) using an interactive interface.

##  Key Features
### 🔹 1. Dataset Upload
* Supports **CSV and Excel files**
* Validates dataset structure
* Displays preview of uploaded data
### 🔹 2. Dataset Analysis
* Dataset shape (rows & columns)
* Column data types
* Missing values report
* Statistical summary
### 🔹 3. Data Cleaning
* Missing value handling:
  * Mean Imputation
  * Median Imputation
* Improves data quality before training
 
### 🔹 4. Feature Selection
* Select **target variable**
* Select **input features**
* Automatically detects:
  * Classification
  * Regression
* Applies:
  * One-hot encoding for categorical variables
### 🔹 5. Data Visualization (EDA)
Supports dynamic, user-driven visualizations:
* Bar Graph
* Scatter Plot
* Pie Chart
* Histogram
* Heatmap
* Box Plot
* Violin Plot
* 3D Plot
Users can select columns for each visualization.
### 🔹 6. Model Training (Improved)
The platform supports multiple machine learning models:
####  Classification Models
* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)
* Gradient Boosting


####  Regression Models
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* KNN Regressor
* Support Vector Regressor (SVR)
* Gradient Boosting Regressor
###  Smart Preprocessing (NEW)
* **One-hot encoding** for categorical data
* **Stratified train-test split** (for classification)
* **Selective feature scaling**:
  * Applied only to:
    * Logistic Regression
    * KNN
    * SVM
  * Not applied to tree-based models
This ensures:
* Better accuracy
* No data leakage
* Optimized model performance
### 🔹 7. Model Evaluation
* Automatically compares all models
#### Metrics (Classification):
* Accuracy
* Precision
* Recall
* F1 Score
 
#### Metrics (Regression):
* MSE
* RMSE
* R² Score
Results are displayed in a **comparison table**.
### 🔹 8. Prediction
* Select any trained model
* Input feature values manually
* Get real-time prediction output
##  Project Structure
AutoML_Offline/
│

├── app.py                      # Main Streamlit application

│

├── src/

│   ├── data_loader.py         # Dataset loading & validation

│   ├── data_info.py           # Dataset analysis

│   ├── data_cleaning.py       # Data preprocessing

│   ├── feature_selection.py   # Feature handling

│   ├── visualization.py       # All plots

│   ├── model_training.py      # Model training + scaling

│   ├── model_evaluation.py    # Evaluation metrics

│   ├── prediction.py          # Prediction logic

└── requirements.txt           # Dependencies

 
##  Installation
### 1. Clone Repository
git clone <your-repo-link>
cd AutoML_Offline
### 2. Install Dependencies
pip install -r requirements.txt
### 3. Run Application
streamlit run app.py
##  Methodology
The system follows a structured pipeline:
1. Data Input
2. Data Understanding
3. Data Cleaning
4. Feature Engineering
5. Visualization
6. Model Training
7. Model Evaluation
8. Prediction
Each stage is modular, making the system scalable and maintainable.
##  Major Improvements (Latest Version)
* One-hot encoding instead of label encoding
* Selective feature scaling (model-specific)
* Stratified splitting for classification
* Improved model configurations
* Better accuracy and stability
* Modular and clean architecture
##  Known Limitations
* Performance depends on dataset quality
* Large datasets may slow visualization
* Logistic Regression may require higher iterations


##  Future Enhancements
* Auto Hyperparameter Tuning
* Feature Importance Visualization
* Advanced Models (XGBoost, LightGBM)
* Interactive dashboards (Plotly)
* Cloud deployment version
##  Technologies Used
* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
##  Conclusion
This project demonstrates how **Automated Machine Learning (AutoML)** can streamline the machine learning process by automating preprocessing, training, and evaluation.
It provides a practical and user-friendly system that brings machine learning capabilities to non-expert users.

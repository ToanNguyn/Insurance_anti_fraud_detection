## Insurance Claim Fraud Detection
This repository contains a Python implementation for predicting insurance claim fraud. The pipeline covers the complete machine learning workflow, from data preprocessing and feature engineering to model training, feature selection, and evaluation.

### Dataset
The dataset contains multiple fields related to insurance policies, policyholders, incidents, vehicles, and claim amounts.
* The target variable is `fraud_reported` (1 = Fraudulent, 0 = Not Fraudulent).
* Missing values and inconsistencies are handled in preprocessing.

### Project Workflow
1.  **Load Dataset**: Load CSV file using pandas.
2.  **Preprocessing**:
    * Handle missing values, negative values, and inconsistent entries.
    * Convert dates and split complex columns.
    * Generate new features based on domain knowledge.
3.  **Exploratory Data Analysis**: Boxplots, class distributions, outlier detection.
4.  **Feature Engineering**: One-Hot, Frequency, Ordinal, and Custom Ordinal encoding.
5.  **Model Training**:
    * Decision Tree and Random Forest classifiers.
    * Apply SMOTE-Tomek resampling for imbalanced data.
    * Feature selection with RFE.
    * Hyperparameter tuning with GridSearchCV.
6.  **Evaluation**: Weighted F1-score, recall, precision, and accuracy reported for both cross-validation and test set.

### Results
* **Decision Tree**: Test F1-score = 0.8482
* **Random Forest**: Test F1-score = 0.8201

The tuned Decision Tree model performed slightly better in predicting fraudulent claims.

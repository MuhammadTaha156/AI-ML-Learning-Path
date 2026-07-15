# 🚀 CreditWise Loan System

## Overview
The **CreditWise Loan System** is a machine learning pipeline designed to predict loan approval status based on applicant data. 

As highlighted in the project summary:
* Built an end-to-end supervised ML pipeline using **KNN**, **Logistic Regression**, and **Naive Bayes** to predict loan approval.
* Implemented Binary classification along with **EDA** (Exploratory Data Analysis), feature engineering, and model evaluation (Precision, Recall, F1-score).

## Dataset
The project utilizes the `loan_approval_data.csv` dataset, which contains 1,000 records of loan applicants. Key features include:
* **Financial Metrics**: Applicant Income, Coapplicant Income, Credit Score, Savings, Collateral Value, DTI (Debt-to-Income) Ratio.
* **Loan Details**: Loan Amount, Loan Term, Existing Loans, Loan Purpose.
* **Demographics & Employment**: Age, Gender, Marital Status, Dependents, Education Level, Employment Status, Employer Category, Property Area.
* **Target Variable**: `Loan_Approved` (Yes/No)

## Technologies & Libraries Used
* **Python 3**
* **Data Manipulation**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn (`LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`)
* **Preprocessing**: `SimpleImputer`, `StandardScaler`, `LabelEncoder`, `OneHotEncoder`

## Pipeline Architecture
1. **Exploratory Data Analysis (EDA)**: 
   * Visualizing class balance (Loan Approved vs. Not Approved).
   * Analyzing categorical distributions (e.g., Gender, Education Level).
   * Checking for outliers using boxplots (e.g., Income vs. Loan Approval, Credit Score vs. Loan Approval).
   * Correlation heatmaps to identify relationships between numerical features.
2. **Data Preprocessing**:
   * **Missing Value Handling**: Imputed numerical features using the mean and categorical features using the most frequent value.
   * **Encoding**: Applied `LabelEncoder` for binary/ordinal categorical variables and `OneHotEncoder` for nominal categorical variables.
   * **Scaling**: Standardized numerical features using `StandardScaler` to ensure uniform scale across variables.
3. **Model Training & Evaluation**:
   * Evaluated binary classification models utilizing algorithms like Logistic Regression.
   * Measured model success using robust evaluation metrics including Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Classification Reports.

## How to Run
1. Ensure Python 3.x is installed along with the required libraries (`pip install pandas numpy matplotlib seaborn scikit-learn`).
2. Clone the repository and navigate to the project directory.
3. Keep the dataset `2 loan_approval_data.csv` in the same directory as the notebook.
4. Run the Jupyter Notebook `main_2.ipynb` to view the EDA, preprocessing steps, and model evaluation results.


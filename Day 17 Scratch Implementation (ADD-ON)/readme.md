# Algorithm Implementations from Scratch

## Overview
This file contains pure Python and NumPy implementations of fundamental Machine Learning algorithms built entirely from scratch[cite: 10]. The purpose of this project is to demonstrate the mathematical foundations and core mechanics behind popular predictive models without relying heavily on high-level machine learning libraries for the underlying logic[cite: 10].

---

## Dependencies
To run the code in this notebook, ensure you have the following Python libraries installed:
* **NumPy:** Used heavily for matrix multiplications, vectorization, and mathematical operations[cite: 10].
* **Pandas:** Used for data frame manipulation and structuring input/output data[cite: 10].
* **Seaborn / Matplotlib:** Used for visualizing decision boundaries and data points (e.g., plotting KNN test vs. train points)[cite: 10].
* **Scikit-Learn:** Imported primarily for data splitting (`train_test_split`), grid search, and providing baseline models for comparison[cite: 10].

---

## Implemented Algorithms

### 1. Linear Regression (Gradient Descent)
A standard linear regression model that iteratively updates its weights and bias to minimize the error[cite: 10]. 
* **Hyperparameters:** Learning Rate (`lr`), Number of Iterations (`n_iter`)[cite: 10].
* **Mechanics:** Computes the predictions, calculates the gradient of the loss with respect to the weights ($dw$) and bias ($db$), and updates the parameters using the convergence theorem[cite: 10].

### 2. Linear Regression (Ordinary Least Squares - OLS)
An analytical approach to solving Linear Regression using the Normal Equation[cite: 10]. 
* **Mechanics:** Directly computes the optimal weights ($\theta$) without iteration by taking the dot product of the inverted matrices[cite: 10].
* **Formula Used:** $\theta = (X_b^T X_b)^{-1} X_b^T y$[cite: 10].

### 3. Logistic Regression
A binary classification algorithm that squashes linear predictions into probabilities between 0 and 1[cite: 10].
* **Hyperparameters:** Learning Rate (`lr`), Number of Iterations (`n_iter`)[cite: 10].
* **Mechanics:** Utilizes the Sigmoid activation function to calculate probabilities and updates weights using Gradient Descent[cite: 10]. It also features an adjustable decision threshold (default `0.5`) to map probabilities into discrete classes ($0$ or $1$)[cite: 10].
* **Sigmoid Formula:** $g(z) = \frac{1}{1 + e^{-z}}$[cite: 10].

### 4. K-Nearest Neighbors (KNN)
A "lazy learner" distance-based algorithm for binary classification[cite: 10].
* **Hyperparameters:** Number of Neighbors ($K$, default `k=3`)[cite: 10].
* **Mechanics:** Stores the training data during the `.fit()` stage[cite: 10]. During prediction, it calculates the distance between the test point and all training points, sorts them to find the nearest $K$ indices, and assigns the class based on a majority vote[cite: 10].
* **Distance Metric:** Euclidean Distance $\sqrt{\sum(x_1 - x_2)^2}$[cite: 10].

---

## Usage Example
Each class follows the standard `scikit-learn` API style with `.fit()` and `.predict()` methods[cite: 10].

```python
import numpy as np

# 1. Define Training Data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# 2. Initialize and Train Model
model = linearRegression(learning_Rate=0.01, n_iter=1000)
model.fit(X_train, y_train)

# 3. Make Predictions
predictions = model.predict(X_train)
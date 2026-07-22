# Machine Learning Part 5: Support Vector Machines (SVM)

## 1. Introduction to SVM (Binary Classification)
Support Vector Machine (SVM) is a powerful supervised learning algorithm used for both classification and regression. The primary objective of an SVM is to find a hyperplane in an N-dimensional space that distinctly classifies the data points.

### The Decision Boundary
In a 2D space, the decision boundary is a line:
$$y = mx + c \implies -mx + y - c = 0$$

In SVM notation, we use vectors where $W$ is the weight vector and $b$ is the bias. The equation for the decision boundary becomes:
$$W^T x + b = 0$$

Where:
* $W^T = [w_1, w_2]$
* $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$

---

## 2. Hard Margin vs. Soft Margin SVM

### Margins and Support Vectors
* **Support Vectors:** The critical data points closest to the decision boundary. These points define the margin.
* **Margin Hyperplanes:** Lines drawn parallel to the decision boundary through the support vectors.
  * They are equi-distant from the Decision Boundary (DB).
  * They are parallel to each other.

### Hard Margin SVM
Used when classes are perfectly and linearly separable. The goal is to maximize the margin distance between the two classes.

* **Equation for Margin Hyperplanes:**
  * Positive Class: $W^T x + b = 1$
  * Negative Class: $W^T x + b = -1$

* **Margin Calculation:** 
  The distance between the two margin hyperplanes is calculated as:
  $$\text{Margin} = \frac{2}{||W||}$$
  *(Where $||W||$ is the Euclidean norm of the W vector).*

* **Optimization Goal:** To maximize the margin ($\frac{2}{||W||}$), we must **minimize** $||W||$. Mathematically, this is expressed as:
  $$\min_{w,b} \frac{||W||^2}{2}$$
  **Subject to the constraint:** $y_i(W^T x_i + b) \ge 1$ (where $y_i \in \{+1, -1\}$)

### Soft Margin SVM
Used when data is not perfectly separable (i.e., there is overlapping data). Soft Margin allows for some misclassification to achieve a better overall fit and avoid overfitting.

* **Slack Variable ($\xi$):** Represents the degree of violation (distance of misclassified points from the margin hyperplane).
* **Cost/Loss Function:** We add a "slack penalty" to the minimization equation.
  $$\min_{w,b} \left( \frac{||W||^2}{2} + C \sum_{i=1}^{n} \xi_i \right)$$

* **The Role of the Hyperparameter $C$:**
  * If $C$ is **large:** Heavy penalty for misclassifications $\rightarrow$ Fewer misclassifications $\rightarrow$ Narrower margin $\rightarrow$ Risk of **Overfitting**.
  * If $C$ is **small:** Low penalty for misclassifications $\rightarrow$ Larger margin $\rightarrow$ More misclassifications allowed $\rightarrow$ Risk of **Underfitting**.

---

## 3. Support Vector Regression (SVR)
SVM can also be used for regression (predicting continuous values). Instead of a decision boundary, SVR uses an $\epsilon$-insensitive tube around the best fit line.

* **$\epsilon$ (Epsilon):** The margin of tolerance. Errors within this tube are ignored (cost is 0).
* **Objective:** Minimize the weights while keeping the error within the $\epsilon$-tube.
* **Cost Function:**
  $$\min_{w,b} \frac{||W||^2}{2}$$
  **Subject to the constraint:** $|y_i - \hat{y}_i| \le \epsilon \implies |y_i - (W^T x_i + b)| \le \epsilon$

* **Handling Outliers (Soft Margin SVR):**
  If data points fall outside the $\epsilon$-tube, we introduce slack variables ($\xi_i, \xi_i^*$) to add a penalty.
  $$\min_{w,b} \left( \frac{||W||^2}{2} + C \sum_{i=1}^{n} (\xi_i + \xi_i^*) \right)$$

---

## 4. Kernels in SVM (The Kernel Trick)
When data is not linearly separable in its original space, SVM uses "Kernel Tricks" to map the data into a higher-dimensional space where a linear hyperplane can be drawn.

### Common Types of Kernels:
1. **Linear Kernel:** 
   $$K(x, x') = x^T x'$$
2. **Polynomial Kernel:** 
   $$K(x, x') = (x^T x' + r)^d$$
3. **RBF (Radial Basis Function / Gaussian) Kernel:** *Highly popular for complex non-linear data.*
   $$K(x, x') = e^{-\gamma ||x - x'||^2}$$
4. **Sigmoid Kernel:** 
   $$K(x, x') = \tanh(\gamma x^T x' + r)$$


#  Ensemble Learning

## 1. Introduction to Ensemble Learning
An ML technique that aggregates two or more learners in order to produce better predictions. It aims to solve the issues of overfitting (low bias, high variance) and underfitting (high bias, low variance).

Ensemble Learning can be split into two main approaches:
1. **Bagging** (Parallel execution) $\rightarrow$ Reduces Variance (increases stability).
2. **Boosting** (Sequential execution) $\rightarrow$ Reduces Bias (increases accuracy).

Both methods can be used for Regression and Classification tasks.

---

## 2. Bagging (Bootstrap Aggregation)
Bagging uses modified replicates of a given training dataset to train multiple base learners, generally with the same training algorithm. 
* *Example:* **Decision Tree (DT) vs. Random Forest**
  * A single Decision Tree is prone to overfitting (low bias, high variance).
  * A Random Forest combines multiple DTs to produce a model with much lower variance (though slightly higher bias), resulting in a highly stable classifier/regressor.

### The 3 Steps of Bagging (Random Forest)
1. **Data Sampling:**
   * **Row Sampling (Bootstrap Sampling):** Randomly selecting rows from the original dataset *with replacement*. Some data points may be repeated, while others are left out completely.
   * **Feature Sampling (Feature Bagging):** Randomly selecting a subset of features (columns) for each tree to ensure the models are diverse and not highly correlated.
2. **Model Training:** Training individual base learners (e.g., Decision Trees) on these bootstrapped samples in parallel.
3. **Aggregation:** Combining the individual predictions into one final ensemble prediction.
   * **Classification:** Majority Voting.
   * **Regression:** Mean / Average.

---

## 3. Out of Bag (OOB) Evaluation
OOB is a built-in validation method in bagging that lets us estimate model performance without needing a separate validation set.

When creating a Bootstrap Sample with replacement:
* The probability of a specific row *not* getting selected in $n$ draws approaches $e^{-1} \approx 36.8\%$.
* This means roughly **$1/3$ of the training data is left out** of any given tree's training set.

These left-out samples are called **Out of Bag (OOB) samples**. Because the tree has never seen them during training, they act as an automatic, built-in validation dataset to calculate the **OOB accuracy/score** or **OOB error estimate**.

---

## 4. Boosting
Boosting is an ensemble method where models are trained **sequentially**, and each model tries to correct the mistakes of the previous ones.

### The Boosting Process:
1. **Sequential Training:** Train Model #1 on the initial dataset.
2. **Weight Adjustment:** Evaluate Model #1. Identify the incorrect predictions. Assign higher *weights* (importance) to these misclassified points.
3. Pass this new "weighted" data to Model #2 so it focuses specifically on the hard-to-predict cases.
4. Repeat this process for $n$ models. 

The final result combines all these "weak learners" into a single **"strong learner"** with high accuracy.
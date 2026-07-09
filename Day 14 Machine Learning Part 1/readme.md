# Machine Learning Part 1

## 1. Types of Machine Learning
* **Supervised Learning:** Uses *labelled data* to predict outcomes. 
  * Divided into: **Regression** and **Classification**.
* **Unsupervised Learning:** Uses *unlabelled data* to find hidden patterns. 
  * Commonly used for: **Clustering**.
* **Reinforcement Learning:** Goal-oriented learning based on **Rewards / Penalties**.

---

## 2. Supervised Learning
Supervised ML trains algorithms on labeled datasets, meaning each input has a correct output, to learn patterns and predict outcomes from new data.

### Data Structure Overview
| Input ($X$) - Features (Independent) | Output ($y$) - Label (Dependent) |
| :--- | :--- |
| Email Text, Sender, Link | Label (e.g., Spam / Not Spam) |

### Machine Learning Workflow
1. **Data** $\rightarrow$ Pre-process $\rightarrow$ Feature Engineering
2. **Select Model**
3. **Train / Test**
4. Evaluate for **Underfit / Overfit**

---

## 3. Types of Problems in Supervised Learning

### Regression
* **Output is a continuous numerical value.**
* *Examples:* House Price, Weather API, Stock Price.

### Classification
* **Output is a category or class.**
* *Examples:*
  * Spam / Non-spam $\rightarrow$ **Binary Classification**
  * Cat / Dog / Hen $\rightarrow$ **Multiclass Classification**
  * Benign / Malignant $\rightarrow$ Binary Classification

### Algorithms Used in Machine Learning
* **Linear Regression:** $\rightarrow$ Regression
* **Logistic Regression:** $\rightarrow$ Classification
* **K-Nearest Neighbors (KNN):** $\rightarrow$ Regression / Classification (R/C)
* **Decision Trees:** $\rightarrow$ Regression / Classification (R/C)
* **Naive Bayes:** $\rightarrow$ Classification (C)
* **Support Vector Machines (SVM):** $\rightarrow$ Regression / Classification (R/C)

---

## 4. Linear Regression
A fundamental algorithm for regression problems where the goal is to fit a line to the data points.

### Example: Study Hours vs. Scores
| $X$ (Independent / Features) - Study Hrs | $y$ (Dependent / Target) - Score |
| :---: | :---: |
| 1 | 50 |
| 2 | 55 |
| 3 | 65 |
| 4 | 75 |
| 5 | 90 |

* **Line Equation:** $y = mx + c$
  * $m$ = Slope (Coefficient / Weight)
  * $c$ = Intercept (Bias)
* **Objective:** Find the **Best Fit Line** by minimizing the error.
* **Error Formulation:** $\text{Error} = \text{Actual Value} - \text{Predicted Value}$ (Residual Error)

---

## 5. Mathematical Representation of Linear Regression

### The Hypothesis Function
The equation $y = c + mx$ is represented in ML terms as:
$$h_\theta(x) = \theta_0 + \theta_1 x_1$$
* $h_\theta(x)$ = Hypothesis function
* $\theta_0$ = Bias / Intercept
* $\theta_1$ = 1st Parameter / Coefficient
* $x_1$ = 1st Feature

For $n$ input features ($x_1, x_2, x_3, ..., x_n$):
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

### Cost Function
Measures how well the model's predictions match the actual data. In Linear Regression, we generally use **Mean Squared Error (MSE)**.

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
* $m$ = Number of training examples
* $h_\theta(x^{(i)})$ = Predicted value for $i^{th}$ point
* $y^{(i)}$ = Actual value for $i^{th}$ point

*Note on Cost Functions (CF): Other types include Squared error, Log of error, Absolute error.*

---

## 6. Gradient Descent
An optimization algorithm used to minimize the cost function $J(\theta)$ and find the global minimum (the best parameters $\theta_0, \theta_1$).

**Process:**
1. Initialize $\theta_0, \theta_1$ with random values.
2. Calculate cost $J(\theta)$.
3. Update $\theta_0, \theta_1$ iteratively until convergence.

### Manual Calculation Example
**Given Data Points:** $(1,2), (2,4), (3,6)$
Using simplified hypothesis: $h_\theta(x) = \theta_1 x$ (Assuming $\theta_0 = 0$)

* **Case 1:** Let $\theta_1 = 2$
  * $h_\theta(x) = 2x$ (Predictions: 2, 4, 6)
  * $J(\theta) = \frac{1}{2m} [0^2 + 0^2 + 0^2] = 0$ (Perfect fit!)

* **Case 2:** Let $\theta_1 = 1$
  * $h_\theta(x) = x$ (Predictions: 1, 2, 3)
  * $J(\theta) = \frac{1}{2(3)} [1^2 + 2^2 + 3^2] = \frac{14}{6} = 2.34$

* **Case 3:** Let $\theta_1 = 3$
  * $h_\theta(x) = 3x$ (Predictions: 3, 6, 9)
  * $J(\theta) = \frac{1}{2(3)} [(-1)^2 + (-2)^2 + (-3)^2] = \frac{14}{6} = 2.34$

When we plot $J(\theta)$ against $\theta_1$, it forms a parabolic curve (convex shape). Gradient Descent helps us step down this curve to find the **Global Minimum** (where $J(\theta)$ is lowest), updating $\theta$ at each step.


## 7. Gradient Descent (Continued)
Gradient Descent is an iterative optimization algorithm used to minimize the cost function by adjusting model parameters in the direction of the steepest descent of the function gradient.

* **Gradient:** $\frac{d(J(\theta))}{d\theta} \rightarrow$ Slope
* **Descent:** $\rightarrow$ Moving downwards

### Convergence Theorem
The mathematical formula to update the parameters iteratively:
$$\theta_k = \theta_k - \alpha \cdot \frac{d(J(\theta))}{d\theta_k}$$

Specifically for $\theta_1$:
$$\theta_1 = \theta_1 - \alpha \cdot \frac{d(J(\theta))}{d\theta_1}$$
$$\theta_1 = \theta_1 - \alpha(\text{Slope})$$

**The Role of Alpha ($\alpha$ - Learning Rate):**
* If $\alpha$ is **too small** $\rightarrow$ Slow training.
* If $\alpha$ is **too large** $\rightarrow$ Miss minima / Divergence.

### Steps for Gradient Descent:
1. Initialize parameters $\theta_0, \theta_1$.
2. Make predictions and calculate the cost function $J(\theta)$.
3. Use the convergence theorem to update parameters to find the Global Minima.

---

## 8. Evaluation Metrics
Evaluation metrics are how we measure how good a model is.

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}}$$

### R-Squared ($R^2$)
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
*(Where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value, and $\bar{y}$ is the mean of the actual values)*

### Adjusted R-Squared
$$\text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$
*(Where $n$ is the number of data points and $p$ is the number of predictors)*

# Machine Learning Part 2: Feature Engineering & Regularization

## 1. Feature Engineering
Transforming, creating, or selecting input variables to make our model perform better.

### Key Techniques:
* **Encoding Categorical Variables:** Converting text/categories into numbers. 
    * *Example (One Hot Encoding):* Smoker feature split into `smoking-yes` (1 or 0) and `smoking-no` (1 or 0).
* **Dummy Variable Trap:** When variables are highly correlated (multi-collinearity). 
    * *Example:* If you have Region A, B, and C, knowing A and B means you know C (`Region A + Region B + Region C = 1`).
    * *Issue:* Leads to inflated errors and unstable coefficients in the hypothesis function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3$.
    * *Solution:* Remove 1 redundant variable.
* **Create Useful Derived Features:** Extracting meaning from raw data.
    * *Example:* Converting a continuous `bmi` score into categories (e.g., `<18 underweight`, `18-25 normal`, `>25 overweight`).
* **Interaction Features:** Combining two or more features.
    * *Examples:* `age + smoker`, `bmi + smoker`.
* **Scaling Numeric Features:** Bringing all values to a similar scale using:
    * Normalization
    * Standardization
* **Feature Selection:** Picking the most important variables.

---

## 2. Underfitting vs. Overfitting
The goal of a machine learning model is to find the **Optimal Model** via the Bias-Variance Tradeoff (aiming for Low Bias & Low Variance).

### Underfitting (Model is too simple)
* **Characteristics:** High Bias, Low Variance.
* **Symptom:** Low training accuracy (e.g., 70%) and low test accuracy (e.g., 65%).
* **Visual:** Trying to fit a straight linear line through quadratic curve data ($y = x^2$).

### Overfitting (Model is too complex)
* **Characteristics:** Low Bias, High Variance.
* **Symptom:** High training accuracy (e.g., 95%) but low test accuracy (e.g., 65%).
* **Visual:** A squiggly line that perfectly hits every training data point but fails to generalize.
* **Cause:** Sensitivity to small data changes; captures errors from wrong assumptions.

---

## 3. Fixes for Underfitting & Overfitting

| Strategy | To Fix Underfitting (High Bias) | To Fix Overfitting (High Variance) |
| :--- | :--- | :--- |
| **Model Complexity** | **Increase** (Increase # of params, neurons, or layers) | **Decrease** (Decrease # of params, neurons, or layers) |
| **Regularization** | **Decrease** | **Increase** (Use L1 / L2 Regularization) |
| **Features** | **Increase** | **Decrease** |
| **Data** | Does **not** help | **Does help** (Add more training data) |

---

## 4. Regularization
Regularization improves model **generalization** by adding a penalty for complexity.
* **General Formula:** $\text{Cost} = \text{Error} + \lambda \cdot \text{Penalty}$
* **Lambda ($\lambda$):** Regularization strength (referred to as `alpha` in code). If $\lambda = 0$, the penalty disappears.

### 1. Lasso Regression (L1 Regularization)
Adds the absolute value of coefficients as a penalty term.
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$
* **Effect:** Can shrink parameters ($\theta_j$) of unimportant features exactly to $0$. 
* **Benefit:** Helps with feature selection and simplifies the model.

### 2. Ridge Regression (L2 Regularization)
Adds the squared magnitude of coefficients as a penalty term.
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$
* **Effect:** Prevents overfitting by minimizing the overall $J(\theta)$ and penalizing large slopes/weights, but rarely forces them exactly to zero.

### 3. ElasticNet
A combination of both L1 (Lasso) and L2 (Ridge) regularization techniques.

## 5. ElasticNet Math & Regularization Parameters
When implementing these algorithms in code (such as with Scikit-Learn), the regularization parameter $\lambda$ is often referred to as `alpha`.

### The ElasticNet Cost Function
ElasticNet combines both L1 (Lasso) and L2 (Ridge) penalties using a mixing parameter, denoted here as $x$ (often called the `l1_ratio` in code).

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + x \cdot \lambda \sum_{j=1}^{n} |\theta_j| + (1-x) \lambda \sum_{j=1}^{n} \theta_j^2$$

**Understanding the L1 Ratio ($x$):**
* **If $x = 1$:** The second penalty term $(1-1=0)$ cancels out. You are left with pure **Lasso (L1)**.
* **If $x = 0$:** The first penalty term $(0)$ cancels out. You are left with pure **Ridge (L2)**.

**The Role of Lambda ($\lambda$):**
* **If $\lambda = 0$:** The entire penalty portion is canceled out, and the cost function reverts entirely to standard **Linear Regression**.

---

## 6. Core Concepts Review List
A quick checklist of the foundational topics covered in this regression module:
* Linear Regression
* Lasso / Ridge / ElasticNet
* Evaluation Metrics
* Feature Engineering
* Underfit / Overfit
* **Bias-Variance Tradeoff** *(The central balancing act of model training)*


# Machine Learning Part 3: Logistic Regression

## 1. Introduction to Logistic Regression (Classification)
Unlike Linear Regression which predicts continuous values, Logistic Regression is used for **Classification** problems to predict discrete categories (e.g., Yes/No, 0/1).

### Why not use Linear Regression for Classification?
1. **Values on the Best Fit Line (BFL) are not probabilities:** They can go to infinity.
2. **Unbounded Output:** The hypothesis $h_\theta(x)$ can be $< 0$ or $> 1$, which doesn't make sense for probabilities.
3. **Sensitivity to Outliers:** Adding an outlier causes a *regressor shift*, drastically changing the prediction threshold and causing incorrect classifications.

**Solution $\rightarrow$ Squashing:**
We "squash" the straight line into an S-curve (convex curve). This gives us a **bounded range $[0, 1]$** which can be interpreted as a probability for prediction.

---

## 2. The Hypothesis & Sigmoid Function
To keep our predictions strictly between 0 and 1, we use the **Sigmoid Function** (also known as the Activation Function).

### The Sigmoid Function
$$g(z) = \frac{1}{1 + e^{-z}}$$
*(Where $e$ is Euler's number $\approx 2.718$)*

### The Logistic Regression Hypothesis
We substitute the linear equation $z = \theta_0 + \theta_1 x_1$ into the sigmoid function:
$$h_\theta(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1)}}$$

**For multiple features (Vectorized Form):**
If $\theta^T = [\theta_0, \theta_1, \theta_2]$ and $X$ is the feature vector, then $\theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2$.
The general hypothesis becomes:
$$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$

### Sigmoid Graph Properties
* **Range:** $(0, 1)$
* If $z = 0$, then $g(z) = 0.5$
* If $z \ge 0$, then $g(z) \ge 0.5$ (Predict Class 1)
* If $z \le 0$, then $g(z) \le 0.5$ (Predict Class 0)

---

## 3. Cost Function (Log Loss / Binary Cross Entropy)
We cannot use Mean Squared Error (MSE) for Logistic Regression because substituting the non-linear sigmoid function into MSE results in a non-convex graph (with many local minima). Instead, we use **Log Loss**.

### Piecewise Cost Function for a Single Point
$$
\text{Cost}(h_\theta(x), y) =
\begin{cases}
  -\log(h_\theta(x)) & \text{if } y = 1 \\
  -\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}
$$

**The Goal is to PENALIZE severe wrong predictions:**
* If actual $y=0$ but predicted $\hat{y}=0.9 \rightarrow$ Massive penalty.
* If actual $y=1$ but predicted $\hat{y}=0.1 \rightarrow$ Massive penalty.

### Combined Cost Function Equation
To make it mathematically easier for Gradient Descent, we combine the piecewise functions into a single equation known as **Binary Cross Entropy**:

$$J(\hat{y}, y) = -y \cdot \log(\hat{y}) - (1 - y) \cdot \log(1 - \hat{y})$$

*(Note: $\hat{y}$ is the predicted probability $h_\theta(x)$, and $y$ is the actual label which is strictly 0 or 1).*
# Machine Learning Part 6: Boosting Algorithms

## 1. Gradient Boosting (GBM)
Gradient Boosting is an ensemble learning method that builds a strong predictive model by combining multiple weak learners (usually Decision Trees) sequentially. Unlike Bagging (e.g., Random Forest) which builds trees in parallel and averages them, **Boosting builds trees one after another, where each new tree attempts to correct the errors (residuals) of the previous tree.**

### Gradient Boosting for Regression
Let's walk through the steps of Gradient Boosting for a regression task.

1.  **Initialize with a Simple Model ($F_0$):**
    The first step is to predict the average value ($\mu$) for all samples. This becomes our initial predicted value ($\hat{y}_0$).
    *   *Example:* If predicting GPA given SEM and BRANCH, and the target GPAs are $[8, 6, 8, 7, 9]$.
    *   $\hat{y}_0 = 	ext{mean} = rac{8+6+8+7+9}{5} = 7.6$

2.  **Calculate Residuals ($R_1$):**
    For every sample, calculate the error (residual) between the actual target value ($y$) and the predicted value ($\hat{y}$).
    *   $R_1 = y - \hat{y}_0$
    *   *Example Residuals:* $[0.4, -1.6, 0.4, -0.6, 1.4]$

3.  **Train a Decision Tree on the Residuals:**
    Train a new Decision Tree ($DT_1$) using the original features ($X$) as inputs, but use the **Residuals ($R_1$) as the target output** instead of the original $y$ values.

4.  **Calculate New Predictions ($\hat{y}_{new}$):**
    Update the predictions by adding the scaled output of the new Decision Tree to the previous predictions.
    *   $	ext{pred}_1 = 	ext{pred}_0 + (\eta \cdot DT_1)$
    *   *Note: $\eta$ (eta) is the Learning Rate.*

5.  **Repeat:**
    Calculate new residuals ($R_2 = y - 	ext{pred}_1$), train $DT_2$ on $(X, R_2)$, update predictions, and repeat for $m$ boosting steps (often 100, 200, or 300 trees).

**General Mathematical Formula:**
$$F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x)$$
*   $F_m(x)$: Final model after $m$ steps
*   $F_{m-1}(x)$: Model after $m-1$ steps
*   $\eta$: Learning Rate
*   $f_m(x)$: New weak learner (Decision Tree)

### Gradient Boosting for Classification
The process is very similar to regression, but it deals with probabilities and log odds instead of raw continuous values.
1.  **Simple Model ($F_0$):** Predict the log odds of the target class. Convert log odds to probability using a sigmoid function $\sigma(F_0) = \hat{y}$.
2.  **Calculate Residuals:** $R_1 = y - \hat{y}$ (where $y$ is 0 or 1, and $\hat{y}$ is the predicted probability).
3.  **Train DT:** Train a Decision Tree on the features and the calculated residuals.
4.  **Update Predictions:** Convert the tree outputs back to probabilities and update.

---

## 2. AdaBoost (Adaptive Boosting)
AdaBoost is another sequential ensemble method, primarily used for binary classification. Instead of fitting new models to residuals like Gradient Boosting, **AdaBoost adjusts the weights of the training samples.**

### The AdaBoost Process:
1.  **Initialize Weights:** Assign an equal weight ($w_i$) to all $N$ samples. Ensure $\sum w_i = 1$. ($w_i = rac{1}{N}$)
2.  **Train a Weak Learner (Stump):** A "stump" is a Decision Tree with only a single split (max depth = 1). Train the stump on the weighted data.
3.  **Calculate Weighted Error ($\epsilon$):** Calculate the total error of the stump.
    $$\epsilon = \sum_{	ext{incorrect}} w_i$$
4.  **Calculate Stump Importance ($ lpha$):** Determine how much "say" or importance this stump will have in the final prediction.
    $$ lpha = rac{1}{2} \ln \left( rac{1-\epsilon}{\epsilon} 
ight)$$
5.  **Update Sample Weights:**
    *   **Incorrectly classified samples:** Increase their weight to force the next model to focus on them. ($w_{i,new} = w_i \cdot e^{ lpha}$)
    *   **Correctly classified samples:** Decrease their weight. ($w_{i,new} = w_i \cdot e^{- lpha}$)
6.  **Normalize Weights:** Ensure the new weights sum to 1.
    $$w_{i,new} = rac{w_{i,new}}{\sum w_{i,new}}$$
7.  **Repeat:** Train the next stump ($WL_2$) on the newly weighted data. Repeat for $m$ boosting steps.
8.  **Final Output:** The final prediction is a weighted vote of all the individual stumps.
    $$F(x) =  lpha_1 f_1(x) +  lpha_2 f_2(x) + ... +  lpha_m f_m(x)$$

---

## 3. Other Modern Boosting Algorithms

### XGBoost (Extreme Gradient Boosting)
*   An optimized and highly scalable version of Gradient Boosting.
*   Very accurate, widely used in competitive machine learning and real-world tasks.
*   **Sparsity aware:** Handles sparse data (many zeros/missing values) exceptionally well.
*   **Regularization:** Built-in L1 and L2 regularization prevents overfitting.
*   *Con:* Can be slower to train on extremely large datasets compared to LightGBM.

### LightGBM (Light Gradient Boosting Machine)
*   **Histogram-based:** Speeds up training significantly by binning continuous features into discrete bins.
*   Uses less memory and provides swift computation.
*   Extremely fast on large datasets.
*   *Con:* Prone to overfitting on small datasets.

### CatBoost (Categorical Boosting) is a powerful, modern gradient boosting algorithm developed by Yandex.

*   **Optimized for Categorical Features:** Unlike XGBoost or LightGBM, CatBoost automatically handles and encodes categorical variables without requiring manual preprocessing (like One-Hot Encoding).
*   **Symmetric Trees:** It builds oblivious (symmetric) decision trees, which helps in preventing overfitting and speeding up predictions.
*   **Con:** It can be slightly slower to train than LightGBM when dealing with exceptionally large, numeric-only datasets.

## 1. Ensemble Learning Overview
Ensemble learning involves combining multiple base models to produce a single, optimal predictive model. 

Ensemble methods are broadly categorized into two types:
*   **Homogenous Ensemble Models:** Uses the *same* base algorithm repeatedly (e.g., combining 100 Decision Trees).
    *   *Examples:* Bagging (Random Forest), Boosting (Gradient Boosting, AdaBoost).
*   **Heterogenous Ensemble Models:** Combines *different* types of algorithms (e.g., combining Logistic Regression, SVM, and a Decision Tree).
    *   *Examples:* Voting, Stacking.

---

## 2. Voting
Voting is a simple heterogenous ensemble technique used for both regression and classification tasks. It operates on the principle of a "democracy" to aggregate the strengths of different models and combine weaker models into a stronger one.

### How it Works:
1.  Take an initial Dataset.
2.  Train multiple *different* models independently on the same dataset. 
    *   *Example:* Model 1 = Logistic Regression, Model 2 = Support Vector Classifier (SVC), Model 3 = Decision Tree (DT).
3.  Gather the individual predictions ($P_1, P_2, P_3$).
4.  **Aggregate the Predictions (VOTING):**
    *   **For Classification:** Use **Majority Vote** (Mode). The class predicted by the most models becomes the final prediction.
    *   **For Regression:** Use the **Mean** (Average) of the predicted values.

---

## 3. Stacking
Stacking (Stacked Generalization) is a more advanced heterogenous ensemble technique designed specifically for **preventing overfitting** through a process called "blending." 

Instead of simple majority voting, Stacking trains a new model (the "Meta-Model") to figure out how to best combine the predictions of the base models.

### Step 1: K-Fold CV (Creating Validation Predictions)
To prevent the meta-model from overfitting, we cannot train it on the same data used to train the base models. We use $K$-Fold Cross Validation (e.g., $CV=3$).
1. Divide the training data into $K$ folds. 
2. For each fold, train the base models (Model 1, Model 2, Model 3) on $K-1$ folds and make predictions on the remaining *validation fold*.
3. Repeat this process until every fold has been used as a validation set. 
4. The predictions made on the validation folds are combined to create a new dataset called the **Validation Meta-Set** ($val\_pred1, val\_pred2, val\_pred3$).

### Step 2: Training the Meta-Model
1. Train a new **Meta-Model** (often a simple model like Logistic Regression for classification or Linear Regression for regression) using the **Validation Meta-Set** as its input features.
2. The Meta-Model learns which base models are reliable and which ones are not, essentially learning how to weigh their predictions optimally.

### Step 3: Final Predictions
1. Retrain the base models (Model 1, Model 2, Model 3) on the *entire* original training dataset.
2. Pass the completely unseen **Test Data** through the base models to generate a **Test Meta-Set** ($test\_pred1, test\_pred2, test\_pred3$).
3. Pass this Test Meta-Set into the trained Meta-Model to generate the **Final Prediction**.
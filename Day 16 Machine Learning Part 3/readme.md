# Machine Learning Part 3

## 1. Naive Bayes
It is a probabilistic supervised ML algorithm mainly used for classification tasks.

### Bayes' Theorem Foundation
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

The core assumption (the "Naive" part) is that all features are **conditionally independent**.
$$P(A \cap B) = P(A) \cdot P(B)$$

### Naive Bayes Equation for Classification
When predicting an output $y$ based on input features $X = (x_1, x_2, x_3)$:

$$P\left(\frac{y}{x_1, x_2, x_3}\right) = \frac{P\left(\frac{x_1, x_2, x_3}{y}\right) \cdot P(y)}{P(x_1, x_2, x_3)}$$

Because features are independent, the numerator expands to:
$$P\left(\frac{x_1}{y}\right) \cdot P\left(\frac{x_2}{y}\right) \cdot P\left(\frac{x_3}{y}\right) \cdot P(y)$$

Since the denominator $P(x_1, x_2, x_3)$ remains constant for all classes, we can drop it and calculate the proportional probability ($\propto$).

### Example Calculation
Given features: $x_1 = \text{Hot}$, $x_2 = \text{Strong}$. Predict if Rain ($y$) is Yes or No.

* **Probability of Yes:**
  $P\left(\frac{\text{yes}}{\text{hot, strong}}\right) \propto P\left(\frac{\text{hot}}{\text{yes}}\right) \cdot P\left(\frac{\text{strong}}{\text{yes}}\right) \cdot P(\text{yes})$
  $\propto \frac{2}{3} \cdot \frac{1}{3} \cdot \frac{3}{5} = \frac{2}{15} \approx 0.134$

* **Probability of No:**
  $P\left(\frac{\text{no}}{\text{hot, strong}}\right) \propto P\left(\frac{\text{hot}}{\text{no}}\right) \cdot P\left(\frac{\text{strong}}{\text{no}}\right) \cdot P(\text{no})$
  $\propto \frac{1}{2} \cdot \frac{1}{2} \cdot \frac{2}{5} = \frac{1}{10} = 0.1$

Since $0.134 > 0.1$, the prediction is **Yes**.

### Types of Naive Bayes
* **Gaussian Naive Bayes:** Used for continuous values (e.g., medical diagnosis: temp, weight, height, scores).
* **Multinomial Naive Bayes:** Used for discrete values (e.g., text classification).
* **Bernoulli Naive Bayes:** Used for binary features (0/1).
* *Other types:* Complementary, Categorical.

---

## 2. k-Nearest Neighbors (kNN)
A "lazy learner" algorithm that doesn't strictly have a training phase; it memorizes the training data and performs computations only during prediction.

### Distance Metrics
To find the "nearest" neighbors, we calculate the distance between data points.

1. **Euclidean Distance** (most common): Straight-line distance.
   $$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
2. **Manhattan Distance**: Grid-based path distance (used in high-dimensional data).
   $$d = |x_1 - x_2| + |y_1 - y_2|$$

### kNN Algorithm Steps
1. **Choose K:** Select the number of nearest neighbors ($K$).
   * $K$ is a hyperparameter and is typically chosen to be an odd number to prevent ties in voting.
2. **Calculate Distance:** Find the distance of the new data point from *all* the data points in the training set.
3. **Sort and Select:** Sort the distances and find the nearest $K$ neighbors.
4. **Predict:**
   * **For Classification:** Take the majority vote of the $K$ neighbors.
   * **For Regression:** Take the mean/average of the $K$ neighbors.

### Limitations of kNN
* **Memory Intensive & Slow at Predictions:** It must calculate the distance to every single point, making it unsuitable for huge datasets.
* **Sensitive to Outliers.**
* **Sensitive to Feature Scaling:** Features with larger magnitudes will dominate the distance calculation; scaling is mandatory.
* **Curse of Dimensionality (High-Dimensional Data):** As dimensions increase, data sparsity increases. Distance metrics stop making sense, and kNN performs poorly.


---

## 3. Validation Data & Hyperparameter Tuning
When training models (like kNN where $K$ is a hyperparameter), we need a way to tune these parameters without "cheating" by looking at the test data.

### The Problem with 2-Way Split
Normally, we split our dataset into:
* **Train Data (80%)**
* **Test Data (20%)**

If we tune our hyperparameter $K$ by repeatedly checking the model's accuracy on the Test Data, the model inadvertently "learns" the test data. This leads to biased results.

### The Solution: 3-Way Split (Validation Set)
A validation dataset is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model's hyperparameters.

We take the 80% Training Data and split it again to carve out a **Validation Set**.
* **Workflow:**
    1. Train model on Training Set.
    2. Evaluate model on Validation Set.
    3. Tweak hyperparameters (e.g., try $K = 1, 3, 5, 7, 9, 11$).
    4. Repeat until you find the optimal hyperparameter.
    5. Pick the model that does best on the Validation Set.
    6. **Confirm results on the completely unseen Test Set.**

---

## 4. Cross Validation (K-Fold CV)
If we only use one static chunk of data for validation, our model might just get lucky (or unlucky) based on how that specific chunk was split. Cross Validation makes our evaluation much more **reliable**.

### How K-Fold CV Works
Instead of a single validation split, we divide the training data into $K$ equal parts (folds). *Note: This '$K$' is different from the '$K$' in kNN.*

Example: `K-Fold CV` where $K=5$ (CV=5) on 800 training values.
* Divide the 800 values into 5 folds.
* **Iteration 1:** Fold 1 is the Validation Data; Folds 2,3,4,5 are Training Data. $\rightarrow$ Get `accuracy 1`.
* **Iteration 2:** Fold 2 is the Validation Data; Folds 1,3,4,5 are Training Data. $\rightarrow$ Get `accuracy 2`.
* ... repeat for all 5 iterations.

Finally, we look at the evaluation metrics across all iterations to find the **best model** and calculate the **average** accuracy. This ensures our model performs well regardless of how the data is sliced.

---

## 5. Machine Learning Pipelines
A Pipeline allows us to sequentially apply a list of transformers to preprocess the data and, if desired, conclude the sequence with a final predictor for predictive modeling.

### Preventing Data Leakage
Data leakage occurs when information from outside the training dataset is used to create the model.
* For example, if you `fit` a scaler (calculate mean/std dev) on the *entire* dataset before splitting, the training process "knows" something about the validation/test data.

### How a Pipeline Helps
```text
Scale -> Train (e.g., kNN) -> Validate
```
By placing the scaler *inside* the pipeline, cross-validation will correctly apply the `fit_transform` only to the training folds of that specific iteration, and then `transform` the validation fold. It recalculates the mean and standard deviation for every split, perfectly sealing off the validation data and preventing leakage.
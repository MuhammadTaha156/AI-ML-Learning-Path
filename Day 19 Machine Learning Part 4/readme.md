# Machine Learning Part 4: Decision Trees

## 1. Introduction to Decision Trees
A Decision Tree predicts an outcome by asking a sequence of **if-then questions** about the input features. The goal is to continuously split the data until it reaches a "pure" state (converges to one class).

**Basic Logic Example:**
```python
if loan_amt >= 100000:
    print("Yes")
elif employed == "Yes":
    print("Yes")
else:
    print("No")
```

---

## 2. Core Algorithms
Several algorithms can be used to build Decision Trees:
1. **CART** (Classification and Regression Trees) - *Most common in Scikit-Learn*
2. **ID3 / C4.5**
3. **CHAID**

---

## 3. Feature Selection & Purity
To build a tree, the algorithm must decide **where** and **how** to split the data.
1. **Feature Selection for Split:** Evaluate which feature best separates the data.
2. **Degree of Purity of a Split:** Measured using Impurity Metrics (Entropy or Gini Impurity).
   * **Pure Node:** Homogeneous (contains only 1 class, e.g., all "Yes").
   * **Impure Node:** Heterogeneous (contains a mix of classes).

---

## 4. Impurity Metrics

### Entropy
Measures the impurity, disorder, or uncertainty in a dataset. 
* **Max Value:** $1$ (Completely impure/uncertain, e.g., 50/50 split)
* **Min Value:** $0$ (Certain/Sure, pure node)

$$E(S) = - \sum_{i=1}^{c} p_i \cdot \log_2 p_i$$
*(Where $p_i$ is the probability of class $i$)*

**Example Calculation (Binary Classification):**
* **Impure $[3	ext{y}, 3	ext{n}]$:** $E = - rac{1}{2}\log_2(rac{1}{2}) - rac{1}{2}\log_2(rac{1}{2}) = rac{1}{2} + rac{1}{2} = 1$
* **Pure $[0	ext{y}, 3	ext{n}]$:** $E = 0 + rac{3}{3}\log_2(1) = 0$

### Gini Impurity (Corrado Gini)
Measures the likelihood of a random element being misclassified in a dataset. 
* **Max Value:** $0.5$ (For binary classification)
* **Min Value:** $0$ (Pure node)

$$GI = 1 - \sum_{i=1}^{c} (p_i)^2$$

**Example Calculation:**
* **Impure $[3	ext{y}, 3	ext{n}]$:** $GI = 1 - ((rac{1}{2})^2 + (rac{1}{2})^2) = 1 - (rac{1}{4} + rac{1}{4}) = 0.5$
* **Pure $[0	ext{y}, 3	ext{n}]$:** $GI = 1 - (1^2 + 0) = 0$

### Which Metric to Use?
| Scenario | Entropy | Gini Impurity |
| :--- | :--- | :--- |
| **Training Speed** | Slightly slower (due to log calculations) | Faster computation (no log operations) |
| **Split Behaviour** | Produces more balanced node partitions | Creates splits quickly, favouring dominant classes |
| **Dataset Size** | Useful for structured datasets with balanced classes (*Small datasets*) | Efficient for large, high-dimensional datasets (*Large datasets*) |
| **Sensitivity to Distribution** | More sensitive to subtle probability differences | Less sensitive to small probability changes |
| **Common Usage** | Preferred when theoretical information gain matters | Often default in libraries like CART |

---

## 5. Information Gain & Gini Gain
**Information Gain (IG)** is the measure used to decide which feature to split on. The goal is to maximize the reduction in Entropy (or Gini Impurity). The algorithm uses a **Greedy** approach, always picking the split with the maximum IG value.

$$IG = 	ext{Entropy of Parent} - 	ext{Weighted Entropy of Children}$$
$$IG = H(S) - \sum_{v \in A} rac{|S_v|}{|S|} \cdot H(S_v)$$
*(Where $A$ is the feature we split on, and $v$ represents the possible values of $A$)*

**Gini Gain:**
$$	ext{Gini Gain} = 	ext{Gini Impurity of root node} - 	ext{Weighted } GI 	ext{ of children}$$

---

## 6. Pruning (Preventing Overfitting)
Decision Trees are highly prone to **overfitting** (where Training Accuracy >>>> Test Accuracy). **Pruning** is a technique used to remove parts of the tree that are unnecessary, in order to reduce overfitting, simplify the model, and improve generalization on unseen data.

### Types of Pruning:
1. **Pre-pruning:** Halting the tree's growth early (e.g., setting `max_depth` or `min_samples_leaf`).
2. **Post-pruning:** Growing the full tree first, then cutting back branches (e.g., Cost Complexity Pruning using `ccp_alpha`).

### Common Pruning Rules (Hyperparameters)
* **`max_depth`**: Stop splitting when the tree reaches a maximum specified depth.
* **`min_samples_split`**: A node must contain at least $N$ samples to be eligible for a split (e.g., $N=20$).
* **`min_samples_leaf`**: A leaf node must contain at least $N$ samples.
* **`max_leaf_nodes`**: Limit the total number of terminal leaf nodes.
* **`min_impurity_decrease`**: Only split a node if the impurity reduction exceeds a certain threshold.

# Machine Learning Part 4 (Continued): Decision Trees for Regression

## 1. Introduction to Regression Trees
While Classification Trees predict categorical outcomes, **Decision Trees for Regression** predict continuous numerical values. 
* The algorithm still follows the same basic steps:
  1. Decide which feature to split on.
  2. Split the data into two parts.
  3. Repeat recursively.
* **Decision at Leaf Node:** The final predicted value at any leaf node is simply the **mean (average)** of the data points that fall into that node.

---

## 2. Variance & Variance Reduction (VR)
In classification, we evaluate the "purity" of a split using Information Gain (derived from Entropy or Gini Impurity). For regression, we use **Variance Reduction** (derived from Variance, MSE, MAE, etc.).

### Variance Formula
$$Var = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$

### Variance Reduction (VR) Formula
To decide the best split, the algorithm calculates the variance of the parent node and subtracts the weighted variance of the child nodes. The split that yields the highest Variance Reduction is chosen.

$$VR = V_S - \left( \frac{|S_L|}{|S|} \cdot V_L + \frac{|S_R|}{|S|} \cdot V_R \right)$$
*(Where $V_S$ is the variance of the source/parent node, and $V_L, V_R$ are the variances of the left and right children).*

---

## 3. Mathematical Example: Calculating the Best Split
Given the following dataset:
| $x$ | $y$ |
| :---: | :---: |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 10 |

**Step 1: Calculate Variance of the Source Node ($V_S$)**
* Target values ($y$): $[2, 4, 6, 10]$
* Mean ($\bar{y}$) = $\frac{22}{4} = 5.5$
* $V_S = \frac{1}{4} [ (2-5.5)^2 + (4-5.5)^2 + (6-5.5)^2 + (10-5.5)^2 ]$
* $V_S = 8.75$

### Evaluating Split Candidates

**Case 1: Split at $x \le 2$**
* **Left Node (Yes):** $[2, 4]$
  * Mean $\bar{y} = 3$
  * Variance $V_L = \frac{1}{2}[(2-3)^2 + (4-3)^2] = \frac{1}{2}[1+1] = 1$
* **Right Node (No):** $[6, 10]$
  * Mean $\bar{y} = 8$
  * Variance $V_R = \frac{1}{2}[(6-8)^2 + (10-8)^2] = \frac{1}{2}[4+4] = 4$
* **Variance Reduction ($VR_1$):**
  * $VR_1 = 8.75 - \left( \frac{2}{4} \cdot 1 + \frac{2}{4} \cdot 4 \right) = 8.75 - (0.5 + 2)$
  * $VR_1 = 6.25$

**Case 2: Split at $x \le 3$**
* **Left Node (Yes):** $[2, 4, 6]$
  * Mean $\bar{y} = 4$
  * Variance $V_L = \frac{1}{3}[(2-4)^2 + (4-4)^2 + (6-4)^2] = \frac{1}{3}[4+0+4] = \frac{8}{3}$
* **Right Node (No):** $[10]$
  * Mean $\bar{y} = 10$
  * Variance $V_R = 0$ (since it's a single value)
* **Variance Reduction ($VR_2$):**
  * $VR_2 = 8.75 - \left( \frac{3}{4} \cdot \frac{8}{3} + \frac{1}{4} \cdot 0 \right) = 8.75 - 2$
  * $VR_2 = 6.75$

### Conclusion
Because **Case 2** provides a higher Variance Reduction ($6.75 > 6.25$), the greedy algorithm will choose to split the data at $x \le 3$.
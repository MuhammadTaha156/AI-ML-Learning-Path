# Unsupervised Machine Learning (Part 2): Dimensionality Reduction & PCA

## 1. Dimensionality Reduction
In Machine Learning, a "dimension" refers to a feature ($f_1, f_2, ..., f_n$). Dimensionality Reduction (DR) is the process of reducing the number of random variables under consideration.

### The Curse of Dimensionality
Why do we need to reduce dimensions?
1.  **Data Sparsity:** As dimensions increase, the volume of the space increases exponentially, making the available data sparse. This makes it hard for algorithms to find patterns.
2.  **Distance Becomes Meaningless:** In high-dimensional space, the distance between any two points tends to converge, which severely hurts algorithms that rely on distance metrics (like Clustering and KNN).
3.  **Expensive Computation:** More features require more memory and processing power.
4.  **Model Performance Drops:** Adding more features initially helps, but after a certain point (e.g., going from 50 to 100 features), model performance degrades.

### Ways to Reduce Dimensions
1.  **Feature Selection:** Manually removing redundant features or keeping only the important ones.
2.  **Feature Transformation (Dimensionality Reduction):** Transforming the original high-dimensional features ($f_1, f_2, f_3$) into a smaller set of entirely new, meaningful features called Principal Components (PCs). *Example: PCA*.
    *   *Goal:* Compress/extract data into a smaller, meaningful space that behaves the same way as the original data.

---

## 2. Principal Component Analysis (PCA) Intuition
PCA takes highly correlated data and reduces its dimensions while preserving as much variance (information/spread) as possible.
*   *Example:* You have two highly correlated features: Avg Watch Time ($f_1$) and % of Completion Rate ($f_2$). Since they are highly correlated (high Pearson Correlation Coefficient, $r$), you can compress this 2D data into a 1D line.

### How PCA conceptually works:
1.  Look at the **spread of the data** (variance = information).
2.  Calculate the mean and center the data around the **origin** (0,0).
3.  Find the **Best Fit Line** that passes through the origin.
    *   This line becomes **Principal Component 1 (PC1)**. It captures the maximum spread (variance/information) of the data.
    *   A second line perpendicular to PC1 becomes **PC2**.
4.  If PC1 captures 80% of the variance and PC2 captures only 20%, you can project the data onto PC1 and drop PC2, effectively shifting the axis and reducing the data from 2D to 1D.

---

## 3. PCA Steps (Mathematical Breakdown)

**Step 1: Standardize the data** (Z-score normalization)
$$x_{std} = \frac{x - \mu}{\sigma}$$
*(This ensures $\mu = 0$ and $\sigma = 1$)*

**Step 2: Compute the Covariance Matrix**
This matrix shows how much features $f_1$ and $f_2$ vary with each other.
$$cov(X,Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$
$$A = \begin{bmatrix} var(X) & cov(X,Y) \\ cov(Y,X) & var(Y) \end{bmatrix}$$

**Step 3: Compute the Best Fit Line (Eigenvectors & Eigenvalues)**
The goal is to find a line that either:
*   Minimizes the distance ($d_1$) from the samples to the line.
*   OR Maximizes the projected distance ($d_2$) along the line for all samples. 
    $$\text{Maximize } \sum_{i=1}^{n} d_{i}^2$$

To find this line, we solve for the **Eigenvector** and **Eigenvalue** using the covariance matrix ($A$):
$$A \cdot \vec{v} = \lambda \cdot \vec{v}$$
*   **$A$:** Covariance Matrix
*   **$\vec{v}$ (Eigenvector):** A unit vector ($||\vec{v}|| = 1$) that dictates the *direction* of the Principal Component (e.g., PC1).
*   **$\lambda$ (Eigenvalue):** A scalar that tells us *how much variance* that specific line (eigenvector) captures.

**Step 4: Select Principal Components and Project**
*   Each line (PC1, PC2) has an associated Eigenvector and Eigenvalue.
*   If $var(PC1) \gg var(PC2)$, we keep PC1 as our new feature and drop PC2.
*   Finally, we project the original data points onto the new PC1 axis using a linear combination of $f_1$ and $f_2$.


## 1. Introduction to Anomaly Detection
Anomaly Detection (or Outlier Detection) is the process of identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

*   **Outlier:** An extreme data point.
*   **Anomaly:** A suspicious, unusual, or unexpected data point. Often referred to as *noise*.

### Key Use Cases:
1.  **Finance:** Credit card fraud detection (e.g., detecting a sudden \$90k transaction when typical transactions are \$10k-\$50k).
2.  **Manufacturing:** Defect detection.
3.  **Medical Diagnosis:** Identifying abnormal scans or vitals.
4.  **Cybersecurity:** Network intrusion detection.
5.  **Bot Detection:** Identifying non-human traffic.

---

## 2. Algorithms for Anomaly Detection
Since anomalies do not fit neatly into standard clusters, specific unsupervised algorithms are used to isolate them.

### A. DBSCAN (Density-Based)
DBSCAN can be repurposed for anomaly detection. We previously learned it classifies points into Core, Border, and Noise.
*   **Mechanism:** Any data point that has fewer than `min_samples` within its $\epsilon$-neighborhood AND is completely disconnected from any core point is considered an anomaly.
*   **Output Label:** DBSCAN labels these noise/anomaly points as **`-1`**.

### B. Isolation Forest
An ensemble algorithm built specifically for anomaly detection. It works on the premise that anomalies are "few and different," making them easier to isolate than normal points.

*   **Mechanism:** 
    1.  It builds multiple random Decision Trees (**Isolation Trees**).
    2.  It randomly splits features until every single point is isolated into its own leaf node.
    3.  Because anomalies are far away from dense, normal data (they live in sparse space), they require **fewer splits** to isolate.
*   **Path Length ($PL$):** The number of splits (edges) from the root node to the leaf node holding the sample.
    *   $PL \downarrow \implies$ Outlier $\uparrow$ (Short path = likely anomaly)
    *   $PL \uparrow \implies$ Normal $\uparrow$ (Long path = deep inside a dense cluster)
*   **Anomaly Score ($AS$):** The average path length across all the trees is used to calculate the score.
    $$AS(x) = 2^{ - \frac{E(h(x))}{c(n)} }$$
    *(Where $E(h(x))$ is the average path length for sample $x$, and $c(n)$ is the average path length of all points in the dataset $n$.)*
    *   If $E(h(x)) \ll c(n)$, then **$AS(x) \approx 1$ (Strong Anomaly)**.
    *   If $AS(x) \approx 0.5$ (Borderline).
    *   If $AS(x) \approx 0$ (Normal).

### C. LOF (Local Outlier Factor)
LOF measures the local deviation of density of a given sample with respect to its neighbors. It compares the density of a point to the density of its neighbors.

*   **Mechanism:**
    1.  Choose $K$ (number of neighbors, e.g., $K=3$).
    2.  Find the distance to the $K^{th}$ nearest neighbor ($K$-distance).
    3.  Calculate the **point density** and the **neighbor density**.
*   **Formula:**
    $$LOF = \frac{\text{neighbor density}}{\text{point density}}$$
*   **Interpretation:**
    *   If a point is inside a dense cluster, its density is roughly equal to its neighbors' density, so $LOF \approx 1$.
    *   If a point is an outlier, its density is much lower than its neighbors' density.
    *   **$LOF \gg 1 \implies$ Strong possibility of being a Local Outlier.**
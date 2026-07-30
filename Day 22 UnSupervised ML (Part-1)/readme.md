# Unsupervised Machine Learning (Part 1): Clustering

## 1. Introduction to Unsupervised ML
Unsupervised Machine Learning is used when we have **unlabelled data**. The goal is to find meaningful patterns, groupings, or structures within the dataset without a predefined target variable.

**Example: Bank Transactions**
* Features: $f_1$ (Number of online transactions), $f_2$ (Number of ATM withdrawals)
* By plotting these features, we might discover patterns:
  * Users with high online transactions (target for online adoption campaigns).
  * Users with high ATM withdrawals.
  * Anomalies: A user behaving completely differently from the rest (could indicate fraud).

### Key Applications:
1.  **Clustering:** Grouping similar data points together.
2.  **Anomaly Detection:** Identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

---

## 2. What is Clustering?
Clustering is an unsupervised ML technique that automatically groups similar, unlabeled data points into clusters based on defined similarity metrics.

### Types of Clustering Algorithms:
1.  **Partitioning-based:** Divides data into non-overlapping groups (e.g., **K-Means Clustering**).
2.  **Hierarchical Clustering:** Builds a tree of clusters.
3.  **Density-based:** Groups points that are closely packed together (e.g., **DBSCAN**).

---

## 3. K-Means Clustering Algorithm

K-Means is a partitioning-based algorithm where '$K$' represents the number of clusters.

### Algorithm Steps:
1.  **Choose $K$:** Determine the number of clusters you want to create.
2.  **Initialize Centroids:** Randomly initialize $K$ centroids (the central point/representative of a cluster).
3.  **Assign Points:** Calculate the distance (usually Euclidean or Manhattan) from each data point to all centroids. Assign the point to the nearest centroid.
    *   *Distance Formula:* $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$
4.  **Update Centroids:** Recalculate the position of each centroid. The new centroid becomes the mean position of all the points currently assigned to that cluster.
5.  **Repeat:** Repeat steps 3 and 4 until the centroids stop moving (the algorithm **converges**).

---

## 4. How to Choose K? (Evaluation Metrics)

Since Unsupervised learning doesn't have labels to calculate accuracy, we use specific metrics to evaluate how "good" a cluster is. A good cluster has high cohesion (points are close together) and high separation (clusters are far apart from each other).

### 1. The Elbow Method & WCSS
**WCSS (Within-Cluster Sum of Squares) / Inertia:** Measures the sum of squared distances between each data point and its nearest centroid.
$$WCSS = \sum_{k=1}^{K} \sum_{i=1}^{n} (x_i - \mu_k)^2$$

*   **The Elbow Method:** We run K-Means for various values of $K$ (e.g., $K=1, 2, 3, 4, 5...$) and calculate the WCSS for each. We then plot $K$ vs. WCSS.
*   As $K$ increases, WCSS decreases. The optimal $K$ is at the "elbow point"‚Äîthe point where the rate of decrease sharply shifts, indicating that adding more clusters doesn't significantly improve the model.

### 2. Silhouette Score
Evaluates the quality of clustering for a single sample $i$. 
*   **Range:** $[-1, 1]$ (Closer to 1 means better clustering).

**Calculations for a single sample $(i)$:**
1.  **$a(i)$ [Cohesion]:** Intra-cluster distance. The mean of the distances between point $i$ and all other points in the *same* cluster ($C_I$).
2.  **$b(i)$ [Separation]:** Nearest cluster distance. The minimum mean distance between point $i$ and all points in the *nearest neighboring* cluster ($C_J$).

**The Silhouette Score Formula:**
$$S(i) =  rac{b(i) - a(i)}{\max\{a(i), b(i)\}} \quad 	ext{if } |C_I| > 1$$
*(Note: Overall Silhouette Score (SS) is the mean of $S(i)$ for all $N$ samples: $SS =  rac{1}{N} \sum_{i=1}^{N} S(i)$)*

**Interpreting $S(i)$:**
*   **Case 1: $a(i) > b(i)$** (Intra-cluster distance is greater than nearest cluster distance).
    *   $S(i)$ is negative ($< 0$). This means **Bad Clustering** (the point is closer to another cluster than its own).
*   **Case 2: $a(i) < b(i)$**
    *   $S(i)$ is positive ($> 0$). This means **Good Clustering**.
*   **Case 3: $a(i)  pprox b(i)$**
    *   $S(i)  pprox 0$. The point is right at the decision boundary between two clusters.

## 1. The Random Initialization Trap (K-Means++)
In standard K-Means, centroids are initialized randomly. This can lead to a "Random Initialization Trap" (RIT) where the final clusters are suboptimal because the initial centroids were placed poorly (e.g., two centroids initialized very close to each other in the same natural cluster).

**Solution: K-Means++**
K-Means++ is an improved initialization algorithm that ensures the initial centroids are placed as **far apart** from each other as possible. This greatly improves the chances of finding the optimal, ideal clustering and speeds up convergence.

---

## 2. Hierarchical Clustering
Unlike K-Means, Hierarchical Clustering does not require us to specify the number of clusters ($K$) beforehand, nor does it use centroids. 

There are two main types:
1.  **Agglomerative (Bottom-up):** Starts with each data point as its own individual cluster and merges them together step-by-step.
2.  **Divisive (Top-down):** Starts with the entire dataset as a single massive cluster and recursively splits it into sub-clusters by finding the maximum gap.

### The Agglomerative Algorithm Steps:
1.  Treat each sample as an individual cluster.
2.  Calculate a **Distance Matrix** (using Euclidean distance) to find the distance between every cluster $(i, j)$.
3.  Find the nearest cluster and **merge** the two closest clusters into a single new cluster.
4.  **Repeat** steps 2 and 3 until all points are merged into one large cluster.

### The Dendrogram
A tree diagram used to visualize the history of cluster merging in Hierarchical Clustering. It helps us decide the optimal number of clusters ($K$) *after* the algorithm has run.

*   **How to find optimal $K$:** Look for the **longest vertical line** in the Dendrogram that does not have any horizontal lines crossing it. Draw a horizontal line through it. The number of vertical lines this new horizontal line intersects is suggestive of the optimal $K$ value.

### K-Means vs. Hierarchical Clustering
| Feature | K-Means | Hierarchical |
| :--- | :--- | :--- |
| **K Value** | Must be decided beforehand. | Decided later using a Dendrogram. |
| **Dataset Size** | Best for large datasets. | Best for small-to-medium datasets. |
| **Time Complexity** | $O(n \cdot K)$ (Faster) | $O(n^2)$ or $O(n^3)$ (Slower) |

---

## 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN groups together points that are closely packed together (high density) and marks points that lie alone in low-density regions as outliers/noise.

### Key Hyperparameters:
1.  **$\epsilon$ (Eps):** The radius of the neighborhood around a data point.
2.  **$min\_samples$:** The minimum number of points required within the $\epsilon$-neighborhood to form a dense region (density threshold).

### Categories of Points:
1.  **Core Point:** A point that has at least $min\_samples$ points (including itself) within its $\epsilon$-neighborhood.
2.  **Border Point:** A point that has fewer than $min\_samples$ in its neighborhood, but it falls within the $\epsilon$-neighborhood of a Core Point.
3.  **Noise / Outlier:** A point that is neither a core point nor a border point.

### The DBSCAN Algorithm Steps:
1.  Pick one unvisited sample.
2.  Check if it is a **Core Point**.
    *   If yes: Build a new cluster around it. Repeat the process to find all reachable core and border points within the $\epsilon$-neighborhoods to expand the cluster.
    *   If no: Mark it as unvisited/noise (for now) and move on.
3.  Find the next unvisited sample and repeat until all points have been evaluated.
# SmartCart Clustering System

An unsupervised machine learning project that segments SmartCart's e-commerce customers into meaningful groups based on demographics, spending, and engagement behavior — enabling targeted marketing instead of one-size-fits-all campaigns.

## Problem Statement

SmartCart is a growing e-commerce platform with customers spread across multiple countries. It had collected a rich dataset — **2,240 customer records with 22 attributes** covering demographics, purchase behavior, website activity, and campaign response — but was still treating every customer the same way, using **generic marketing and engagement strategies** for the entire customer base.

This "one-size-fits-all" approach led to:
- Inefficient marketing spend (discounts sent to customers who don't need them)
- Missed opportunities to nurture and retain high-value customers
- Delayed identification of price-sensitive or churn-prone users

**Goal:** Build a data-driven customer segmentation system using unsupervised ML that discovers hidden patterns in customer behavior, so SmartCart can design personalized marketing and retention strategies for each segment instead of guessing.

## Dataset

`smartcart_customers.csv` — 2,240 rows × 22 columns.

| Category | Features |
|---|---|
| **Demographics** | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer` |
| **Spending (amount)** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| **Purchase frequency** | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| **Feedback / target** | `Recency` (days since last purchase), `Complain`, `Response` |

## Approach / Pipeline

The full workflow lives in `main.ipynb`:

1. **Data Cleaning**
   - Filled missing `Income` values with the median.
2. **Feature Engineering**
   - `Age` = 2026 − `Year_Birth`
   - `Customer_Tenure_Days` = days since enrollment relative to the most recent signup
   - `Total_Spending` = sum of all `Mnt*` product spend columns
   - `Total_Children` = `Kidhome` + `Teenhome`
   - Simplified `Education` into `Undergraduate` / `Graduate` / `Postgraduate`
   - Simplified `Marital_Status` into `Living_With` = `Partner` / `Alone`
   - Dropped now-redundant raw columns (`ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, individual `Mnt*` columns)
3. **Outlier Removal**
   - Removed customers with `Age` ≥ 90 and `Income` ≥ 600,000, using pairplots to spot them.
4. **Encoding & Scaling**
   - One-hot encoded `Education` and `Living_With`
   - Standardized all features with `StandardScaler`
5. **Dimensionality Reduction**
   - Applied **PCA** (3 components) to compress the feature space and visualize customers in 2D/3D.
6. **Choosing the Number of Clusters (K)**
   - **Elbow Method** (WCSS) + `KneeLocator`
   - **Silhouette Score** across K = 2–10
   - Both methods pointed toward **K = 4** as the optimal number of clusters.
7. **Clustering**
   - Compared **K-Means** and **Agglomerative Clustering** (Ward linkage), both with `n_clusters=4`, on the PCA-reduced data.
8. **Cluster Profiling**
   - Labeled each customer with its cluster, then analyzed cluster sizes, income-vs-spending scatter plots, and per-cluster feature averages to build a business-readable profile for each segment.

## Results — The 4 Customer Segments

| Cluster | Profile | Traits | Marketing Strategy |
|---|---|---|---|
| **Cluster 0 — 🔴 Red** | Family Shoppers | Low/moderate income, low/moderate spending, more children, lower response, more web/catalog/store activity | Discounts & coupons |
| **Cluster 1 — 🔵 Blue** | Loyal, Price-Sensitive | High income, high spending, fewer children, slightly older, average response, prefers store & catalog over web | Loyalty programs, premium service |
| **Cluster 2 — 🟡 Yellow** | Sales & Discount Hunters | Low income, low spend, more children, lives alone, heavy web visits | Digital boosters, deep discounts/coupons |
| **Cluster 3 — 🟢 Green** | High-Value Singles | Moderate–high income, high spend, fewer children, slightly older, best campaign response, lives alone | Premium service — **best ROI segment** |

Overall pattern: **income and spending rise together**, and **web-visit frequency tends to fall as income/spending rises** (higher-value customers shop more through catalog/store rather than browsing the site repeatedly).

## Tech Stack

- **Python**, **Pandas**, **NumPy** — data handling
- **Matplotlib**, **Seaborn** — visualization (pairplots, heatmaps, cluster scatter plots)
- **scikit-learn** — `StandardScaler`, `OneHotEncoder`, `PCA`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`
- **kneed** — automatic elbow-point detection for the Elbow Method

## Project Structure

```
├── main.ipynb                 # Full analysis: cleaning → EDA → clustering → cluster profiling
├── smartcart_customers.csv    # Raw dataset (2,240 customers × 22 features)
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
jupyter notebook main.ipynb
```

## Key Takeaway

Instead of one generic campaign for all 2,240 customers, SmartCart can now run **four targeted strategies**: discount-driven offers for price-sensitive family shoppers, loyalty perks for high-income regulars, aggressive digital discounts for budget-conscious singles, and premium/VIP treatment for the highest-ROI segment — high-spending singles who already respond best to campaigns.
# Unsupervised-Learning

## Definition

Machine learning that deals with the exploration and extraction of patterns from unlabeled data.

## Clustering

- **Definition**
- Grouping similar data points together 
- Create  clusters where data points within the same cluster share more similarities with each other than with those in other clusters.
- Uncover the inherent structure of the data.

- **Different clustering approaches**

1. Centroid-Based Clustering
2. Hierarchical Clustering
3. Density-Based Clustering
4. Distribution-Based Clustering
5. Graph-Based Clustering
6. Fuzzy Clustering
7. Model-Based Clustering

- **1. Centroid-Based Clustering**

-  partitions the data into clusters, where each cluster is represented by a  centroid.
-  Minimize the distance between data points in the same cluster and maximize the separation between different clusters.
-  One of the most widely used centroid-based clustering algorithms is K-Means.
-   K-Means Algorithm:
    - Randomly select 𝑘 initial centroids
    - For each data point 𝑥𝑖, compute its distance to each centroid 𝑐𝑗 
    - Assign 𝑥𝑖 to the cluster with the nearest centroid:
    - Recalculate the centroids based on the data points in each cluster (update)
    - Repeat the Assignment and Update steps until convergence criteria are met.
-  Objective Function (Cost Function):
    - The K-Means algorithm minimizes the  objective function, often referred to as the “within-cluster sum of squares (WCSS)” or “inertia”:

- **2.Hierarchical Clustering**

- organizes data points into a tree-like structure, known as a dendrogram.
- This method creates a hierarchy of clusters, revealing relationships and structures within the dataset.
- Can be either agglomerative (bottom-up) or divisive (top-down), and it does not require a predefined number of clusters.
- 1. Step 1: First, we assign all the points to an individual cluster:
- 2. Step 2: Next, we will look at the smallest distance in the proximity matrix and merge the points with the smallest distance. We then update the proximity matrix.

 ## Dimensionality Reduction

-  transforming high-dimensional data into a lower-dimensional representation, preserving essential information while improving computational efficiency and model
-  extract the most relevant information from the original features while reducing redundancy and noise.
-  Computational Efficiency
-  Overfitting Prevention
-  Improved Model Performance
-  crucial tool in the realm of feature engineering

- **Different clustering approaches**

1. Principal Component Analysis (PCA)
2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
3. Autoencoders

- **1. Principal Component Analysis (PCA)**

- PCA transforms high-dimensional data into a lower-dimensional representation, capturing the most significant variance in the data. The resulting components, called 
  principal components.
- Step 1: Data Mean-Centering
- Step 2: Covariance Matrix Calculation
- Step 3: Eigen decomposition of Covariance Matrix
- Step 4: Principal Components Selection


## Anomaly Detection

- Anomaly detection detects data points in data that does not fit well with the rest of the data.
- Used for fraud detection, surveillance, diagnosis, data cleanup, and predictive maintenance.
- Anomalies or outliers come in three types.
    - Point Anomalies
    - Contextual Anomalies
    - Collective Anomalies
  


# Report on Analysis of Feedback Data for Course Improvement

## 1. Problem Statement:
The aim of this project is to analyze feedback data collected from Intel Certification course participants over various sessions to identify patterns, trends, and areas for improvement. The feedback data consists of ratings on different aspects such as content quality, training effectiveness, expertise of resource persons, and session organization, along with additional comments and suggestions provided by the participants.

## 2. Introduction:
In an educational institution or training program, gathering feedback from participants is crucial for assessing the effectiveness of the courses offered and understanding areas that require improvement. The feedback provides valuable insights into the quality of content, training methods, and overall organization of the sessions. Analyzing this feedback data can aid in making informed decisions to enhance the learning experience and satisfaction of the participants.

## 3. Dataset Details:

The feedback data was gathered via Google Forms, where participants provided their responses. The dataset includes responses from participants, detailing their perceptions and suggestions regarding course content, effectiveness, and overall organization.

## 4. Methodology

### Exploratory Data Analysis approach:

- **Frequency Calculation**:
Determining the occurrence of each unique value in the dataset, specifically focusing on the "Resource Person" and "Name" columns.

- **Percentage Calculation**:
Converting the frequency counts into percentages to analyze the relative contribution of each resource person and participant.

- **Rounding**:
Rounding the calculated percentages to improve readability and presentation.

Analyzing the distribution of feedback data across resource persons and participants is relevant as it sheds light on the engagement levels and participation patterns within the course. Understanding how feedback data is distributed facilitates the generation of insights into trends, preferences, and potential areas for improvement. It helps in identifying effective resource persons and active participants.The insights derived from EDA serve as a basis for data-driven decision-making in course administration and improvement efforts. Course administrators can allocate resources more efficiently, tailor course content to meet participants' needs, and address any issues identified through the analysis.

### ML approach:

K-means clustering was chosen as the ML technique to segment participants based on their satisfaction levels. The Elbow Method was used to determine the optimal number of clusters, and GridSearchCV was employed to fine-tune the hyperparameters of the K-means algorithm. The model was trained on features related to satisfaction levels (Content Quality, Effectiveness, Expertise, Relevance, Overall Organization), and participants were segmented into clusters based on their feedback.

## 5. Training Workflow

- **Importing Required Libraries**: 

Here, necessary libraries such as NumPy, Pandas, Seaborn, and Matplotlib are imported. %matplotlib inline is used for displaying plots within the Jupyter Notebook, and warnings are ignored to enhance readability.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

- **Loading Data**:

The data is loaded from a CSV file hosted on GitHub into a Pandas DataFrame named df_class.

```python
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```

- **Data Exploration**:

Initial exploration of the dataset is performed using head(), sample(), and info() functions to understand its structure and contents

```python
df_class.head()
df_class.sample(5)
df_class.info()
```

- **Data Wrangling**:

Unnecessary columns are dropped, and the remaining columns are renamed for better readability.

```python
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
df_class.columns = ["Name","Branch","Semester","Resource Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
```


## 6. EDA

### Percentage analysis of Resource_person wise distribution of data

| Resource Person         | Percentage |
|-------------------------|------------|
| Mrs. Akshara Sasidharan | 28.99      |
| Mrs. Veena A Kumar      | 26.57      |
| Mrs. Gayathri J L       | 14.98      |
| Mr. Arun Sebastian      | 14.98      |
| Dr. Anju Pratap         | 14.49      |

### Percentage analysis of Name wise distribution of data

| Name                       | Percentage |
|----------------------------|------------|
| Rizia Sara Prabin          | 3.86       |
| Sidharth V Menon           | 3.86       |
| Aaron James Koshy          | 3.38       |
| Rahul Biju                 | 3.38       |
| Abna Ev                    | 3.38       |
| Allen John Manoj           | 3.38       |
| Christo Joseph Sajan       | 3.38       |
| Jobinjoy Ponnappal         | 3.38       |
| Varsha S Panicker          | 3.38       |
| Nandana A                  | 3.38       |
| Rahul Krishnan             | 3.38       |
| Anjana Vinod               | 3.38       |
| Kevin Kizhakekuttu Thomas  | 3.38       |
| Lara Marium Jacob          | 3.38       |
| Abia Abraham               | 3.38       |
| Shalin Ann Thomas          | 3.38       |
| Jobin Pius                 | 3.38       |
| Sebin Sebastian            | 2.90       |
| Aaron Thomas Blessen       | 2.90       |
| Sani Anna Varghese         | 2.90       |
| Bhagya Sureshkumar         | 2.90       |
| Leya Kurian                | 2.90       |
| Jobin Tom                  | 2.90       |
| Anaswara Biju              | 2.42       |
| Muhamed Adil               | 2.42       |
| Aiswarya Arun              | 2.42       |
| Mathews Reji               | 1.93       |
| Marianna Martin            | 1.93       |
| Riya Sara Shibu            | 1.93       |
| Riya Sara Shibu            | 1.45       |
| MATHEWS REJI               | 1.45       |
| Sarang kj                  | 1.45       |
| Lisbeth Ajith              | 1.45       |
| Aiswarya Arun              | 0.97       |
| Muhamed Adil               | 0.97       |
| Marianna Martin            | 0.97       |
| Lisbeth Ajith              | 0.48       |
| Anaswara Biju              | 0.48       |
| Lisbeth                    | 0.48       |
| Jobin Tom                  | 0.48       |
| Aaron Thomas Blessen       | 0.48       |

### Data Visualisation

- **Subplot regarding Faculty-wise distribution of data**:

   The bar chart illustrates the distribution of data among various faculty members. Each bar represents a faculty member, and the height of the bar indicates the proportion of data associated with that faculty member.
<p align="left">
  <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/5abbbda1-2a27-4fce-b56d-111342f551b1" alt="[faculty wise distribution of data]" width="500">
</p>

- **Piechart regarding Resource person distribution of data**:

   The pie chart offers a comprehensive view of the distribution of resource persons by showcasing the proportion of each resource person relative to the whole. Each slice of the pie represents a resource person, and the size of the slice corresponds to their relative frequency.
<p align="left">
  <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/864c76b4-ccb0-4da2-834a-39036d9ec940" alt="[resource person]" width="500">
</p>

- **Summary of Responses**:

This report analyzes the evaluation of resource persons across various dimensions in training sessions. The dataset used for this analysis contains information on resource persons' performance metrics such as content quality, effectiveness, expertise, relevance, overall organization, and the branch they belong to.
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/f5494a6e-88df-4e9a-9e65-4bccf38bb588" alt="SUM 1" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/72ef6282-bbef-4c49-bc76-9e905d01ddd4" alt="SUM 2" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/411818a6-1dd4-4ead-b8e3-da7419d33834" alt="SUM 3" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/e3d4cc49-3f79-4490-8c70-8e8cc9776f21" alt="SUM 4" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/76eb6d4b-4182-4994-bac7-b986bd633db9" alt="SUM 5" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/f18482be-8d69-4f57-ae14-6a9b334e7ea8" alt="SUM 6" width="500" style="float:right;">
</div>
<div style="margin: 0 auto; width: 500px;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/8f6f116a-912b-404f-977d-9a8f5c2994f1" alt="SUM 7" width="400">
</div>

## 7. Machine Learning Model to study segmentation: K-means clustering

### Finding the best value of k using elbow method

The Elbow Method is a heuristic technique used to determine the optimal number of clusters in a dataset. It works by plotting the within-cluster sum of squares (WCSS) against the number of clusters and identifying the point where the rate of decrease in WCSS slows down, forming an "elbow" shape. This point represents the optimal number of clusters. By using the Elbow Method, we can make an informed decision about the appropriate number of clusters to use in the K-means algorithm, thereby ensuring that the segmentation is meaningful and interpretable. 

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/df84c285-a905-4b92-a31b-9b287d18cf3e" alt="elbow_method" width="400">

### Using Gridsearch method

K-means clustering involves certain hyperparameters, such as the number of clusters (k) and the initialization method. GridSearchCV is a technique used for hyperparameter tuning, wherein a grid of hyperparameter values is specified, and the algorithm searches through this grid to find the combination of hyperparameters that yields the best performance. By employing GridSearchCV, we can systematically explore different hyperparameter configurations and select the optimal ones, thereby improving the effectiveness of the K-means algorithm in segmenting participants based on their satisfaction levels.

```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

### Implementing K-means clustering

K-means clustering is a widely used unsupervised learning algorithm for clustering data points into distinct groups based on similarities in their features. In the context of participant feedback analysis, we are dealing with unlabeled data where we aim to discover inherent patterns or structures. K-means clustering allows us to segment participants into clusters based on their satisfaction levels without requiring labeled data, making it an appropriate choice for this analysis. 

```python
# Perform k-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```

### Extracting labels and cluster centers

Adding cluster labels to the DataFrame to perform f analysis or visualization based on the cluster assignments, such as comparing the characteristics of different clusters or visualizing the data points within each cluster. 

```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
```

### Visualizing the clustering using first two features

This visualization provides a graphical representation of the clusters formed by the K-means algorithm
<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/9b2a2739-cca8-4ff8-b4c0-c63c02056a9a" alt="Kmeans" width="400">

### Perception on content quality over Clustors

The pd.crosstab() function in Python's pandas library is used to compute a cross-tabulation of two (or more) factors. It creates a frequency table where the intersection of rows and columns represents the frequency count of observations that match the respective row and column values.

|           | Cluster 0 | Cluster 1 | Cluster 2 |
|-----------|-----------|-----------|-----------|
| Content Quality |           |           |           |
| 3         | 0         | 7         | 11        |
| 4         | 18        | 67        | 1         |
| 5         | 95        | 8         | 0         |


## 8. Results and conclusion

The analysis of the distribution of data among faculty members provides valuable insights into the contribution patterns within the dataset. While Mrs. Akshara Sasidharan and Mrs. Veena A Kumar emerge as the primary contributors, it's noteworthy that Mrs. Gayathri J L, Mr. Arun Sebastian, and Dr. Anju Pratap also play significant roles in data provision.

Understanding these distribution patterns is pivotal for various reasons:

Resource Allocation: Recognizing the key contributors enables more efficient resource allocation, ensuring that support, recognition, and further opportunities are appropriately distributed among faculty members based on their contributions.

Research Focus: Identifying individuals with substantial contributions can help steer research initiatives, ensuring that efforts are aligned with the areas where data availability is highest, thus maximizing research outcomes and impact.

Collaboration Opportunities: Awareness of data distribution facilitates collaboration opportunities, allowing researchers to leverage each other's strengths and resources for interdisciplinary projects or collaborative research endeavors.

Quality Control: Monitoring the distribution of data can aid in quality control efforts by pinpointing areas where data collection may be lacking or where discrepancies exist, prompting corrective actions to ensure data integrity.

In conclusion, the combination of bar and pie charts provides a clear visual representation of the data distribution among faculty members, offering actionable insights for decision-making processes related to resource management, research direction, collaboration strategies, and quality assurance within the academic context.








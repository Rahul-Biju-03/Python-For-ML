# Data analysis - Student's Behavior

## 1. Problem Statement:

The problem at hand involves analyzing student behavior to identify patterns and clusters that can provide insights into their engagement levels. By understanding these patterns, educational institutions can tailor interventions and strategies to improve student engagement and academic performance.

## 2. Introduction:

In today's educational landscape, understanding student behavior is crucial for educators and administrators to provide effective support and guidance. Analyzing various aspects of student engagement, such as participation in class discussions, resource utilization, and interaction with announcements, can offer valuable insights into their learning experience and academic outcomes.

## 3. Dataset Details:

| Variable                | Description                               |
|-------------------------|-------------------------------------------|
| gender                  | Gender of the student (M/F)              |
| NationalITy             | Nationality of the student              |
| PlaceofBirth            | Place of birth of the student            |
| StageID                 | Stage of education (e.g., lowerlevel)    |
| GradeID                 | Grade level                               |
| SectionID               | Section identifier                        |
| Topic                   | Topic of study                            |
| Semester                | Semester of the academic year            |
| Relation                | Relationship of the respondent to the student (e.g., father) |
| raisedhands             | Number of times the student raised their hand in class |
| VisITedResources        | Number of resources visited by the student |
| AnnouncementsView       | Number of announcements viewed by the student |
| Discussion              | Number of times the student participated in class discussions |
| ParentAnsweringSurvey   | Whether the parent answered the survey (Yes/No) |
| ParentschoolSatisfaction| Parent's satisfaction with the school (Good/Bad) |
| StudentAbsenceDays      | Number of days the student was absent from school |
| Class                   | Categorized class label (e.g., M: Medium) |

This dataset provides detailed information about student behavior, demographics, and academic performance. It includes variables such as gender, nationality, grade level, participation metrics (raised hands, resources visited, announcements viewed, discussions participated), parental involvement, student absence, and class categorization. 

The dataset appears to be comprehensive and suitable for exploring various aspects of student engagement and academic outcomes. Further analysis, including exploratory data analysis (EDA) and clustering techniques, can provide valuable insights into student behavior and help identify strategies to improve educational outcomes.






## 4. Methodology

### Exploratory Data Analysis approach:
In the exploratory data analysis (EDA) phase, we analyzed a dataset containing information about student behavior. The dataset included features such as gender, nationality, class, and various behavioral metrics such as the number of raised hands, visited resources, announcements viewed, and discussions participated in. Here's a brief summary of the EDA:


#### Numeric Variables Distribution:
- We visualized the distribution of numeric variables using pair plots. These plots allowed us to explore the relationships between different numeric variables and identify potential patterns or trends.

#### Categorical Variables Distribution:
- For categorical variables, such as gender, nationality, and class, we created count plots to visualize their distributions. This helped us understand the frequency distribution of different categories within each variable.

#### Correlation Analysis:
- We computed the correlation matrix for numeric variables and visualized it using a heatmap. This allowed us to identify correlations between different numeric variables. For example, we observed correlations between "raised hands," "visited resources," and other behavioral metrics.

#### Cross-Tabulation:
- We performed cross-tabulation between the 'NationalITy' and 'Class' variables to examine the distribution of student classes across different nationalities. This analysis provided insights into potential relationships between nationality and academic performance.

#### ANOVA Test:
- We conducted an ANOVA test to analyze the relationship between the 'raised hands' variable and student classes ('L', 'M', 'H'). The ANOVA results provided statistical evidence of whether there are significant differences in the mean number of raised hands across different class levels.

Overall, the EDA helped us gain a deeper understanding of the dataset's characteristics, identify potential relationships between variables, and uncover insights that can inform further analysis and decision-making processes. It provided valuable insights into student behavior and engagement levels, laying the foundation for subsequent analysis such as clustering or predictive modeling.


### ML approach:

#### Elbow Method and Silhouette Analysis
- **Elbow Method**: The elbow method helped determine the optimal number of clusters based on the within-cluster sum of squares (WCSS). From the elbow plot, it appears that the optimal number of clusters is around 2 or 3.
- **Silhouette Analysis**: Silhouette scores were computed for a range of cluster numbers to evaluate cluster quality. The silhouette scores suggest that 2 clusters might be optimal, as it yields the highest silhouette score.

#### K-means Clustering
- **Number of Clusters**: Based on the elbow method and silhouette analysis, we chose to proceed with 2 clusters.
- **Cluster Visualization**: Scatter plots were used to visualize the clusters based on the features 'raisedhands' and 'VisITedResources'. Each data point was colored according to its assigned cluster, and centroids were marked with 'X' symbols.
- **Cluster Profiles**: The cluster profiles were visualized using a scatter plot of 'raised hands' versus 'visited resources', with each point representing a student colored by cluster membership. Centroids, representing the mean values of each cluster, were marked for reference.

#### Cluster Characteristics
- **Cluster Summary**: The mean and median values of 'raised hands' and 'visited resources' were compared for each cluster. Cluster 0 shows lower values for both features compared to Cluster 1.
- **Cross-Tabulation**: Cross-tabulation was performed to examine the distribution of clusters across different levels of 'raised hands'. It provides a summary of how many students fall into each cluster based on their 'raised hands' behavior.

Overall, the K-means clustering analysis helped identify distinct groups of students based on their engagement levels, particularly in terms of 'raised hands' and 'visited resources'. The clustering results can be used to tailor interventions or strategies to improve student engagement and academic performance.


## 5. Training Workflow

### Importing Required Libraries

Here, necessary libraries such as NumPy, Pandas, Seaborn, and Matplotlib are imported.

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```

### Loading Data

The data is loaded from a CSV file  into a Pandas DataFrame named data.

```python
data = pd.read_csv('xAPI-Edu-Data.csv')
```

### Data Exploration

Initial exploration of the dataset is performed using head(), sample(), describe(), isnull().sum() and info() functions to understand its structure and contents

```python
data.head(5)
data.info()
data.sample()
print(data.describe())
print(data.isnull().sum())
```
### Feature Selection 

- **distribution of categorical variables**:
  
```python

categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 
                    'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
                    'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']
 ```

- **distribution of numerical variables**:

```python
numeric_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
```
  
## 6. EDA

### Visualize the distribution of numeric variables

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/2a250919-809a-4c93-904b-819674dc0a59" alt="1 numeric" width="400">

### Visualize the distribution of categorical variables

<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/407d0471-2e10-4a57-8622-c4373e3a119f" alt="2.1" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/9e40ffa8-42b2-4f82-bef8-4e691ca2a60f" alt="2.2" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/61355dc6-ebfb-4100-8a50-04e2922c5c1b" alt="2.3" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/14c5c030-b5e9-4a46-a060-6febb7632a9e" alt="2.4" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/373bb1c2-0f0b-4c71-8be3-02e3283df5ef" alt="2.5" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/5f36f7c7-82d0-4234-9fef-c5d3e8f9f331" alt="2.6" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/08571f0d-8b9d-4878-88eb-eeb8714071c3" alt="2.7" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/66045b90-cc6a-4825-9344-7b10521114ff" alt="2.8" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/46deec94-fbb2-456f-9c5e-bb11f600fb38" alt="2.9" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/f82069c4-1aa1-4966-bb8f-b931625fd35a" alt="2.10" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/7f0cacf6-f056-4d2c-a3b9-0fcca57b23a4" alt="2.11" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/afe90f96-c075-47d3-85f1-6f477bc4ae14" alt="2.12" width="500" style="float:right;">
</div>

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/56678b71-46c9-4248-9c2e-0a6b80819363" alt="2.13" width="500">

### Correlation matrix

Compute the correlation matrix between numeric variables and visualize it using a heatmap. This will help identify any significant correlations between variables.

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/13738db8-9cf4-4380-af57-aa4a02cdf311" alt="3 Correlation matrix" width="500">

### Cross-tabulation: 

Creating cross-tabulations between pairs of categorical variables to observe relationships between them.

| Class        |        |  H  |  L  |  M  |
|--------------|--------|-----|-----|-----|
| NationalITy  |        |     |     |     |
|--------------|--------|-----|-----|-----|
| Egypt        |        |  2  |  3  |  4  |
| Iran         |        |  0  |  2  |  4  |
| Iraq         |        | 14  |  0  |  8  |
| Jordan       |        | 53  | 37  | 82  |
| KW           |        | 36  | 68  | 75  |
| Lybia        |        |  0  |  6  |  0  |
| Morocco      |        |  1  |  1  |  2  |
| Palestine    |        | 12  |  0  | 16  |
| SaudiArabia  |        |  6  |  1  |  4  |
| Syria        |        |  2  |  2  |  3  |
| Tunis        |        |  3  |  4  |  5  |
| USA          |        |  3  |  1  |  2  |
| lebanon      |        |  9  |  2  |  6  |
| venzuela     |        |  1  |  0  |  0  |

### ANOVA test

Using ANOVA test to compare numeric variables across different levels of categorical variables. The ANOVA test yielded the following results:

- ANOVA F-Statistic: 176.389
- ANOVA p-value: 4.51020980899929e-58 (extremely small)

The obtained ANOVA F-statistic of 176.389 is large, indicating a significant difference among the means of the groups ('L', 'M', 'H' classes) regarding the frequency of raised hands in class.

Furthermore, the associated p-value of 4.51020980899929e-58 is extremely small, much smaller than any reasonable significance level (e.g., 0.05). This suggests strong evidence to reject the null hypothesis, indicating that there is a significant difference in the mean frequency of raised hands across at least two of the academic performance classes.

## 7. Machine Learning Model to study segmentation: K-means clustering

### Finding the best value of k using elbow method

The Elbow Method is a heuristic technique used to determine the optimal number of clusters in a dataset. It works by plotting the within-cluster sum of squares (WCSS) against the number of clusters and identifying the point where the rate of decrease in WCSS slows down, forming an "elbow" shape. This point represents the optimal number of clusters. By using the Elbow Method, we can make an informed decision about the appropriate number of clusters to use in the K-means algorithm, thereby ensuring that the segmentation is meaningful and interpretable. 

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/de222188-7320-4b64-8524-76ec9255c24e" alt="elbow" width="500">

k=2 is the ideal value from this graph

### Using Silhouette Scores

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/ecd1b959-f98f-4492-acd9-dd36a82c8598" width="500">

Best Number of Clusters: 2

### Implementing K-means clustering

```python
# Perform k-means clustering
k = 2 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```

### Extracting labels and cluster centers

```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
data['Cluster'] = labels
```

### Visualizing the clustering using first two features

<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/606edc75-fa01-4967-8ef8-84c38db8be78" alt="KMEANS1" width="500">
<img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/6c2cb661-b575-47e5-a392-77386a520473" alt="KMEANS2" width="500">


| Cluster | Raised Hands (mean) | Raised Hands (median) | Visited Resources (mean) | Visited Resources (median) |
|---------|----------------------|------------------------|--------------------------|----------------------------|
| 0       | 19.10                | 15.0                   | 22.35                    | 15.0                       |
| 1       | 67.94                | 72.0                   | 79.61                    | 82.0                       |

## 8. Results

The clustering analysis based on the "raised hands" and "visited resources" features resulted in two distinct clusters. Here are the key findings from the analysis:

- **Cluster 0:** This cluster represents students with lower levels of engagement. The mean and median values of "raised hands" and "visited resources" are relatively low compared to Cluster 1.
- **Cluster 1:** This cluster represents students with higher levels of engagement. The mean and median values of "raised hands" and "visited resources" are significantly higher compared to Cluster 0.

## 9. Conclusion

The clustering analysis provides valuable insights into the engagement levels of students based on their participation in classroom activities. The results indicate a clear distinction between two groups of students:

1. **Low Engagement Group (Cluster 0):** Students in this group exhibit lower levels of engagement in the classroom. They tend to raise their hands less frequently and visit educational resources less often compared to the high engagement group. This group may benefit from interventions or strategies aimed at increasing their participation and interaction in classroom activities to enhance their learning experience.

2. **High Engagement Group (Cluster 1):** Students in this group demonstrate higher levels of engagement in the classroom. They actively participate by raising their hands frequently and make regular use of educational resources. This group represents students who are actively involved in the learning process and may require different approaches to support their continued engagement and academic success.

## 10. Implications and Recommendations

- Teachers and educators can use the clustering results to identify students who may require additional support or encouragement to increase their engagement levels.
- Tailored interventions or teaching strategies can be developed to address the specific needs of students in each cluster. For example, targeted interventions may be implemented to enhance participation and interaction for students in the low engagement group.
- The clustering analysis can inform classroom management practices, allowing teachers to create more inclusive and engaging learning environments that cater to the diverse needs of students.

Overall, the clustering analysis offers valuable insights that can inform decision-making and interventions aimed at promoting student engagement and enhancing the overall learning experience.





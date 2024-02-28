 # Report on Analysis of Decision Tree Classifier for Car Evaluation Dataset

## 1. Problem Statement:
The objective of this report is to analyze the performance of a decision tree classifier on a car evaluation dataset. Specifically, we aim to understand how varying the tree depth and min samples split parameters affects the accuracy of the classifier.

## 2. Introduction:
Decision tree classifiers are powerful tools for classification tasks, providing intuitive and interpretable models. However, their performance can vary depending on the dataset and the parameters chosen during model training. In this report, we investigate the impact of tree depth and min samples split on the accuracy of a decision tree classifier applied to a car evaluation dataset.

## 3. Dataset Details:
The car evaluation dataset consists of attributes such as buying price, maintenance price, number of doors, number of persons, luggage boot size, safety rating, and the class of the car (acceptable, unacceptable, good, or very good). The dataset contains no missing values and has been preprocessed to encode categorical variables.

## 4. Methodology:
We employ a decision tree classifier with the entropy criterion for our analysis. We vary two key parameters: tree depth and min samples split. By systematically exploring different combinations of these parameters, we aim to identify the configuration that yields the highest accuracy on a held-out test set.

## 5. Model training process

- **Importing Required Libraries**:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import category_encoders as ce
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pickle
import pandas as pd
from sklearn import tree
```

- **Loading Data**:

```python
df = pd.read_csv('car_evaluation.csv')
```

- **Assigning Column Names to DataFrame**

```python
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
col_names
```

- **EDA**

Data Overview

```python
print(df.head())
df.shape
df.info()
df.isnull().sum()
```
Examining the value counts of categorical variables

```python
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
for col in col_names:
  print(df[col].value_counts())
```

Analysis of `class` distribution

```python
df['class'].value_counts()
```
Feature-Target Separation

```python
X = df.drop(['class'], axis=1)
y = df['class']
```

- **Splitting Data**

```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=42)
```

- **Data Encoding**

Encode categorical variables with ordinal encoding

```python
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```

- **Creating model**

Create a Decision Tree Classifier with the "entropy" criterion
 
```python
clf = DecisionTreeClassifier(criterion="entropy")
```

- **Training model**
  
Train the classifier on the training set

```python
clf.fit(X_train, y_train)
```
- **Prediction on Testing Set**

```python
y_pred = clf.predict(X_test)
```
- **Model Performance Evaluation**

  - Training set score: 1.0000
  - Test set score: 0.9702

- **Graphical representation of the decision tree classifier**

![tree](https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/08a42adb-f253-4966-8aab-d0a3322d5a8f)

- **Define lists to store results**
  
```python
depths = [3, 5, 7, 9]
min_samples_splits = [2, 5, 10, 20]
results = []
```

- **Perform systematic analysis**
  
```python
for depth in depths:
    for min_samples_split in min_samples_splits:
        # Create and train decision tree classifier
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_split=min_samples_split, random_state=42)
        clf.fit(X_train, y_train)
```

| Depth | Min Samples Split | Accuracy |
|-------|-------------------|----------|
| 3     | 2                 | 0.8053   |
| 3     | 5                 | 0.8053   |
| 3     | 10                | 0.8053   |
| 3     | 20                | 0.8053   |
| 5     | 2                 | 0.8789   |
| 5     | 5                 | 0.8789   |
| 5     | 10                | 0.8789   |
| 5     | 20                | 0.8789   |
| 7     | 2                 | 0.9281   |
| 7     | 5                 | 0.9281   |
| 7     | 10                | 0.9281   |
| 7     | 20                | 0.9333   |
| 9     | 2                 | 0.9614   |
| 9     | 5                 | 0.9561   |
| 9     | 10                | 0.9368   |
| 9     | 20                | 0.9404   |

- **Decision Tree Accuracy vs. Min Samples Split**

![min_samples](https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/f5dd4409-7a82-4bfa-94cc-bf3ea3813f10)

## 6. Result

- Accuracy tends to increase with the increase in tree depth.
- Accuracy remains constant for a depth of 3, regardless of the min_samples_split parameter.
- Accuracy improves significantly with a depth of 5 and above, especially with a min_samples_split of 20.
- The best combination of parameters, achieving the highest accuracy of approximately 96.14%, is a depth of 9 with a min_samples_split of 2.

## 7. Conclusion

These findings indicate that deeper trees with smaller min_samples_split tend to provide better predictive performance on this dataset. However, it's important to keep in mind the potential risk of overfitting with deeper trees, especially if the dataset is small. Therefore, further analysis, such as cross-validation, could be beneficial to ensure the robustness of the model.

In conclusion, the decision tree classifier shows promising performance on the car evaluation dataset, achieving high accuracies across various parameter configurations. Further optimization and fine-tuning of the model parameters could potentially lead to even better performance. Additionally, it would be beneficial to assess the model's robustness through techniques such as cross-validation and to compare its performance against other machine learning algorithms. Overall, the results demonstrate the effectiveness of decision tree classifiers for the task of car evaluation.





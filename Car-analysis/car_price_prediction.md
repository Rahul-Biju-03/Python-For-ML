# Report on Car price prediction

## 5. Training Workflow

- **Importing Required Libraries**:

  This script imports necessary libraries and modules for building a linear regression model, evaluating its performance, and visualizing the results. It also includes functionality for handling data with pandas, saving and loading models with pickle, and generating visualizations using matplotlib and seaborn. Additionally, it sets up the environment for splitting the dataset into training and testing sets for model validation.

```python 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

```
- **Loading Data**:

  This line of code reads the data from a CSV file named 'car_dataset.csv' and loads it into a pandas DataFrame called 'df'. 

```python
df = pd.read_csv('car_dataset.csv')
```

- **EDA**:

EDA is the process of examining and understanding data to summarize its main characteristics, often involving techniques like summarizing data distributions, identifying patterns, and detecting outliers. The code performs several key EDA tasks:

- Data Inspection: Checking the dimensions, data types, and memory usage of the DataFrame.
- Data Summarization: Displaying the first few rows and providing summary statistics to understand the dataset's structure and distribution.
- Data Exploration: Examining unique values for each feature to understand the diversity and cardinality of categorical variables.
- Column Identification: Retrieving column names for reference during analysis.

```python
# check the shape
df.shape
# first five rows of the dataframe
df.head()
# describe the dataframe with some statistical info
df.describe()
# check data types in the dataframe
df.info()
# check unique data for each feature in the dataframe
df.nunique()
# column names of the dataframe
df.columns
```
- **Data Selection**:

  Select the numerical columns

```python
numerical_columns = ['wheelbase', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg']
```

- **Correlation of `Price` with other numerical variables**:

  Higher absolute correlation coefficients (close to 1 or -1) suggest stronger relationships between the independent variable and the price, while coefficients closer to 0 indicate weaker relationships.

```python
# Calculate correlation coefficients with respect to "price"
correlation_with_price = df[numerical_columns].corrwith(df['price']).abs().sort_values(ascending=False)

# Convert correlation_with_price series to DataFrame for better tabulation
correlation_table = correlation_with_price.reset_index()
correlation_table.columns = ['Feature', 'Correlation with Price']

# Generate pretty table
pretty_table = tabulate(correlation_table, headers='keys', tablefmt='pretty', showindex=False)

print(pretty_table)
```
| Feature          | Correlation with Price |
|------------------|------------------------|
| enginesize       | 0.8741448025245112     |
| horsepower       | 0.8081388225362212     |
| citympg          | 0.68575133602704       |
| wheelbase        | 0.57781559829215       |
| boreratio        | 0.5531732367984434     |
| peakrpm          | 0.08526715027785685    |
| stroke           | 0.079443083881931      |
| compressionratio | 0.06798350579944262    |

**The feature with the highest correlation with 'price' is 'enginesize' with a correlation coefficient of 0.874145.**

- **Creating a New DataFrame**:

Create a new DataFrame using `enginesize` and `price`.

```python
x = new_df['enginesize']
y = new_df['price']
```

- **Splitting data**:

20% of the data will be used for testing, while the remaining 80% will be used for training.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

X_train: (164,)
  
X_test: (41,)
  
Y_train: (164,)
  
Y_test: (41,)

- **Model Creation**:
  
Creates a Linear Regression model.
This model is commonly used for predicting a target variable based on one or more independent variables.
Linear regression assumes a linear relationship between the independent variables and the target variable.

```python
model = LinearRegression()
```

- **Model Training**:

  - The fit() method is used to train the model on the training data. In this case, we are fitting the model using the feature 'enginesize' (X_train) and the corresponding target variable 'price' (y_train).
  - Since Linear Regression expects the features to be in a two-dimensional array, we use .values.reshape(-1, 1) to reshape the feature 'enginesize' (X_train) into a column vector (i.e., a 2D array with a single column).
   This is necessary because scikit-learn expects the input features to be a 2D array, even if there is only one feature.

```python
model.fit(x_train.values.reshape(-1,1), y_train)
```

- **Evaluation Metrics**:

  - MSE of 16835544.03813768 indicates that, on average, the squared difference between the actual and predicted prices is approximately 16,835,544.
  - RMSE of 4103.113944084137 indicates that, on average, the predicted prices deviate from the actual prices by approximately $4,103.11.
  - MAE of 3195.0312395000433 indicates that, on average, the absolute difference between the actual and predicted prices is approximately $3,195.03.
  - R2 score of 0.7825324721447274 indicates that the model explains approximately 78.25% of the variance in the target variable, which suggests a reasonably good fit.
 
- **Plotting actual vs. predicted values**:

 # Report on Car price prediction

## 1. Problem Statement:
The goal of this project is to develop a predictive model to estimate the price of cars based on certain features. We aim to explore the relationship between various numerical attributes of cars such as engine size, horsepower, etc., and their prices. By building a linear regression model, we intend to predict car prices accurately, which could be valuable for both buyers and sellers in the automotive market.

## 2. Introduction:
Predicting car prices accurately is crucial for various stakeholders in the automotive industry, including dealerships, manufacturers, and consumers. Understanding the factors that influence car prices can help in making informed decisions regarding pricing strategies, purchasing, and investment.

In this project, we utilize a dataset containing information about different cars, including their attributes like engine size, horsepower, and more, along with their corresponding prices. By analyzing this data and building a predictive model, we aim to provide insights into how these attributes contribute to the pricing of cars. This analysis can assist in better understanding market trends and predicting future pricing trends.

## 3. Dataset Details:

The provided dataset contains information about various cars, with each row representing a different car model.

| Attribute         | Description                                                                                               |
|-------------------|-----------------------------------------------------------------------------------------------------------|
| ID                | An identifier for each car entry.                                                                         |
| Symboling        | A rating assigned to each car indicating its risk level (e.g., insurance risk).                           |
| Name              | The name or model of the car.                                                                             |
| Fuel Types        | The type of fuel the car uses (e.g., gas, diesel).                                                        |
| Aspiration        | The method of aspiration for the engine (e.g., std for standard, turbo for turbocharged).                  |
| Door Numbers      | The number of doors the car has.                                                                         |
| Car Body          | The body style of the car (e.g., sedan, convertible, hatchback).                                          |
| Drive Wheels      | The type of wheels that receive power from the engine (e.g., fwd for front-wheel drive, rwd for rear-wheel drive). |
| Engine Location   | The location of the engine in the car (e.g., front, rear).                                                |
| Wheelbase         | The distance between the centers of the front and rear wheels.                                            |
| Engine Size       | The volume of the engine displacement, often measured in liters or cubic centimeters.                     |
| Fuel System       | The type of fuel delivery system used by the engine (e.g., mpfi, 2bbl).                                   |
| Bore Ratio        | The diameter of each cylinder in the engine.                                                              |
| Stroke            | The length of the piston stroke in the engine.                                                            |
| Compression Ratio | The ratio of the volume of the combustion chamber when the piston is at the bottom of its stroke to the volume when it's at the top. |
| Horsepower        | The power output of the engine.                                                                           |
| Peak RPM          | The maximum revolutions per minute of the engine.                                                         |
| City MPG          | The estimated miles per gallon (MPG) the car can achieve in city driving conditions.                      |
| Highway MPG       | The estimated miles per gallon (MPG) the car can achieve on the highway.                                  |
| Price             | The price of the car.                                                                                    |




## 4. Methodology

Data Loading and Inspection: Load the dataset into a DataFrame and inspect its structure, including the features and target variable.

Data Preprocessing: Perform necessary data preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features if required.

Exploratory Data Analysis (EDA): Explore the dataset to gain insights into the distribution of data, identify correlations between features and the target variable, and visualize patterns using plots and statistical summaries.

Feature Selection: Select relevant features that have a significant correlation with the target variable (price) for model training.

Model Training: Split the data into training and testing sets, then train a linear regression model using the selected features.

Model Evaluation: Evaluate the trained model using various evaluation metrics such as MSE, MAE, R2, and RMSE to assess its performance in predicting car prices.

Visualization: Visualize the actual vs. predicted prices using scatter plots and regression lines to understand the model's predictive accuracy and identify any discrepancies.

Conclusion: Summarize the findings, including the model's performance, key insights from the analysis, and recommendations for further improvements or actions based on the results.

## 5. Model training process

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

- **Comparing Engine Size and Fuel Type with Price using Scatter Plot**
  
![enginesize](https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/fdc9f819-1d89-4cb7-93f8-63cbb080de6c)
![fueltype](https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/63a28142-2a69-414d-a14f-9b44d0694c0a)

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

![actual vs predicted](https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/39cb2b13-e403-42bf-b0c6-ca8d136c5c4c)

- **Plotting Regression model line**:
  
![regression model line](https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/602c3da4-93d7-4660-8ddf-892447bf2cd1)

- **Predicting prices based on engine size**:

| Engine Size | Predicted Price |
|-------------|-----------------|
| 91          | 7339.335167318599  |
| 161         | 18841.416787940445 |
| 136         | 14733.530494861214 |
| 61          | 2409.8716156235205 |
| 109         | 10297.013298335645 |
| 146         | 16376.685012092908 |
| 92          | 7503.650619041768  |
| 92          | 7503.650619041768  |
| 181         | 22127.72582240383  |
| 92          | 7503.650619041768  |
| 164         | 19334.363143109953 |
| 203         | 25742.665760313554 |
| 70          | 3888.7106811320446 |
| 134         | 14404.899591414876 |
| 90          | 7175.019715595428  |
| 146         | 16376.685012092908 |
| 132         | 14076.268687968539 |
| 136         | 14733.530494861214 |
| 110         | 10461.328750058812 |
| 92          | 7503.650619041768  |
| 110         | 10461.328750058812 |
| 120         | 12104.483267290507 |
| 132         | 14076.268687968539 |
| 146         | 16376.685012092908 |
| 171         | 20484.57130517214  |
| 97          | 8325.227877657613  |
| 98          | 8489.543329380784  |
| 120         | 12104.483267290507 |
| 98          | 8489.543329380784  |
| 97          | 8325.227877657613  |
| 109         | 10297.013298335645 |
| 109         | 10297.013298335645 |
| 151         | 17198.262270708754 |
| 122         | 12433.114170736844 |
| 97          | 8325.227877657613  |
| 209         | 26728.55847065257  |
| 109         | 10297.013298335645 |
| 121         | 12268.798719013677 |
| 90          | 7175.019715595428  |
| 304         | 42338.526384353645 |
| 90          | 7175.019715595428  |

## 6. Result

After conducting an in-depth analysis and implementing a linear regression model, we have successfully developed a predictive model to estimate car prices based on engine size. Here are the key findings and results:

Key Findings:
Feature Importance: Engine size has been identified as the most influential feature in determining car prices, exhibiting the highest correlation coefficient with the target variable (price) among the numerical features considered in the dataset.

Model Performance: The trained linear regression model demonstrates reasonable performance in predicting car prices. Evaluation metrics reveal that:

Mean Squared Error (MSE) is approximately 16,835,544.04.
Root Mean Squared Error (RMSE) is approximately 4,103.11.
Mean Absolute Error (MAE) is approximately 3,195.03.
R-squared (R2) score is approximately 0.78.
These metrics indicate that the model captures a significant portion of the variance in the target variable and provides a satisfactory level of predictive accuracy.

Prediction Insights: Despite the inherent variability in car prices, the model's predictions align well with the actual prices, as evidenced by the evaluation metrics and scatter plot visualization.

## 7. Conclusion

In conclusion, our analysis and model development have provided valuable insights into the pricing dynamics of cars, with a specific focus on the role of engine size in influencing car prices. By leveraging machine learning techniques, we have built a predictive model that effectively estimates car prices based on this crucial feature.

The predictive model serves as a valuable tool for stakeholders in the automotive industry, enabling them to make informed decisions regarding pricing strategies, purchasing, and investment. Furthermore, the model's performance metrics indicate its reliability and utility in real-world applications.

Moving forward, further refinements to the model, such as incorporating additional relevant features and exploring advanced regression techniques, could enhance its predictive capabilities and broaden its applicability in the automotive market.

Overall, this project underscores the importance of data-driven approaches in understanding market trends and optimizing decision-making processes within the automotive industry.

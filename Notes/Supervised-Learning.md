# Supervised-Learning

## Definition

- Approach where algorithms learn from labeled training data to make predictions or decisions.

## Key Elements of Supervised Learning

- input data
- output labels
- labeled dataset
- model
- training
  
## Real-World Use Cases of Supervised Learning

- Image Classification
- Speech Recognition
- Fraud Detection
- Sentiment Analysis
- Predictive Maintenance
- Health Diagnosis
- Financial Forecasting

## Popular Supervised Learning Algorithms

**1. Regression Models**

- Model the relationship between a dependent variable and one or more independent variables.
- Predict the value of the dependent variable based on the values of the independent variables.

**Assumptions**

- Linearity: The relationship between the independent and dependent variables is assumed to be linear.
- Independence: Observations are assumed to be independent of each other.
- Homoscedasticity: Residuals (differences between observed and predicted values) should have constant variance.
- Normality of Residuals: Residuals are assumed to be normally distributed.
- No Perfect Multicollinearity: Independent variables are not perfectly correlated.

**1.1. Simple Linear Regression**

- Only one independent variable (X)
- 𝑌 = 𝛽0 + 𝛽1𝑋 + 𝜖

**1.2. Multiple Linear Regression**

-  multiple independent variables (𝑋1, 𝑋2, … , 𝑋𝑛)
-  𝑌 = 𝛽0 + 𝛽1𝑋1 + 𝛽2𝑋2 + … + 𝛽𝑛𝑋𝑛 + 𝜖

**1.3. Nonlinear Linear Regression**

- Model the relationship between a dependent variable and one or more independent variables when the relationship is not linear.
- More complex and flexible modeling of relationships that may exhibit curvature, exponential growth, saturation, or other nonlinear patterns.

**2. Classification Algorithm**

-  Assign predefined labels or categories to input data based on its features.
-  Mapping between input features and corresponding output labels, enabling the algorithm to make
   predictions on new, unseen data

**Key Characteristics**

- Supervised Learning:
- Input Features:
- Output Labels or Classes:
- Training Process:
- Model Representation:
- Prediction:
- Evaluation Metrics

 **Common Classification Algorithms**
 
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- K-Nearest Neighbors (KNN)
- Naive Bayes

**1. Logistic Regression**

- For binary and multiclass classification in machine learning.
- Used for classification tasks, not regression.
- The model is well-suited for scenarios where the dependent variable is categorical, and the goal is to predict the probability of an observation belonging to a particular class

**2. Decision Trees**

- For both classification and regression tasks.
- These models make decisions based on the features of the input data by recursively partitioning it into subsets.
- Each partition is determined by asking a series of questions based on the features, leading to a tree-like structure.
- node-feature/attribute , branch-link

**STEPS FOR DECISION TREE**

- create a root node (calculate entropy ,avg info ,info gain)
   - check nb for eqns.
   - entropy=0 when either all are positive or all are negative.
   - value=0 for entropy is good.
   -  Higher gain value is good. Attribute with highest gain value is selected as the root node.
  
- Left node
   - Recalculate using eqns for all other attributes again w/o root node.
   - Select the next attrbute with the highest gain value and put it as the left node.
  
- Recalculate once again. Attribute with highest gain value is placed at right node.

- Final decision tree is completed.

**STEPS FOR GINI**

- calculate gini value for all attributes
- Attribute with lowest gini value is the root node.
- Repeat for left node and then right node.

**ISSUES IN DECISION TREE**

- accuracy first increase then decrease.
- overfitting

**CODE FOR DECISION TREE**

- Set criterion=entropy if we want gain value.
- or set criterion=gini.

**3. Support Vector Machines**

find optimal hyperplane (plane that has max. distance from both the classes used)

 **TYPES OF SVM**

- linear
  -  Data set:linearly sperable.
  - divides categories using a simple straight lines.
  - Is able to divide data into 2 categories using a straight line.
  
- non linear
  - Data set:not linearly seperable.
  - transforms data into 2d.
           
**4. ENSEMBLE LEARNING**

- Combines decisions from multiple models to improve performance.
- enhances accuracy
- mitigates error in individual models

**TYPES OF ENSEMBLE LEARNING**

- max voting-highest number of votes
- averaging - take avg of predictions.
- weighted averaging -models are assigned weights. Higher weight-more importance.
- soft voting takes the average value(average voting)
- hard voting is majority voting


  

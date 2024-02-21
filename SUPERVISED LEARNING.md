## Regression

TYPES:

linear
logistic
decision tree

attributes/features
dependant variables
independant variables

Linear regression
pattern recognition
model

input dataset along with algorithm(such as linear regression) gives different model everytime due to difference in domain used (such as health system,face recognition).
Model is formed as a result of training.
Model studies the pattern of data.
Therefore when new data comes, it will recognise the data.
generalisation-performs well on unseen data.

parameters

## STEPS FOR CODE IN GENERAL

load library
load dataset
EDA
split dataset
create model

### LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
seaborn
pickle

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from prettytable import PrettyTable

category_encoders- convert labels into numerical
---
## LINEAR REGRESSION

### TYPES OF LINEAR REGRESSION:

single linear regression
y=mx+c
m and c are paramters.

multiple linear regression
y=c+m1x1+m2x2+....

non linear regression
y=g(x/theta)
g()-model , theta-paramters

### COST FUNCTION()

measures performance of model.
finds error
goal_minimise cost fn
we need to minimise the following fn
j=1/n sigma(predi-yi)^2    (sigma=>i=1 to n)

### REGULARISATION OF LINEAR MODELS

1)ridge (l2 regularisation)
2)lasso (l1 regularisation)
3)elastic net (combines l1 and l2)

overfit-w/o regularisation
good fit- with regularisation
regualarisation helps in getting a good fit model.

ridge expresson = loss fn + regularised term
lasso expresson = loss fn + regularised term
create fit predict evaluate model

try other attributes for car.

## LOGISTIC REGRESSION

non linear
while plotting a graph,we have to bent the line at a particular point.We need to find this point.

### EVALUATION METRICS

check n/b

### TICK MARKS
small indicators along axes of graph along with their labels.

### ILOC
integer location

## DECISION TREE

node-feature/attribute
branch-link

### STEPS FOR DECISION TREE

1)create a root node (calculate entropy ,avg info ,info gain)
  check nb for eqns.
  entropy=0 when either all are positive or all are negative.
  value=0 for entropy is good.
  Higher gain value is good. Attribute with highest gain value is selected as the root node.
  
2)Left node
  Recalculate using eqns for all other attributes again w/o root node.
  Select the next attrbute with the highest gain value and put it as the left node.
  
3)Recalculate once again. Attribute with highest gain value is placed at right node.

4)Final decision tree is completed.

### STEPS FOR GINI

1)calculate gini value for all attributes
2)Attribute with lowest gini value is the root node.
3)Repeat for left node and then right node.

### ISSUES IN DECISION TREE

accuracy first increase then decrease.
overfitting

### CODE FOR DECISION TREE

Set criterion=entropy if we want gain value.
or set criterion=gini.

## SVM

find optimal hyperplane (plane that has max. distance from both the classes used)

### TYPES OF SVM

linear- Data set:linearly sperable.
        divides categories using a simple straight lines.
        Is able to divide data into 2 categories using a straight line.
non linear - Data set:not linearly seperable.
             transforms data into 2d.
           
## ENSEMBLE LEARNING

Combines decisions from multiple models to improve performance.
enhances accuracy
mitigates error in individual models

### TYPES OF ENSEMBLE LEARNING

max voting-highest number of votes
averaging - take avg of predictions.
weighted averaging -models are assigned weights. Higher weight-more importance.
soft voting takes the average value(average voting)
hard voting is majority voting

Bagging

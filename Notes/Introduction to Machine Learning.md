# Introduction to Machine Learning

## Brief Introduction
- The first ML application that really became mainstream: the spam filter. 
- Machine Learning is the science of programming computers so they can learn from data.
- Used to solve complex problems.
- Perform better than traditional approach.
- Data mining- Apply ML onto large data to find patterns
  
## Applications:
- Detecting brain tumors using CNN
- Creating chatbot using NLP
- Fraud Credit card detection using Anomaly detection.
      
## Types of Machine Learning
- Whether or not they are trained with human supervision 
   1. Supervised Learning
   2. Semisupervised Learning
   3. Unsupervised Learning
   4. Reinforcement Learning
- Whether or not they can learn incrementally on the fly
   1. Online Learning
   2. Batch Learning
- Whether they work by simply comparing new data points to known data points, or by detecting patterns in the training data and building a predictive model
   1. Instance-based Learning
   2. Model based Learning

## Supervised Learning
- Training Data contains labels.
- Important algorithms are:
  1. k-Nearest Neighbors
  2. Linear Regression
  3. Logistic Regression
  4. Support Vector Machines (SVMs)
  5. Decision Trees and Random Forests
  6. Neural networks

## Unsupervised Learning
- Training data is unlabeled
- Important algorithms are:
  1. Clustering (detect groups of similar visitors)
     - K-Means  
     - DBSCAN
     - Hierarchical Cluster Analysis-HCA (Subdivides each group into smaller groups)

       Clustering- detect groups of similar visitors
       
  2. Anomaly detection and novelty detection
     - One-class SVM
     - Isolation Forest
    
       Anomaly detection- removing outliers from a dataset
       Novelty detection- detect new instances that look different from all instances in the training set.
       Difference bw Anomaly detection and novelty detection:
       For example, if you have thousands of pictures of dogs, and 1% of these pictures represent Chihuahuas, then a novelty detection algorithm should not treat new 
       pictures of Chihuahuas as novelties. On the other hand, anomaly detection algorithms may consider these dogs as so rare and so different from other dogs that 
       they would likely classify them as anomalies.

  3. Visualization and dimensionality reduction
     - Principal Component Analysis (PCA)
     - Kernel PCA
     - Locally Linear Embedding (LLE)
     - t-Distributed Stochastic Neighbor Embedding (t-SNE)
       
       Visualization- Output 2D or 3D of data that can be plotted
       Dimensionality reduction- simplify the data without losing too much information.
                                merge correlated features into one.
                                better to reduce dimension before using supervised algorithm.
       
  5. Association rule learning
     - Apriori
     - Eclat

     Association rule learning- dig into large amounts of data and discover interesting relations between attributes.

## Semisupervised learning
- Partially labeled data
- Plenty of labeled data and few unlabeled data.
- Combinations of unsupervised and supervised algorithms.

## Reinforcement Learning
- The learning system, called an agent can
  1. observe the environment,
  2. select action using policy
  3. perform actions, and
  4. get rewards in return or penalties in the form of negative rewards
  5. update policy
  6. iterate until optimal policy
 
## Batch Learning
-  incapable of learning incrementally
-  must be trained using all the available data and then runs w/o learning.
-  Called offline learning.
-  Need to train new version of system from scratch.
-  Then stop old system and replace it.
-  Requires a lot of computing resources and time.

## Online learning
- train the system incrementally
- by feeding it data  either individually or in small groups called mini-batches.
- Fast and cheap
- Needs limited computing resources
- Can train systems on huge datasets (out-of-core learning)
- learning rate: how fast they should adapt to changing data
- high learning rate: rapidly adapt to new data, but quickly forget the old data.
- low learning rate:  learn more slowly, but it will also be less sensitive to noise in the new data or to outliers.
- if bad data is fed to the system, the system’s performance will gradually decline.
- To reduce this risk, you need to monitor your system closely and promptly switch learning off.
  
## Instance-based learning
- learns the examples by heart, then
- generalizes to new cases by using a similarity measure to compare them to the learned examples.

## Model-based learning
- build a model of examples and then use that model to make predictions.
- Steps:
  -  Define parameter values by feeding data to algorithm (Model fitting)
  -  Performance measure: utility function (or fitness function) that measures how good your model is,a cost function that measures how bad it is typically used by 
     Linear regression.
  -  Model makes predictions.




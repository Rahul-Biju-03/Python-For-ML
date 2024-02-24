#!/usr/bin/env python
# coding: utf-8

# ## Explainable Computational Mathematics behind Logistic Regression

# ## Key points
# 
# - Logistic Regression is a supervised learning algorithm used for binary classification.
# >e.g. ( True or False, Yes or No, 1 or 0).
# - It can also be used for multiclass classification.
# 
# >but for multiclass classification we have to do something else there are concepts called one vs rest, one vs all. we will see those in upcoming articles.
# 
# ## Case of binary classification

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# creating random data points
x =  np.linspace(-10, 10, 10)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
plt.figure(figsize=(7, 4), dpi=100)
plt.title('X vs class(1 or 0)')
plt.xlabel('X values')
plt.ylabel('class (0 or 1)')
plt.scatter(x, y)
plt.savefig('logistic_regression.jpg')
plt.show()


# >Intuition from the plot:
# 
#  $f(x)=\begin{cases} 0&\quad x<0\\ 1&;\quad x\geq 0\end{cases}$

# ## Linear Regression
# 

# In[2]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x.reshape(-1,1), y)
pred = lr.predict(x.reshape(-1,1))
print(y, pred, sep='\n')
plt.figure(figsize=(7, 4), dpi=100)
plt.title('X vs class(1 or 0)')
plt.xlabel('X values')
plt.ylabel('class (0 or 1)')
plt.scatter(x, y, label="Actual")
plt.plot(x, pred, label="Predicted")
plt.legend(loc='upper left')
plt.savefig('logistic_regression_1.jpg')
plt.show()


# ## Statistical apprach to linear regression

# In[3]:





# In[4]:


import statsmodels.api as sm
mod = sm.OLS(y, x)
res = mod.fit()
print(res.summary())


# So the Mathematical model for the linear regression is:
# 
# $$y=0.682 x+0.029$$

# In[5]:


y_p=0.682*x+0.029
print(y_p)


# ## Disadvantages of Linear Regression over Logistic regression or classification problem:
# 
# - The error rate is very high.
# - very sensitive to outliers.
# - most of the time we will get predicted values that are greater than 1 and less than 0.

# ## Linear regression is sensitive to outliers

# In[6]:


# creating data random points
x1 = np.append(x, 50)
y1 = np.append(y, 1)

lr = LinearRegression()
lr.fit(x1.reshape(-1,1), y1)
predu = lr.predict(x1.reshape(-1,1))

print(y1, predu, sep='\n')

plt.figure(figsize=(7, 4), dpi=100)
plt.title('X vs class(1 or 0)')
plt.xlabel('X values')
plt.ylabel('class (0 or 1)')

plt.scatter(x1, y1, label="Actual")
plt.plot(x1, predu, label="Predicted")
# plt.plot([0, 100, 100], [0.5, 0.5, 0], linestyle='-')

plt.legend()
plt.savefig('logistic_regression_12.jpg')
plt.show()


# ## Intuition behind the logistic regression
# 
# The logistic regression transform a linear predicted line to a sigmoid form:
# $$h_\theta(x)=\dfrac{1}{1+e^{-\theta^TX}}$$
# Where $\theta^T x$ is the linear expression $mx+c$

# In[7]:


def sigmoid(x, theta=1):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# In[8]:


y_pred = sigmoid(x, 1)
y_pred


# In[9]:


y


# In[10]:


pred


# In[11]:


plt.figure(figsize=(7, 4), dpi=100)
plt.ylabel('class (0 or 1)')
plt.scatter(x, y, label="Actual")
plt.scatter(x, y_pred, label=r'Transformed by $\frac{1}{1+e^{-\theta^Tx}}$ ')
plt.plot(x, y_pred, linestyle='-.')
plt.plot(x, pred,label="Regressor",color='b')
plt.legend()
plt.savefig('logistic_regression_12.jpg')
plt.show()


# ## Cost Function:
# >Cost Function is used to check the error between actual and predicted values.
# but we don’t use the MSE function in logistic regression.
# 
# $$J(h_\theta(x))=-\frac{1}{m}\sum \left[y^i\log(h_\theta(x^i))+(1-y^i)\log(1-h_\theta(x^i))\right]$$

# In[12]:


def cost_function(x, y, t): # t= theta value
    # Computes the cost function for all the training samples
    m = len(x)
    total_cost = -(1 / m) * np.sum(
        y * np.log(sigmoid(x, t)) +
        (1 - y) * np.log(1 - sigmoid(x, t)))
    return total_cost


# ## Plot of cost function over $\theta$.

# In[13]:


# ploting graph for diffrent values of m vs cost function
plt.figure(figsize=(10,5))
T = np.linspace(-1, 1,10)
error = []
i= 0
for t in T:

    error.append(cost_function(x,y, t))
    #print(f'for t = {t} error is {error[i]}')
    i+=1
plt.plot(T, error)
plt.scatter(T, error)
plt.ylabel("cost function")
plt.xlabel("t")
plt.title("Error vs t")
plt.savefig('costfunc.jpg')
plt.show()


# ##`python` implementation

# In[14]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x.reshape(-1,1), y)
pred = lr.predict(x.reshape(-1,1))
prob = lr.predict_proba(x.reshape(-1,1))
print(y, np.round(np.array(pred), 2), sep='\n')


# In[15]:


prob


# In[16]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y, pred)
cnf_matrix


# ## Visualizing Confusion Matrix

# In[17]:


import seaborn as sns
import pandas as pd
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[18]:


from scipy.interpolate import make_interp_spline

B_spline_coeff = make_interp_spline(x, pred)
X_Final = np.linspace(x.min(), x.max(), 500)
Y_Final = B_spline_coeff(X_Final)


# In[19]:


plt.figure(figsize=(7, 4), dpi=100)
plt.title('X vs class(1 or 0)')
plt.xlabel('X values')
plt.ylabel('class (0 or 1)')

plt.scatter(x, y, label="actual")
plt.plot(x, pred, label="predicted", color='red')
plt.plot(X_Final, Y_Final, label="predicted", color='orange')
plt.legend(loc='upper left')
plt.savefig('logistic_regression.jpg')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Notebook Imports
# 

# In[207]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# ## Gather Data
# 
# [Source: Data Source](https://archive.ics.uci.edu/ml/machine-learning-databases/00374/)

# In[138]:


df=pd.read_csv("energydata_complete.csv")


# ## Data exploration with Pandas dataframes

# In[3]:


df.head() # The top rows look like this:


# In[4]:


df.tail() #Rows at the bottom look like this:


#  ## Cleaning data - check for missing values

# In[5]:


df.info()


# ## Visualizing Data - Histograms, Distributions and Bar Charts

# In[13]:


plt.figure(figsize=(10,6))
plt.hist(df['Appliances'], bins=50, color='#2196f3')
plt.xlabel('Energy use in Wh')
plt.ylabel('Count of Appliances')


# In[17]:


plt.figure(figsize=(10,6))
sns.distplot(df['Appliances'], bins=50)


# In[19]:


plt.figure(figsize=(10,6))
plt.hist(df['lights'], color='#FF5722')
plt.xlabel('Energy use of light fixture in Wh')
plt.ylabel('Count of Lights') 


# ## Descriptive Statistics

# In[20]:


df.describe()


# ## Correlation
# 
# ### $$ \rho _{XY} = corr(X,Y)$$
# ### $$ -1.0 \leq \rho _{XY} \leq +1.0 $$

# In[21]:


df.corr()


# In[22]:


mask = np.zeros_like(df.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True


# In[32]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr(), annot=True, annot_kws={"size":10})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[42]:


get_ipython().run_cell_magic('time', '', '\nsns.pairplot(df)')


# ## Training & Test Dataset Split

# In[190]:


df.drop(['date', 'lights'], axis=1, inplace=True)


# In[191]:


df.head()


# ### Normalizing dataset

# In[192]:


scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features = normalised_df.drop(columns=['Appliances'])
target = normalised_df['Appliances']


# In[194]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


# ## Multivariable Regression

# In[196]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[197]:


y_pred = reg.predict(X_test)


# In[206]:


print('Mean Absolute Error:', round(mean_absolute_error(y_test, y_pred), 3))
print('R-Squared:',round(r2_score(y_test, y_pred), 3))
print('Mean Squared Error:', round(mean_squared_error(y_test, y_pred), 3))
print('Root Mean Squared Error:', np.sqrt(round(mean_squared_error(y_test, y_pred), 3)))
print('Residual Sum of Squares:', np.sum(np.square(y_test - y_pred)))


# ### Regularization

# In[208]:


## LASSO
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

## RIDGE
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(X_train, y_train)


# In[210]:


def get_weights_df(model, feat, col_name):
  #this function returns the weight of every feature
  weights = pd.Series(model.coef_, feat.columns).sort_values()
  weights_df = pd.DataFrame(weights).reset_index()
  weights_df.columns = ['Features', col_name]
  weights_df[col_name].round(3)
  return weights_df

linear_model_weights = get_weights_df(reg, X_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, X_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, X_train, 'Lasso_weight')

final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='Features')


# In[211]:


final_weights


# ## Quiz Linear Regression

# In[218]:


X1 = pd.DataFrame(X_train['T2'], columns=['T2'])
y1 = pd.DataFrame(X_train['T6'], columns=['T6'])

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)


# In[220]:


reg.fit(X1_train, y1_train)


# In[221]:


y1_pred = reg.predict(X1_test)


# In[229]:


print('Mean Absolute Error:', round(mean_absolute_error(y1_test, y1_pred), 2))
print('R-Squared:',round(r2_score(y1_test, y1_pred), 2))
print('Mean Squared Error:', round(mean_squared_error(y1_test, y1_pred), 3))
print('Root Mean Squared Error:', np.sqrt(round(mean_squared_error(y1_test, y1_pred), 3)))
print('Residual Sum of Squares:', round(np.sum(np.square(y1_test - y1_pred)), 2))
print('Coefficient of Determination', reg.score(y1_test, y1_pred))


# In[231]:


## LASSO
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X1_train, y1_train)

## RIDGE
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(X1_train, y1_train)


# In[232]:


## LASSO
y_pred_ridge = ridge_reg.predict(X1_test)


# In[234]:


print('RIDGE Root Mean Squared Error:', np.sqrt(round(mean_squared_error(y1_test, y_pred_ridge), 3)))


# In[248]:


## LASSO NON-ZERO WEIGHT

non_zero = pd.DataFrame(final_weights['Lasso_weight'], columns=['Lasso_weight'])
non_zero


# In[252]:


## NEW RMSE WITH LASSO REGRESSION
lasso_regg = Lasso(alpha=0.001)
lasso_regg.fit(X_train, y_train)
## LASSO
y_pred_lasso = lasso_regg.predict(X_test)
print('LASSO Root Mean Squared Error:', np.sqrt(round(mean_squared_error(y_test, y_pred_lasso), 3)))


# In[ ]:





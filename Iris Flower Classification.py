#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier


# In[2]:


df = pd.read_csv('D:/Trishala/CodSoft/Iris/IRIS.csv')
df.rename(columns={'sepal_length':'sepallength','sepal_width':'sepalwidth','petal_length':'petallength','petal_width':'petalwidth'},inplace=True)
df.head()


# In[3]:


z = df.species.replace({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
df.insert(4,'nspecies',z)
df['nspecies'] = df['nspecies'].astype(float)
df.head()


# In[4]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Blues')


# In[5]:


sns.pairplot(df, hue='species')


# In[6]:


plt.figure(figsize=(10,10))
sns.countplot(x='nspecies', data=df, hue='sepallength')


# In[7]:


plt.figure(figsize=(10,10))
sns.countplot(x='nspecies', data=df, hue='sepalwidth')


# In[8]:


plt.figure(figsize=(15,15))
sns.countplot(x='nspecies', data=df, hue='petallength')


# In[9]:


plt.figure(figsize=(10,10))
sns.countplot(x='nspecies', data=df, hue='petalwidth')


# In[10]:


x = np.array(df.drop(['species','nspecies'], axis=1))
y = np.array(df.nspecies)


# In[11]:


x_test, x_train, y_test, y_train = train_test_split(x,y,test_size=0.05, shuffle=True)


# In[12]:


p = df.iloc[:,0:4]
q = df.iloc[:,4]
feat = ExtraTreesClassifier()
feat.fit(p,q)
feat.feature_importances_
feat_imp = pd.Series(feat.feature_importances_, index=p.columns)
feat_imp.nlargest(4).plot(kind='barh')


# In[13]:


regressor = DecisionTreeRegressor()


# In[14]:


regressor.fit(x_train, y_train)


# In[15]:


y_pred = regressor.predict(x_test)
final = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
final.head()


# In[16]:


confusion_matrix(y_test, y_pred)


# In[17]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid()
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)], color='red')
plt.title('Actual V/S Predicted')


# In[18]:


r2_score(y_test, y_pred)


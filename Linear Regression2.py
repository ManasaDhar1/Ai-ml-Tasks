#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1=pd.read_csv("NewspaperData.csv")
data1.head()


# # EDA

# In[3]:


data1.info()


# In[5]:


data1.isnull().sum()


# In[6]:


data1.describe()


# In[7]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[8]:


sns.histplot(data1["daily"],kde=True,stat='density',)
plt.show()


# In[9]:


sns.histplot(data1["sunday"],kde=True,stat='density',)
plt.show()


# In[11]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# ### observations
#  - there are no missing values
#  - the daily column values appears to be right skewed
#  - the sunday column values also appear to be right-sckwed
#  - there are two outliers in both daily and sunday columns as observed from the above plots
#  - 

# In[14]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) +100)
plt.ylim(0, max(y) +100)
plt.show()


# In[15]:


data1["daily"].corr(data1["sunday"])


# In[16]:


data1[["daily","sunday"]].corr()


# In[18]:


data1.corr(numeric_only=True)


# ### observations on Correlation strength
#  - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
#  - The correlation is strong and positive with pearson's correaltion coefficient of 0.958154

# ## Fit a linear regression model

# In[20]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data=data1).fit()


# In[21]:


model1.summary()


# In[ ]:





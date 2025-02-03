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


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[7]:


sns.histplot(data1["daily"],kde=True,stat='density',)
plt.show()


# In[8]:


sns.histplot(data1["sunday"],kde=True,stat='density',)
plt.show()


# In[9]:


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

# In[10]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) +100)
plt.ylim(0, max(y) +100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[13]:


data1.corr(numeric_only=True)


# ### observations on Correlation strength
#  - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
#  - The correlation is strong and positive with pearson's correaltion coefficient of 0.958154

# ## Fit a linear regression model

# In[14]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data=data1).fit()


# In[15]:


model1.summary()


# #### Observations from model summary
# - the probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# - therefore the intercept coefficient may not be that much significant in prediction
# - however the p-value for 'daily' (beta_1) is 0.00 < 0.05
# - therefore the beta_1 coefficient is highly significant and is contributint to prediction

# # Interpretation
#  - rsqaured = 1 :perfect fit(all variance exolained)
#  - rsquared = 0 : model doesnot explain any variance
#  - rsquare close to 1:good model fit
#  - rsquare close to 0:poor model fit

# In[16]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = 'm',marker = 'o',s=30)
b0=13.84
b1=1.33
y_hat = b0+b1*x
plt.plot(x,y_hat,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[17]:


model1.params


# In[23]:


print(f'model t-values:\n{model1.tvalues}\n--------------------------\nmodel p-values: \n{model1.pvalues}')


# In[21]:


(model1.rsquared,model1.rsquared_adj)


# #### Predict for new data point

# In[24]:


newdata = pd.Series([200,300,1500])


# In[28]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[27]:


model1.predict(data_pred)


# In[34]:


#predict on all given training data
pred = model1.predict(data1['daily'])
pred


# In[33]:


#Add predicated values as a column in data1
data1["Y_hat"]=pred
data1


# In[32]:


#compute the error values (residuals)nand add as another column
data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[ ]:





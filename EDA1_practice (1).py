#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


#printing the information 
data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"],axis=1)
data1


# In[6]:


data.info()


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[9]:


data1.rename({'Solar.R':'Solar'},axis=1,inplace = True)
data1


# #impute the missing values

# In[10]:


data1.isnull().sum()


# In[11]:


cols = data1.columns
colors = ['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[12]:


median_ozone = data1['Ozone'].median()
mean_ozone = data1["Ozone"].mean()
print("Median of ozone:",median_ozone)
print("mean of ozone:",mean_ozone)


# In[13]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[14]:


median_solar = data1['Solar'].median()
mean_solar = data1["Solar"].mean()
print("Median of solar:",median_solar)
print("mean of solar:",mean_solar)


# In[15]:


data1['Solar']=data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[16]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[21]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[22]:


print(data1["Day"].value_counts())
mode_day = data1["Day"].mode()[0]
print(mode_day)


# In[23]:


data1["Day"] = data1["Day"].fillna(mode_day)
data1.isnull().sum()


# In[29]:


#detection of outliers using histogram and boxplots
fig,axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Ozone"],ax = axes[0],color='skyblue',width=0.5,orient = 'h')
axes[0].set_title("BoxPlot")
axes[0].set_xlabel("Ozone levels")
sns.histplot(data1["Ozone"], kde=True,ax = axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# #observations
# #the ozone columns has extreme values beyond 81 as seen from box plot
# #the same is confirmed from the below right-skewed histogram

# In[32]:


fig,axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Solar"],ax = axes[0],color='yellow',width=0.5,orient = 'h')
axes[0].set_title("BoxPlot")
axes[0].set_xlabel("Solar levels")
sns.histplot(data1["Solar"], kde=True,ax = axes[1],color='pink',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# .No outliers are observed
# .It is lightly left skewed

# In[ ]:





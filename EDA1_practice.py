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


# In[6]:


data1 = data.drop(['Unnamed: 0',"Temp C"],axis=1)
data1


# In[7]:


data.info()


# In[8]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[9]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[ ]:





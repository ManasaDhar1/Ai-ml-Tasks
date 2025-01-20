#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("universities.csv")
df


# In[5]:


df.sort_values(by="GradRate",ascending=True)


# In[6]:


df.sort_values(by="GradRate",ascending=True)


# In[8]:


df[df["GradRate"]>=90]


# In[10]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[12]:


sal = pd.read_csv("Salaries.csv")
sal


# In[14]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[ ]:





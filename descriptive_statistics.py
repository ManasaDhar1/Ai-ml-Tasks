#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("universities.csv")
df


# In[3]:


np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[12]:


np.mean(df["GradRate"])


# In[8]:


np.var(df["SFRatio"])


# In[9]:


df.describe()


# In[ ]:





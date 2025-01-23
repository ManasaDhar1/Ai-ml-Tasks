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


# In[5]:


np.mean(df["GradRate"])


# In[6]:


np.var(df["SFRatio"])


# In[7]:


df.describe()


# In[8]:


#visualize the gradrate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[13]:


s = [20,15,10,25,30,28,40,45,60]
scores = pd.Series(s)
scores


# In[14]:


plt.boxplot(scores, vert=False)


# In[15]:


plt.boxplot(scores)


# In[17]:


s = [20,15,10,25,30,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[18]:


plt.boxplot(scores, vert=False)


# In[22]:


s = [700,710,657,789,234,78,69]
scores = pd.Series(s)
scores


# In[23]:


plt.boxplot(scores, vert=False)


# In[26]:


#identification of outliers from universities dataset
df = pd.read_csv("universities.csv")
df


# In[29]:


plt.title("Graduation Rate")
plt.boxplot(df["GradRate"])


# In[30]:


plt.title("SAT")
plt.boxplot(df["SAT"])


# In[31]:


plt.title("Top10")
plt.boxplot(df["Top10"])


# In[34]:


plt.title("Accept")
plt.boxplot(df["Accept"],vert = False)


# In[33]:


plt.title("SFRatio")
plt.boxplot(df["SFRatio"],vert = False)


# In[35]:


plt.title("Boxplot for Expenses")
plt.boxplot(df["Expenses"],vert=False)


# In[ ]:





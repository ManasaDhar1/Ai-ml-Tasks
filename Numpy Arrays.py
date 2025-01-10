#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing numpy library
import numpy as np


# In[3]:


dir(np)


# In[4]:


x = np.array([45,67,57,60])
print(x)
print(type(x))
print(x.dtype)


# In[5]:


x = np.array([45,67,57,9.6])
print(x)
print(type(x))
print(x.dtype)


# In[6]:


x = np.array(["A",67,57,9.6])
print(x)
print(type(x))
print(x.dtype)


# In[7]:


a2 = np.array([[20,40],[30,60]])
print(a2)
print(type(a2))
print(a2.shape)


# In[9]:


#Reshaping an array
a = np.array([10,20,30,40])
b = a.reshape(2,2)
print(b)
print(b.shape)


# In[10]:


a = np.array([10,20,30,40])
b = a.reshape(4,1)
print(b)
print(b.shape)


# In[13]:


c = np.arange(3,10)
print(c)
type(c)


# In[15]:


d = np.array([1.3457,3.4567,4.36728,78.9287])
print(d)
np.around(d,1)


# In[18]:


d = np.array([1.3457,3.4567,4.36728,78.9287])
print(d)
print(np.around(np.sqrt(d),2))


# In[20]:


a1 = np.array([[3,4,5,8],[7,2,8,np.NaN]])
print(a1)
a1.dtype


# In[21]:


a1_copy1 = a1.astype(str)
print(a1_copy1)
a1_copy1.dtype


# In[25]:


a1_copy1 = a1.astype(bool)
print(a1_copy1)
a1_copy1.dtype


# In[27]:


a2 = np.array([[3,4,6],[7,9,10],[4,6,12]])
a2


# In[28]:


a2.sum(axis = 1)


# In[29]:


a2.sum(axis = 0)


# In[31]:


a2.mean(axis = 0)

a2.mean(axis = 1)
# In[33]:


a3 = np.array([[3,4,5],[7,2,8],[9,1,6]])
a3


# In[34]:


np.fill_diagonal(a3,0)
a3


# In[35]:


np.fill_diagonal(a3,1)
a3


# In[37]:


A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = np.matmul(A,B)
C


# In[38]:


print(A.T)
print(B.T)


# In[39]:


a4 = np.array([[3,4,5],[7,2,8],[9,1,6],[10,9,18]])
a4


# In[41]:


print(a4[2][0])


# In[42]:


print(a4[2][2])


# In[46]:


print(a4[1:3,0:2])


# In[49]:


print(a4[1:3,0:])


# In[50]:


#accessing max value and its index
a3 = np.array([[3,4,5],[7,2,8],[9,1,6]])
a3


# In[55]:


#print the max value element
print(np.amax(a3,axis = 1))


# In[56]:


print(np.amax(a3,axis = 0))


# In[57]:


#print the index position of max element
print(np.argmax(a3,axis = 1))
print(np.argmax(a3,axis = 0))


# In[ ]:





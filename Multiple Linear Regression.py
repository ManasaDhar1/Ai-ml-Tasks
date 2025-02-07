#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# #### Assumptions:
# 1.**Linearity:** The relationship b/w the predictors(x) and the response(y) is linear
# 2.**Independence:** Observations are independent of each other
# 3.**Normal distribution of errors:** The residuals of the model are bormally distributed
# 4.**Homoscedasticity:** the reiduals (Y-Y_hat ) exhibit constant variance at all levels of the predictor
# 5.**No multicollinearity:** The independent variables should not be too highly correlated with each other
# - Violations of these assumptions may lead to efficiency in the regression parameters and unreliable predictions

# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### Descriptions of columns
# - **MPG:** milege of the car(mile per gallon) (this is y-colums to be predicted)
# - **HP:** horse power of the car(x1 column)
# - **VOL:** Volume of the car(size)(X2 column)
# - **SP :** Top speed of the car(Miles per hour)(x3 column)
# - **WT:** Weight of the car(Pounds)(x4 column)

# ### EDA

# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# ### Observations
# - there are no missing values
# - there are 81 observations (81 different cars data)
# - the data types of the columns are also relevant and valid

# In[6]:


fig, (ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw = {"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
plt.tight_layout()
plt.show()


# In[7]:


fig, (ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw = {"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30,kde=True,stat="density")
plt.tight_layout()
plt.show()


# In[8]:


fig, (ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw = {"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='SP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30,kde=True,stat="density")
plt.tight_layout()
plt.show()


# In[9]:


fig, (ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw = {"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='WT',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins=30,kde=True,stat="density")
plt.tight_layout()
plt.show()


# #### Observations fom boxplot and histograms
# - there are some extreme values (outliers) observed in towards theright tail of SP and HP distributions
# - in vol and wt columns a few outiers are observed in both tails of their observations
# - the extreme values of cars data may have come from the specially designed nature of cars
# - As this is multi-dimensional data,the outliers with respect to spatial dimensions may have to be considered while building the regression model
#   

# In[10]:


cars[cars.duplicated()]


# In[11]:


cars.corr()


# ### Observations
# - Highest correlation strength is b/w wt vs vol
# - next highest correlation strength is b/w hp vs sp
# - the lowest correlation strength is b/w wt vs sp

# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# #### Preparing a  preliminary model considering all x columns

# In[13]:


#Build model
model1 = smf.ols('MPG~WT+SP+VOL+HP',data=cars).fit()


# In[14]:


model1.summary()


# #### Observations from model summary
# - The R-squared and adjusted R-Squared values are good and about 75% of variability in y is explained by x columns
# - The probability values with respect to F-Statistic is close to zero,indicating that all or some of X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored

# In[15]:


df1 = pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[16]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[17]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


# ## checking for multicollinearity amon x-columns using vif method

# In[18]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### observations for vif values:
# - The ideal range of vif values shall be b/w 0 and 10 however slightlt higher values can be tolerated
# - as seen from the very high vif values for vol and wt it is clear that they are prone to multicollinearity problem
# - hence it is decided to drop one of the columns(either vol or wt) to overcome the multicollinearity
# - it is decided to drop wt and retain vol column in further models

# In[19]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[24]:


model2=smf.ols('MPG~HP+VOL+SP',data=cars1).fit()


# In[25]:


model2.summary()


# #### Performance metrics for model2

# In[26]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[27]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"]=pred_y2
df2.head()


# In[30]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


# #### observations
# - the adjusted r-squared value improved slightly to 0.76
# - all the p-values for model parameters are less than 50% hence they are significant
# - therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# - there is no improvement in mse value

# In[ ]:





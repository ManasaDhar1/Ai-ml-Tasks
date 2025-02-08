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


# In[20]:


model2=smf.ols('MPG~HP+VOL+SP',data=cars1).fit()


# In[21]:


model2.summary()


# #### Performance metrics for model2

# In[22]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[23]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"]=pred_y2
df2.head()


# In[24]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


# #### observations
# - the adjusted r-squared value improved slightly to 0.76
# - all the p-values for model parameters are less than 50% hence they are significant
# - therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# - there is no improvement in mse value

# #### Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[26]:


#Define variables and assign values
k=3
n=81
levarage_cutoff = 3*((k+1)/n)
levarage_cutoff


# In[27]:


cars1.shape


# In[29]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=.05)
y=[i for i in range(-2,8)]
x=[levarage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Observations
# - from the above plot,it is evident that datapoints 65,70,76,78,79,80 are the influencers
# - as their H Levrage values are higher and size is higher
#    

# In[30]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[31]:


cars2


# ### Build model3 on cars2 dataset
# 

# In[32]:


model3=smf.ols('MPG~VOL+SP+HP',data=cars2).fit()


# In[33]:


model3.summary()


# In[34]:


df3=pd.DataFrame()
df3["actual_y3"]=cars2["MPG"]
df3.head()


# In[41]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"]=pred_y3
df3.head()


# In[42]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"],df3["pred_y3"])
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[45]:


df3["residuals"]=df3["actual_y3"]-df3["pred_y3"]
df3


# In[47]:


plt.scatter(df3["pred_y3"],df3["residuals"])


# In[48]:


import statsmodels.api as sm
sm.qqplot(df3["residuals"],line='45',fit=True)
plt.show


# In[ ]:





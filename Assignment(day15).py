#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Load dataset
file_path = "Day_15_Healthcare_Data.csv"
df = pd.read_csv(file_path)

# Display initial information
display(df.info())
display(df.head())

# Identify missing values
missing_values = df.isna().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
display(missing_data)

# Visualizing missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

# Imputation Methods

# Mean/Median/Mode Imputation
numeric_cols = ['Blood_Pressure', 'Cholesterol']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Display results after imputation
display(df.info())

# Visualizing the impact of imputation
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot After Imputation")
plt.show()

# Save cleaned dataset
df.to_csv("Cleaned_Healthcare_Data.csv", index=False)


# In[ ]:





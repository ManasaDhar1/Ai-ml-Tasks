#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import os

# Load the CSV file
file_path = "C:/Users/Dharma Manasa/Downloads/Day_8_sales_data.csv"  # Adjusted to current directory
df = pd.read_csv(file_path)

# Filter rows where Sales > 1000
sales_above_1000 = df[df['Sales'] > 1000]

# Filter rows for the "East" region
east_region_sales = df[df['Region'] == 'East']

# Add a new column 'Profit_Per_Unit'
df['Profit_Per_Unit'] = df['Profit'] / df['Quantity']

# Add a new column 'High_Sales' based on the sales threshold
df['High_Sales'] = df['Sales'].apply(lambda x: 'Yes' if x > 1000 else 'No')

# Save the filtered datasets to CSV files in the current directory
sales_above_1000.to_csv('sales_above_1000.csv', index=False)
east_region_sales.to_csv('east_region_sales.csv', index=False)
df.to_csv('updated_sales_data.csv', index=False)

# Display the first few rows to confirm changes
df.head()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


file_path = 'Day_9_banking_data.csv'  
banking_data = pd.read_csv(file_path)


print("First 5 rows of the dataset:")
print(banking_data.head())

print("\nBasic statistics of numerical columns:")
print(banking_data.describe())


print("\nMissing values in each column:")
print(banking_data.isnull().sum())


# In[4]:


import pandas as pd


file_path = 'Day_9_banking_data.csv'  
banking_data = pd.read_csv(file_path)


account_type_group = banking_data.groupby('Account_Type').agg(
    Total_Transaction_Amount=('Transaction_Amount', 'sum'),
    Average_Account_Balance=('Account_Balance', 'mean')
)


branch_group = banking_data.groupby('Branch').agg(
    Total_Transactions=('Transaction_Amount', 'count'),
    Average_Transaction_Amount=('Transaction_Amount', 'mean')
)


print("Grouped by Account_Type:")
print(account_type_group)

print("\nGrouped by Branch:")
print(branch_group)


# In[7]:


import pandas as pd


file_path = 'Day_9_banking_data.csv'
banking_data = pd.read_csv(file_path)


filtered_amount = banking_data[banking_data['Transaction_Amount'] > 2000]


filtered_loan_payment = banking_data[
    (banking_data['Transaction_Type'] == "Loan Payment") &
    (banking_data['Account_Balance'] > 5000)
]


uptown_transactions = banking_data[banking_data['Branch'] == "Uptown"]
banking_data['Transaction_Fee'] = banking_data['Transaction_Amount'] * 0.02


banking_data['Balance_Status'] = banking_data['Account_Balance'].apply(
    lambda balance: "High Balance" if balance > 5000 else "Low Balance"
)

# Print the results
print("Filtered rows (Transaction_Amount > 2000):")
print(filtered_amount)

print("\nFiltered rows (Transaction_Type == 'Loan Payment' and Account_Balance > 5000):")
print(filtered_loan_payment)

print("\nTransactions made in the 'Uptown' branch:")
print(uptown_transactions)

print("\nUpdated dataset with Transaction_Fee and Balance_Status:")
print(banking_data.head())


# In[ ]:





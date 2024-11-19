#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = {
    'customer_id': range(1, 21),
    'total_transactions': [15, 20, 35, 40, 10, 12, 18, 25, 30, 28, 50, 45, 60, 20, 22, 18, 14, 38, 41, 55],
    'average_transaction_value': [50, 60, 55, 70, 40, 45, 48, 65, 68, 60, 90, 85, 95, 50, 52, 47, 43, 75, 80, 100],
    'tenure_months': [12, 24, 36, 48, 6, 8, 15, 30, 40, 35, 60, 50, 70, 20, 22, 18, 16, 45, 50, 65],
    'churn_probability': [0.1, 0.15, 0.05, 0.03, 0.4, 0.35, 0.2, 0.08, 0.05, 0.1, 0.02, 0.05, 0.01, 0.25, 0.22, 0.28, 0.3, 0.04, 0.03, 0.01],
    'clv': [7500, 12000, 19250, 25200, 2400, 3000, 5400, 16250, 20400, 19200, 45000, 38250, 57000, 12500, 13000, 5600, 4800, 28500, 31500, 60000]
}

df = pd.DataFrame(data)


# In[3]:


print(df)


# In[4]:


X = df[['total_transactions', 'average_transaction_value', 'tenure_months', 'churn_probability']]
y = df['clv']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# In[7]:


y_pred = model.predict(X_test)


# In[8]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[9]:


print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# In[10]:


new_customer = [[25, 70, 18, 0.12]]  # Example: 25 transactions, $70 avg, 18 months tenure, 12% churn probability
predicted_clv = model.predict(new_customer)
print("Predicted CLV for new customer:", predicted_clv[0])


# In[ ]:





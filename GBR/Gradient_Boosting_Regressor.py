#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[13]:


#loading dataset
data = pd.read_csv('House_Price_Estimation/Dataset/Processed_Data/numerical_dataset.csv')
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']


#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing model 
gb_reg = GradientBoostingRegressor()

#defining hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'learning_rate': [0.1, 0.05],  # Step size shrinkage used in update to prevent overfitting
    'max_depth': [3, 4],  # Maximum depth of the individual trees
}

#performing GridSearchCV for tuning hyperparameter 
grid_search = GridSearchCV(estimator=gb_reg, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', verbose=1)

#fitting the model on the entire training data with tuned hyperparameter 
grid_search.fit(X_train, y_train)

#getting the best hyperparameters
best_params = grid_search.best_params_

#using the best hyperparameters to create the final model
best_gb_reg = GradientBoostingRegressor(**best_params)
best_gb_reg.fit(X_train, y_train)


# In[18]:


#### predicting on the test set using the final model
y_pred = best_gb_reg.predict(X_test)

#calculating RMSE (Root Mean Squared Error) on test set
mse = (mean_squared_error(y_test, y_pred))
print(f"Mean Squared Error: {mse}")

#calculating R Squared Error (R2 Score)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2}")


# In[ ]:





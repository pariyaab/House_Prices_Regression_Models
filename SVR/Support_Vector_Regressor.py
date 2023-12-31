#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


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

#initializing model 
svr = SVR()

#defining hyperparameters for tuning
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

#performing GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(svr, parameters, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

#selecting best parameters and estimator from GridSearchCV
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

#fiting the best estimator on the training data
best_estimator.fit(X_train, y_train)


# In[6]:


#predicting on the test set
y_pred = best_estimator.predict(X_test)

#calculating RMSE (Root Mean Squared Error)
mse = (mean_squared_error(y_test, y_pred))
print(f"Mean Squared Error: {mse}")

#calculating R Squared Error (R2 Score)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2}")


# In[ ]:





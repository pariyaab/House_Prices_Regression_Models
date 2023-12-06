import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Step 1: Read selected features file
selected_features_df = pd.read_csv("./Data/Processed_Data/new_selected_features.csv")
selected_features = selected_features_df['Feature']

# Step 2: Read the original training and test sets
train_df = pd.read_csv("./Data/Processed_Data/new_train_data.csv")
test_df = pd.read_csv("./Data/Processed_Data/new_test_data.csv")

# Step 3: Extract corresponding features and target variable from the original training set
X_train = train_df[selected_features]
y_train = train_df['SalePrice']

start_time = time.time()

# Define the random grid
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Step 4: Initialize Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Step 5: Use RandomizedSearchCV to search over the random grid
rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=random_grid,
                               n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the random search model using the original training set
rf_random.fit(X_train, y_train)

end_time = time.time()

# Calculate the time elapsed
time_elapsed = end_time - start_time
print(f"Time Elapsed: {time_elapsed} seconds")

# Step 6: Get the best parameters
best_rf_params = rf_random.best_params_

# Step 7: Print the best hyperparameters found
print("\nBest Hyperparameters:", best_rf_params)


# Output:
# Fitting 5 folds for each of 100 candidates, totalling 500 fits
# Time Elapsed: 1187.664225101471 seconds

# Best Hyperparameters: {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False}
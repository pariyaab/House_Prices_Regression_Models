import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read the training and test datasets
train_df = pd.read_csv("./Data/Processed_Data/new_train_data.csv")
train_df = train_df.dropna()

test_df = pd.read_csv("./Data/Processed_Data/new_test_data.csv")
test_df = test_df.dropna()

# Read the selected features
selected_features_df = pd.read_csv("./Data/Processed_Data/new_selected_features.csv")
selected_features = selected_features_df['Feature']

# Extract features and target variable from the training set
X_train = train_df[selected_features]
y_train = train_df['SalePrice']

# Extract features from the test set
X_test = test_df[selected_features]

# Use the hyperparameters obtained from RandomizedSearchCV
best_rf_params = {
    'n_estimators': 600,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_depth': 70,
    'bootstrap': False
}

# Train the Random Forest model with the best hyperparameters
rf_model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_test_predictions = rf_model.predict(X_test)

actual_values = test_df['SalePrice']

# Evaluate the model performance on the test set
rf_test_mse = mean_squared_error(actual_values, rf_test_predictions)
rf_test_mae = mean_absolute_error(actual_values, rf_test_predictions)
rf_test_r2 = r2_score(actual_values, rf_test_predictions)

# Print the evaluation metrics on the test set
print("\nRandom Forest Regression Results on Test Set:")
print("Best Hyperparameters:", best_rf_params)
print("Mean Squared Error:", rf_test_mse)
print("Mean Absolute Error:", rf_test_mae)
print("Root Mean Squared Error:", rf_test_mse**0.5)
print("R-squared (R2):", rf_test_r2)



# # Random Forest Regression Results on Test Set:
# Best Hyperparameters: {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False}

# Mean Squared Error: 0.003063647429941832
# Mean Absolute Error: 0.039839530244678255
# Root Mean Squared Error: 0.05535022520226844
# R-squared (R2): 0.3966691728877939
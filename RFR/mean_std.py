import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load selected features and data
selected_features_df = pd.read_csv("../Data/Processed_Data/new_selected_features.csv")
selected_features = selected_features_df['Feature']
selected_features = list(selected_features)
selected_features.append("SalePrice")

df = pd.read_csv("../Data/Processed_Data/new_numerical_data.csv")
df = df[selected_features]

df_split = np.array_split(df, 10)

# Use the hyperparameters obtained from RandomizedSearchCV
best_rf_params = {
    'n_estimators': 600,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_depth': 70,
    'bootstrap': False
}

# Initialize variables for best hyperparameters and its corresponding error
best_hyperparameters_rf = None
min_error_rf = float('inf')

# Hyperparameter tuning over different n_estimators values
for _ in range(10):  # Repeat 10 times for better stability

    cv_errors_rf = []

    for iteration in range(10):
        train_indices = [(i + iteration) % 10 for i in range(8)]
        val_index = (iteration + 8) % 10
        test_index = (iteration + 9) % 10

        train = pd.concat([df_split[i] for i in train_indices])
        val = df_split[val_index]
        test = df_split[test_index]

        # Separate features and target variable
        X_train_rf = train.drop(columns=['SalePrice'])
        y_train_rf = train['SalePrice']
        X_val_rf = val.drop(columns=['SalePrice'])
        y_val_rf = val['SalePrice']

        # Initialize RandomForestRegressor with the best hyperparameters
        rf_model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)

        # Fit the model and predict on the validation set for RandomForestRegressor
        rf_model.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf_model.predict(X_val_rf)
        mse_rf = mean_squared_error(y_val_rf, y_pred_rf)
        cv_errors_rf.append(mse_rf)

    average_rf = sum(cv_errors_rf) / len(cv_errors_rf)

    # Update the best hyperparameters if the current one has lower error for RandomForestRegressor
    if average_rf < min_error_rf:
        min_error_rf = average_rf
        best_hyperparameters_rf = best_rf_params

# Report the best hyperparameters values for RandomForestRegressor
print("\nBest Hyperparameters values for RandomForestRegressor:", best_hyperparameters_rf)

# Perform test set evaluation for the best hyperparameters
rf_model = RandomForestRegressor(**best_hyperparameters_rf, random_state=42, n_jobs=-1)

final_test_errors_rf = []

for iteration in range(10):
    train_indices = [(i + iteration) % 10 for i in range(8)]
    val_index = (iteration + 8) % 10
    test_index = (iteration + 9) % 10

    train = pd.concat([df_split[i] for i in train_indices])
    val = df_split[val_index]
    test = df_split[test_index]

    X_train_rf = train.drop(columns=['SalePrice'])
    y_train_rf = train['SalePrice']
    X_test_rf = test.drop(columns=['SalePrice'])
    y_test_rf = test['SalePrice']

    # Fit the model and predict on the test set for RandomForestRegressor
    rf_model.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf_model.predict(X_test_rf)

    test_score_rf = mean_squared_error(y_test_rf, y_pred_rf)
    final_test_errors_rf.append(test_score_rf)

# Calculate mean error and standard deviation for the best hyperparameters for RandomForestRegressor
mean_error_rf = np.mean(final_test_errors_rf)
std_deviation_rf = np.std(final_test_errors_rf)

# Report the results for RandomForestRegressor
print(f"\nResults for Best Hyperparameters for RandomForestRegressor:")
print(f"Mean Error: {mean_error_rf}")
print(f"Standard Deviation: {std_deviation_rf}")


# # Output:
# Best Hyperparameters values for RandomForestRegressor: {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False}

# Results for Best Hyperparameters for RandomForestRegressor:
# Mean Error: 0.004719699840950836
# Standard Deviation: 0.0013792468658811263
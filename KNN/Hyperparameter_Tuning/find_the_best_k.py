import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# Step 1: Read selected features file
selected_features_df = pd.read_csv("../../Data/Processed_Data/new_selected_features.csv")
selected_features = selected_features_df['Feature']

# Read the validation set
val_df = pd.read_csv("../../Data/Processed_Data/new_val_data.csv")
val_df = val_df.dropna()

# Extract corresponding features from the validation set
X_val_selected = val_df[selected_features]
y_val = val_df['SalePrice']

# Perform cross-validation to find the best K
k_values = list(range(1, 51))
cv_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_val_selected, y_val, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-np.mean(scores))

# Find the best K
best_k = k_values[np.argmin(cv_scores)]

# Plotting
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.plot(k_values, cv_scores, marker='o')
plt.title('Cross-Validation Scores for Different K values')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Negative Mean Squared Error')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K: {best_k}')
plt.legend()
plt.savefig("best_k.png")

print(f"Best K for KNN: {best_k}")

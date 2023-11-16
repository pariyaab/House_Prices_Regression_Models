import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import numpy as np

validation_data = pd.read_csv('../../Data/Processed_Data/new_val_data.csv')

y_val_augmented = validation_data['SalePrice']
X_val_augmented = validation_data.drop(['SalePrice', 'Id'], axis=1)


# Perform data augmentation by adding random noise
# X_val_augmented = X_val_augmented + 0.01 * X_val_augmented * np.random.normal(0, 1, X_val_augmented.shape)

# Perform Mutual Information feature selection
mi_scores = mutual_info_regression(X_val_augmented, y_val_augmented)

# Create a DataFrame with feature names and their corresponding MI scores
selected_features = pd.DataFrame({'Feature': X_val_augmented.columns, 'MI_Score': mi_scores})

print(len(selected_features))
# Set a threshold for selecting features based on MI score
threshold = 0.01
selected_features = selected_features[selected_features['MI_Score'] > threshold]

print(selected_features)
print(len(selected_features))

# Save the selected features to a CSV file
selected_features.to_csv('../../Data/Processed_Data/new_selected_features.csv', index=False)

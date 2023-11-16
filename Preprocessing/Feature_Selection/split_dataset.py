from sklearn.model_selection import train_test_split
import pandas as pd

# Load the preprocessed numerical dataset
df = pd.read_csv('../../Data/Processed_Data/new_numerical_data.csv')

# Define features (X) and target variable (y)
X = df.drop('SalePrice', axis=1) 
y = df['SalePrice']

# Split the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save the split datasets to separate files
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('../../Data/Processed_Data/new_train_data.csv', index=False)
val_data.to_csv('../../Data/Processed_Data/new_val_data.csv', index=False)
test_data.to_csv('../../Data/Processed_Data/new_test_data.csv', index=False)

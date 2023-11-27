import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


train_df = pd.read_csv("../../Data/Processed_Data/new_train_data.csv")
train_df = train_df.dropna()
test_df = pd.read_csv("../../Data/Processed_Data/new_test_data.csv")
test_df = test_df.dropna()


selected_features_df = pd.read_csv("../../Data/Processed_Data/new_selected_features.csv")
selected_features = selected_features_df['Feature']


X_train = train_df[selected_features]
y_train = train_df['SalePrice']

X_test = test_df[selected_features]


# Train the K-NN model with the best K (44)
knn = KNeighborsRegressor(n_neighbors=44)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

actual_values = test_df['SalePrice']
print(y_pred)
print(actual_values)
mse = mean_squared_error(actual_values, y_pred)
r2 = r2_score(actual_values, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

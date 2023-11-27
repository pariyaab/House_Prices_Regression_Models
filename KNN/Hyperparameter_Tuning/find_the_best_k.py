import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

selected_features_df = pd.read_csv("../../Data/Processed_Data/new_selected_features.csv")
selected_features = selected_features_df['Feature']
selected_features = list(selected_features)
selected_features.append("SalePrice")

df = pd.read_csv("../../Data/Processed_Data/new_numerical_data.csv")
df = df[selected_features]

test_errors = []
validation_errors_dict = {}

np.random.seed(42)

df_shuffled = df.sample(frac=1, random_state=np.random.seed()).reset_index(drop=True)

df_shuffled.to_csv("../../Data/Processed_Data/shuffled_data.csv")

df_split = np.array_split(df, 10)

for k in range(1, 51):

    cv_errors = []

    for iteration in range(10):
        train_indices = [(i + iteration) % 10 for i in range(8)]
        val_index = (iteration + 8) % 10
        test_index = (iteration + 9) % 10

        train = pd.concat([df_split[i] for i in train_indices])
        val = df_split[val_index]
        test = df_split[test_index]

        # Separate features and target variable
        X_train = train.drop(columns=['SalePrice'])
        y_train = train['SalePrice']
        X_val = val.drop(columns=['SalePrice'])
        y_val = val['SalePrice']

        # Initialize KNN regressor with current k
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        cv_errors.append(mse)

    average = sum(cv_errors) / len(cv_errors)
    validation_errors_dict[k] = average

sorted_errors = dict(sorted(validation_errors_dict.items(), key=lambda item: item[1]))

# Find the best k with minimum average error
best_k = min(sorted_errors, key=sorted_errors.get)

# Report the best k values
print("Best K values:", best_k)

# Plotting the errors for different k values
plt.figure(figsize=(10, 6))
plt.plot(sorted_errors.keys(), sorted_errors.values(), marker='o')
plt.title('Average Mean Squared Error for Different K Values')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Average Mean Squared Error')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K: {best_k}')
plt.legend()
plt.savefig("../../Data/Processed_Data/knn_errors.png")
plt.show()

# Step 7: Run on the test set for the best k
final_test_errors = []

knn = KNeighborsRegressor(n_neighbors=best_k)
test_scores = []

for iteration in range(10):
    train_indices = [(i + iteration) % 10 for i in range(8)]
    val_index = (iteration + 8) % 10
    test_index = (iteration + 9) % 10

    train = pd.concat([df_split[i] for i in train_indices])
    val = df_split[val_index]
    test = df_split[test_index]

    X_train = train.drop(columns=['SalePrice'])
    y_train = train['SalePrice']
    X_test = test.drop(columns=['SalePrice'])
    y_test = test['SalePrice']

    # Fit the model and predict on the test set
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    test_score = mean_squared_error(y_test, y_pred)
    test_scores.append(test_score)

mean_error = np.mean(test_scores)
std_deviation = np.std(test_scores)
final_test_errors.append((mean_error, std_deviation))

# Report the mean error and standard deviation for the best k
for i, (mean_error, std_deviation) in enumerate(final_test_errors):
    print(f"\nResults for Best K={best_k}:")
    print(f"Mean Error: {mean_error}")
    print(f"Standard Deviation: {std_deviation}")

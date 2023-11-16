import pandas as pd

# Load the datasets
train_df = pd.read_csv("./Data/train.csv")
test_df = pd.read_csv("./Data/test.csv")
sample_submission_df = pd.read_csv("./Data/sample_submission.csv")

test_with_submission = pd.merge(test_df, sample_submission_df[['Id', 'SalePrice']], on='Id', how='left')

# Concatenate train and test sets
final_df = pd.concat([train_df, test_with_submission], axis=0, ignore_index=True, sort=False)

# Save the merged dataset
final_df.to_csv("./Data/Processed_Data/original_dataset.csv", index=False)
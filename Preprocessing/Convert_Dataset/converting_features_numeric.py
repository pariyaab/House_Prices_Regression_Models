import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the dataset
df = pd.read_csv('../../Data/Processed_Data/original_dataset.csv')

# Save the 'SalePrice' column for later
Sale_Price = df['SalePrice']

# Numerical features scaling using MinMaxScaler
numerical_features = df.drop(columns=['Id']).select_dtypes(include='number').columns
minmax_scaler = MinMaxScaler()
df[numerical_features] = minmax_scaler.fit_transform(df[numerical_features])

# One-hot encode categorical features
enc = OneHotEncoder(handle_unknown='ignore')
categorical_features = [
    'MSZoning', 'Alley', 'LotArea', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'Street',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
    'SaleCondition'
]
enc_df = pd.DataFrame(enc.fit_transform(df[categorical_features]).toarray(),
                      columns=enc.get_feature_names_out(categorical_features))
df = pd.concat([df, enc_df], axis=1)

# Drop original categorical features
df = df.drop(columns=categorical_features)

# Feature engineering: create 'TotalSF'
# df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']

# Drop unnecessary columns
# columns_to_drop = ['1stFlrSF', '2ndFlrSF']
# df = df.drop(columns=columns_to_drop)

# Add back the 'SalePrice' column
# df['SalePrice'] = Sale_Price
df = df.dropna()

# Save the updated DataFrame
# df.to_csv('../../Data/Processed_Data/new_numerical_data.csv', index=False)
print(df.shape)
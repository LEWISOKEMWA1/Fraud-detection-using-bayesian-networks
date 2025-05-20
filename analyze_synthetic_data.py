import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

# Path to the synthetic data file
data_file = 'fraud-detection--main/src/synthetic_fraud_data.csv'

# Read the data
print(f"Reading data from {data_file}...")
df = pd.read_csv(data_file)

# Print basic information
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum().sum()}")

# Take 5000 rows (or all if less than 5000)
sample_size = min(5000, len(df))
print(f"Taking {sample_size} rows from the dataset...")
df_sample = df.sample(n=sample_size, random_state=42)

# Check for missing values
missing_values = df_sample.isnull().sum().sum()
print(f"Missing values in the sample: {missing_values}")

# If there are missing values, use Expectation Maximization (via IterativeImputer) to fill them
if missing_values > 0:
    print("Filling missing values using Expectation Maximization...")
    
    # Separate numeric and categorical columns
    numeric_cols = df_sample.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_sample.select_dtypes(exclude=np.number).columns.tolist()
    
    # Handle numeric columns with IterativeImputer (EM-like)
    if numeric_cols:
        imp = IterativeImputer(max_iter=10, random_state=42)
        df_sample[numeric_cols] = imp.fit_transform(df_sample[numeric_cols])
    
    # Handle categorical columns with mode imputation
    if categorical_cols:
        for col in categorical_cols:
            if df_sample[col].isnull().sum() > 0:
                df_sample[col] = df_sample[col].fillna(df_sample[col].mode()[0])

# Save the prepared dataset
output_file = 'prepared_fraud_data.csv'
df_sample.to_csv(output_file, index=False)
print(f"Prepared dataset saved to {output_file}")

# Print sample statistics
print("\nSample Dataset Statistics:")
print(f"Fraud cases: {df_sample['fraud_Cases'].sum()} ({df_sample['fraud_Cases'].sum() / len(df_sample) * 100:.2f}%)")
print(f"Non-fraud cases: {len(df_sample) - df_sample['fraud_Cases'].sum()} ({(1 - df_sample['fraud_Cases'].sum() / len(df_sample)) * 100:.2f}%)")

# Modify config to use this file
print("\nTo use this prepared dataset, you need to update the config.json file:")
print("1. Set 'use_real_data' to true")
print("2. Set 'csv_file_path' to '" + os.path.abspath(output_file) + "'")
print("3. Run the model with: python fraud-detection--main/src/main.py --config fraud-detection--main/config.json") 
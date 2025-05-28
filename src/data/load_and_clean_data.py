import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    # Load the dataset from the specified file path
    df = pd.read_csv(file_path)
    
    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaN values in 'TotalCharges' with the median value
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Convert 'Churn' column to binary (1 for 'Yes', 0 for 'No')
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError("Column 'Churn' not found in the dataset.")
    
    # Drop columns that are not useful for training
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include='object').columns
    
    # Apply LabelEncoder to all categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y
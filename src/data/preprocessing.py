import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)
    
    # Tratamento TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Criar faixa de tempo (tenure_group)
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 60, df['tenure'].max() + 1],
        labels=['0-1 ano', '1-2 anos', '2-4 anos', '4-5 anos', '5+ anos']
    )
    
    # One-hot encoding para tenure_group e outras categóricas (exceto customerID)
    df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Padronizar numéricas
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Preparar X e y
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return X, y
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Limpa espaços em branco das colunas
    df.columns = df.columns.str.strip()
    
    # Converte TotalCharges para numérico e trata NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Converte 'Churn' para binário
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError("Coluna 'Churn' não encontrada no dataset.")
    
    # Remove colunas não úteis para treino
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Identifica colunas categóricas
    cat_cols = X.select_dtypes(include='object').columns
    
    # Aplica LabelEncoder para todas as categóricas (apenas para treino/validação)
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y
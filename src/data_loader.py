import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Initial data cleaning: type conversion and missing value handling. 
    df = df.copy()
    
    # Coerce TotalCharges to numeric; errors become NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Standardize binary columns
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    return df
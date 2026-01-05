import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def get_cohort(tenure):
        if tenure <= 12: return '0-12 months'
        elif tenure <= 24: return '13-24 months'
        elif tenure <= 36: return '25-36 months'
        elif tenure <= 48: return '37-48 months'
        return '49+ months'
    
    df['tenure_cohort'] = df['tenure'].apply(get_cohort)
    df['charges_per_month_ratio'] = df['TotalCharges'] / (df['tenure'] + 1e-5)

    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies']
    
    for col in service_cols:
        if col in df.columns:
            df[f'{col}_binary'] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    return df

def build_model_pipeline() -> Pipeline:
    categorical_cols = ['Contract', 'PaymentMethod', 'tenure_cohort', 
                        'gender', 'Partner', 'Dependents', 'SeniorCitizen']
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month_ratio']

    binary_cols = ['PhoneService_binary', 'InternetService_binary', 'OnlineSecurity_binary', 
                   'OnlineBackup_binary', 'DeviceProtection_binary', 'TechSupport_binary', 
                   'StreamingTV_binary', 'StreamingMovies_binary']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('bin', 'passthrough', binary_cols) 
        ],
        remainder='drop' 
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=3,  
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline
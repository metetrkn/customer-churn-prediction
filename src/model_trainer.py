import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict, Any
from sklearn.metrics import roc_curve, auc

def split_data(df: pd.DataFrame, target: str = 'Churn') -> Tuple:
    # Splits data into train and test sets.
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Evaluates the model with a custom decision threshold.
    y_probs = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_probs >= threshold).astype(int)
    
    print(f"\n--- Classification Report (Threshold: {threshold}) ---")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Threshold: {threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def generate_risk_report(model, X_data):
    # Generates a full risk report for ALL customers, categorized by risk level. 
    probs = model.predict_proba(X_data)[:, 1]
    
    report = X_data.copy()
    report['Churn_Probability'] = probs
    
    # Tag risk levels based on probability thresholds
    def get_risk_level(prob):
        if prob > 0.7: return 'High Risk'
        if prob > 0.3: return 'Medium Risk'
        return 'Low Risk'

    report['Risk_Level'] = report['Churn_Probability'].apply(get_risk_level)
    
    # Sort by Probability (Highest risk at top)
    report = report.sort_values(by='Churn_Probability', ascending=False)
    
    # Return the full dataset with selected columns
    columns_to_show = ['customerID', 'Churn_Probability', 'Risk_Level', 'Contract', 'MonthlyCharges']
    valid_cols = [col for col in columns_to_show if col in report.columns]
    
    return report[valid_cols]

def plot_roc_curve(model, X_test, y_test):
    
    # Plots the ROC Curve to help choose the best threshold.
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate rates for every possible threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve: Trade-off between Recall and False Alarms')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
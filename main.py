import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, clean_data
from src.feature_engineering import add_domain_features, build_model_pipeline
from src.model_trainer import split_data, evaluate_model, generate_risk_report, plot_roc_curve

def main():
    DATA_PATH = os.path.join('data', 'Telco-Customer-Churn.csv')
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("1. Loading and cleaning data...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    
    print("2. Generating domain features...")
    df = add_domain_features(df)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print("3. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target='Churn')
    
    print("4. Training Pipeline (Preprocessing + Model)...")
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)
    
    print("5. Evaluating model...")
    evaluate_model(model_pipeline, X_test, y_test, threshold=0.3)

    print("\n6. Generating Full Risk Report...")
    full_report = generate_risk_report(model_pipeline, X_test)
    
    print("\n--- REPORT PREVIEW (Top 5 Highest Risk) ---")
    print(full_report.head(5).to_string(index=False))

    print("Generating ROC Curve...")
    plot_roc_curve(model_pipeline, X_test, y_test)

    full_report.to_csv("all_customers_risk_report.csv", index=False)
    print("\nFull report saved to 'all_customers_risk_report.csv'")
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
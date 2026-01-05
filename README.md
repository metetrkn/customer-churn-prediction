# Telco Customer Churn Prediction

## 1. Executive Overview
**Objective:** The primary goal of this project is to reduce customer attrition by predicting which customers are likely to churn. By identifying at-risk customers early, the business can proactively intervene with targeted retention strategies.

**Result:** The final model prioritizes sensitivity, achieving a **Recall of 90%** for the churn class. This means the model successfully identifies **9 out of 10** at-risk customers, ensuring minimal opportunities for churn are missed, even at the cost of higher False Positives.

**OBS!** Even though main.py is the entry point of the app, every process is also repeated in notebooks/analysis.ipynb file for presentation purposes. It also creates an output file named all_customers_risk_report.csv in its current path which is identical to main.py's output.

## 2. Dataset Description
The analysis utilizes the Telco Customer Churn dataset (7,032 records), comprising 21 features:
- **Demographics:** Gender, senior citizen status, partner, dependents.
- **Services:** Phone, multiple lines, internet (DSL/Fiber), online security, backup, device protection, tech support.
- **Contract & Billing:** Contract type (Month-to-month, 1yr, 2yr), paperless billing, payment method, monthly charges, total charges.
- **Target:** Churn status.

## 3. Technical Architecture
The project follows a modular, production-ready structure designed for reproducibility and scalability.

## Project Structure
```text
churn_project/
├── data/
│   └── Telco-Customer-Churn.csv         # Raw input dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data ingestion and cleaning logic
│   ├── feature_engineering.py           # Feature creation and pipeline setup
│   └── model_trainer.py                 # XGBoost training and evaluation wrappers
├── main.py                              # End-to-end execution script
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── LICENSE                              # MIT License
├── all_customers_risk_report.csv        # Final output: Sorted list of at-risk customers
└──  .dcignore                            # Specifies files to exclude from DeepCode (Snyk) security analysis.
```

### The Pipeline Approach
To ensure robust performance and prevent **Data Leakage**, this project utilizes `sklearn.pipeline.Pipeline`:
1.  **Split-First Strategy:** Data is split into Training and Test sets *before* any transformations occur.
2.  **Encapsulation:** Preprocessing steps (Scaling, Encoding) are bundled with the model.
3.  **Production Simulation:** Statistics (mean, variance) and categories learned strictly from training data are applied to test data, simulating real-world conditions.

## 4. Methodology & Feature Engineering

### Data Processing
- **Categorical Encoding:** `OneHotEncoder` is applied to nominal variables to prevent the model from inferring false ordinal relationships.
- **Numerical Scaling:** `StandardScaler` standardizes continuous variables to ensure features contribute equally.

### Key Domain Features
Specific features were engineered to capture customer behavior nuances:
- **Tenure Cohorts:** Customers are grouped by lifecycle stages (e.g., "0-12 months", "49+ months").
- **Interaction Ratios:** The `charges_per_month_ratio` metric identifies customers paying disproportionately high rates relative to their tenure.

## 5. Model Performance & Analysis
We utilized **Gradient Boosting (XGBoost)**, specifically tuned to maximize Recall.

### Strategic Optimization (Threshold: 0.3)
To minimize the business cost of lost customers, the decision threshold was lowered to **0.3**.

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **Recall (Churn)** | **0.90** | The model detects 90% of all actual churners. |
| **Precision (Churn)** | 0.42 | ~42% of flagged customers actually churn; the rest are safe but flagged "at-risk." |

**Business Justification:** A "False Positive" (sending a retention offer to a happy customer) costs significantly less than a "False Negative" (losing a customer because we failed to detect them).

## 6. Final Deliverable: Risk Report
The system generates an actionable `all_customers_risk_report.csv` containing:
1.  **CustomerID**: For direct lookup.
2.  **Churn_Probability**: Raw score (0.0 - 1.0).
3.  **Risk_Level**:
    * 🔴 **High Risk** (> 70%)
    * 🟡 **Medium Risk** (30% - 70%)
    * 🟢 **Low Risk** (< 30%)

## 7. Setup and Usage

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
 - Run the full pipeline (Data Load -> Process -> Train -> Evaluate -> Report):
   ```bash
   python main.py
   ```

The risk report will be saved to the output directory upon completion.

### Example Execution Output
   ```
1. Loading and cleaning data...
2. Generating domain features...
3. Splitting data...
4. Training Pipeline (Preprocessing + Model)...
5. Evaluating model...

--- Classification Report (Threshold: 0.3) ---
              precision    recall  f1-score   support

           0       0.94      0.55      0.69      1033
           1       0.42      0.90      0.57       374

    accuracy                           0.64      1407
   macro avg       0.68      0.73      0.63      1407
weighted avg       0.80      0.64      0.66      1407

6. Generating Full Risk Report...

--- REPORT PREVIEW (Top 5 Highest Risk) ---
customerID  Churn_Probability Risk_Level       Contract  MonthlyCharges
5178-LMXOP           0.960488  High Risk Month-to-month           95.10
6857-VWJDT           0.958122  High Risk Month-to-month           95.65
1069-XAIEM           0.957847  High Risk Month-to-month           85.05
8375-DKEBR           0.953095  High Risk Month-to-month           69.60
3722-WPXTK           0.949763  High Risk Month-to-month           88.35
Generating ROC Curve...

Full report saved to 'all_customers_risk_report.csv'
Pipeline completed successfully.

### Model Precision
--- Classification Report (Threshold: 0.3) ---
              precision    recall  f1-score   support

           0       0.94      0.55      0.69      1033
           1       0.42      0.90      0.57       374

    accuracy                           0.64      1407
   macro avg       0.68      0.73      0.63      1407
weighted avg       0.80      0.64      0.66      1407
   ```

## 8. Results & Achievements

### Model Performance
We successfully implemented an **XGBoost (Extreme Gradient Boosting)** model, tuned to prioritize the identification of at-risk customers.

By adjusting the decision threshold to **0.3**, we achieved a **Recall of 90%** for the churn class.
* **Recall (90%):** We successfully catch 9 out of every 10 customers who are actually going to churn.
* **Precision (42%):** While we accept some False Positives, this trade-off is intentional to ensure we do not miss valuable customers leaving the company.

### Business Value
1.  **Risk Scoring:** Every customer in the database is assigned a `Churn_Probability` score (0.0 to 1.0).
2.  **Actionable Reporting:** We generate a final output file, `all_customers_risk_report.csv`.
    * **Sorted Priority:** This file is automatically sorted by **descending churn probability**, placing the highest-risk customers at the very top.
    * **Immediate Action:** Marketing teams can simply open this file and target the top rows for immediate retention campaigns.

### Author
Mete Turkan
LinkedIn: linkedin.com/in/mete-turkan/

### License
This project is licensed under the [MIT License](LICENSE).
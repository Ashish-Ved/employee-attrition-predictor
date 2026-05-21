# employee-attrition-predictor

Employee Attrition Predictor

ML classification project · IBM HR Analytics dataset · Scikit-learn · XGBoost

Show Image Show Image Show Image Show Image

**Problem statement**
Employee attrition costs organisations an estimated 6–9 months of an employee's salary in recruitment and onboarding costs. This project builds a machine learning model to predict which employees are at risk of leaving, enabling HR teams to intervene early with targeted retention strategies.
Business question: Given an employee's demographic, role, and satisfaction data — how likely are they to leave in the next review cycle?

**Dataset**
PropertyDetailSourceIBM HR Analytics Employee Attrition & PerformanceRecords1,470 employeesFeatures35 (demographic, role, compensation, satisfaction)TargetAttrition — Yes / NoClass balance16.1% attrition (imbalanced)Missing valuesNone
Download from Kaggle.

**Project structure**
employee-attrition-predictor/
│
├── notebook/
│   └── attrition_predictor.ipynb   # Full notebook (EDA → model → report)
│
├── report_assets/                  # Auto-generated charts
│   ├── chart_attrition.png
│   ├── chart_overtime.png
│   ├── chart_importance.png
│   └── chart_cm.png
│
├── employee_attrition_report.pdf   # Client-ready findings report
├── requirements.txt
└── README.md

**Approach**
1. Exploratory data analysis

Visualised attrition by department, overtime, age, and monthly income
Identified overtime and low income as the two strongest attrition signals
Dropped 3 zero-variance columns: EmployeeCount, StandardHours, Over18

**2. Feature engineering**

Label encoded binary columns (Gender, OverTime, etc.)
One-hot encoded multi-class columns (Department, JobRole, MaritalStatus, etc.)
Scaled numeric features with StandardScaler
Applied SMOTE to training data only to address the 16.1% class imbalance

**3. Model training & evaluation**
Three models were trained on an 80/20 stratified split:
ModelF1 ScoreROC-AUCLogistic Regression~0.52~0.80Random Forest~0.62~0.85XGBoost ✓~0.65~0.87

**Primary metric:** F1 Score (not accuracy) — because the dataset is imbalanced, accuracy alone is misleading. A model predicting "No" for everyone achieves 84% accuracy but 0% recall on attrition cases.


**Key findings**

Overtime is the strongest single predictor — employees on overtime leave at ~30% vs ~10% for those who don't
Monthly income is inversely correlated with attrition — lower-paid employees are significantly more likely to leave
Younger employees (25–35) show the highest attrition risk
Sales and HR departments have higher attrition than R&D
SMOTE improved recall on the attrition class without leaking into the test set


**Business recommendations**
Priority      Action
High          Flag high-risk employees monthly using model probability scores
High          Review overtime policies in Sales and HR
Medium        Introduce career development tracks for employees aged 25–35
Medium        Segment employees into risk tiers (high / medium / low) for targeted intervention
Low           Re-train the model quarterly as workforce composition changes

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('heart_failure_clinical_records.csv')

# 2. Preprocessing
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# 3. Define models
models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
]

# 4. Stacking ensemble
stack = StackingClassifier(estimators=models, final_estimator=RandomForestClassifier(), cv=5)

# 5. Hyperparameter tuning (optional)
# param_grid = {...}
# stack = GridSearchCV(stack, param_grid, cv=5, scoring='roc_auc')

# 6. Train
stack.fit(X_res, y_res)

# 7. Evaluate
y_pred = stack.predict(X_test_scaled)
y_proba = stack.predict_proba(X_test_scaled)[:, 1]
print("AUC-ROC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# 8. Explain with SHAP (using XGBoost model)
xgb_model = stack.named_estimators_['xgb']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test, plot_type="bar")

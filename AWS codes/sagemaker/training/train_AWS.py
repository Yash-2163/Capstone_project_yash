import os
import argparse
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import boto3

from data_loader_AWS import load_data  # You must use the args here

# --- Step 0: Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--PREPROCESSING_PIPELINE_URI', type=str)
parser.add_argument('--REDSHIFT_HOST', type=str)
parser.add_argument('--REDSHIFT_PORT', type=str)
parser.add_argument('--REDSHIFT_DB', type=str)
parser.add_argument('--REDSHIFT_USER', type=str)
parser.add_argument('--REDSHIFT_PASSWORD', type=str)
args = parser.parse_args()

# --- Step 1: Load data from Redshift ---
X_raw, y = load_data(
    host=args.REDSHIFT_HOST,
    port=args.REDSHIFT_PORT,
    database=args.REDSHIFT_DB,
    user=args.REDSHIFT_USER,
    password=args.REDSHIFT_PASSWORD
)

# --- Step 2: Load preprocessing pipeline from S3 ---
pipeline_s3_uri = args.PREPROCESSING_PIPELINE_URI
local_pipeline_path = "/opt/ml/input/data/preprocessing_pipeline.pkl"

s3 = boto3.client("s3")
bucket, key = pipeline_s3_uri.replace("s3://", "").split("/", 1)
s3.download_file(bucket, key, local_pipeline_path)

preprocessing = joblib.load(local_pipeline_path)

# --- Step 3: Define model and hyperparameters ---
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'model__n_estimators': [100],
    'model__learning_rate': [0.1],
    'model__max_depth': [3]
}

# --- Step 4: Create pipeline ---
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', xgb_model)
])

# --- Step 5: Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# --- Step 6: Fit ---
grid_search.fit(X_raw, y)

# --- Step 7: Save model ---
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
joblib.dump(grid_search.best_estimator_, os.path.join(output_dir, "final_model_pipeline.pkl"))

# --- Step 8: Evaluate ---
y_pred = grid_search.predict(X_raw)
y_proba = grid_search.predict_proba(X_raw)[:, 1]

print("\n✅ Final Model Performance:")
print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall   : {recall_score(y, y_pred):.4f}")
print(f"F1 Score : {f1_score(y, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y, y_proba):.4f}")

import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure model_training imports resolve when running under Airflow
BASE_DIR = "/opt/airflow"
MODEL_TRAINING_DIR = os.path.join(BASE_DIR, "model_training")
sys.path.insert(0, MODEL_TRAINING_DIR)

from data_loader import load_data
from preprocessing import build_full_preprocessing_pipeline

# Paths for final model artifacts
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
PREPROCESSING_PATH = os.path.join(FINAL_MODEL_DIR, "preprocessing_pipeline.pkl")
OUTPUT_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "final_model_pipeline.pkl")

# --- Step 1: Load Data ---
X_raw, y = load_data()

# --- Step 2: Load saved preprocessing pipeline ---
preprocessing = joblib.load(PREPROCESSING_PATH)

# --- Step 3: Define model and hyperparameters ---
lgbm = LGBMClassifier(random_state=42)
param_grid = {
    'model__n_estimators': [100],
    'model__learning_rate': [0.1],
    'model__max_depth': [3]
}

# --- Step 4: Create full pipeline with loaded preprocessing and model ---
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', lgbm)
])

# --- Step 5: Cross-validation and hyperparameter tuning ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# --- Step 6: MLflow Tracking ---
# Use the mlflow service hostname from docker-compose
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("lead_conversion_retraining")

with mlflow.start_run(run_name="lgbm_retrain_run"):
    # Log hyperparameters
    mlflow.log_params({
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    })

    # Train model
    grid_search.fit(X_raw, y)

    # Save the best pipeline
    joblib.dump(grid_search.best_estimator_, OUTPUT_MODEL_PATH)

    # Log model to MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        artifact_path="model",
        registered_model_name="LeadConversionModel"
    )

    # Evaluate on training set
    y_pred = grid_search.predict(X_raw)
    y_proba = grid_search.predict_proba(X_raw)[:, 1]

    # Log metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba)
    }
    mlflow.log_metrics(metrics)

    # Print performance
    print("\n✅ Final Model Performance:")
    for name, val in metrics.items():
        print(f"{name.capitalize():10}: {val:.4f}")





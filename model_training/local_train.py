import os
import sys
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import shap
import matplotlib.pyplot as plt

# --- Setup Paths and Imports ---
# This setup assumes the script is run from a consistent base directory.
# For local development, you might adjust these paths.
# For Docker/Airflow, these absolute paths are standard.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_TRAINING_DIR = os.path.join(BASE_DIR, "model_training")
if MODEL_TRAINING_DIR not in sys.path:
    sys.path.insert(0, MODEL_TRAINING_DIR)

# Import your custom functions and classes
from data_loader import load_data
# The custom class definitions are needed for joblib to load the pipeline object
from preprocessing import CategoricalPreprocessor 

# --- Define File Paths ---
# This script assumes 'final_model' directory is at the project root.
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
# Path to the pre-saved preprocessing pipeline
PREPROCESSING_PATH = os.path.join(FINAL_MODEL_DIR, "preprocessing_pipeline.pkl")
# Path to save the final, complete model pipeline
OUTPUT_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "final_model_pipeline.pkl")

# Ensure the output directory exists
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)


# --- Step 1: Load and Split Data ---
print("▶ Step 1: Loading data...")
# Assumes data_loader.py is in the same directory or accessible via sys.path
X_raw, y = load_data()

print("▶ Step 1.1: Splitting data into training and validation sets...")
# Stratified split is crucial for imbalanced datasets to ensure both train and
# validation sets have a similar distribution of the target variable.
X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set size: {len(X_train)} rows")
print(f"   Validation set size: {len(X_val)} rows")


# --- Step 2: Load the Pre-saved Preprocessing Pipeline ---
print(f"▶ Step 2: Loading preprocessing pipeline from {PREPROCESSING_PATH}...")
try:
    preprocessing_pipeline = joblib.load(PREPROCESSING_PATH)
    print("   ✅ Preprocessing pipeline loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Preprocessing pipeline not found at: {PREPROCESSING_PATH}")
    print("   Please ensure you have run a script to build and save the preprocessing pipeline first.")
    sys.exit(1)


# --- Step 3: Define Model and Hyperparameter Grid ---
print("▶ Step 3: Defining model and hyperparameter grid...")
lgbm = LGBMClassifier(random_state=42)
# A simple grid for demonstration purposes. You can expand this.
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5]
}


# --- Step 4: Create the Full Training Pipeline ---
# This pipeline chains the loaded preprocessing steps with the model
full_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('model', lgbm)
])


# --- Step 5: Hyperparameter Tuning with GridSearchCV ---
print("▶ Step 5: Setting up GridSearchCV for hyperparameter tuning...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1 # Use all available CPU cores
)


# --- Step 6: Train the Model and Save the Best Version ---
print("▶ Step 6: Starting model training with GridSearchCV...")
grid_search.fit(X_train, y_train)

# After the search is complete, extract the best pipeline
best_pipeline = grid_search.best_estimator_

print(f"\n✅ Training complete. Best ROC AUC score: {grid_search.best_score_:.4f}")
print(f"   Best parameters found: {grid_search.best_params_}")

# --- Save the final, complete pipeline (preprocessor + best model) ---
print(f"\n▶ Saving the best model pipeline to: {OUTPUT_MODEL_PATH}")
joblib.dump(best_pipeline, OUTPUT_MODEL_PATH)
print("   ✅ Model saved successfully.")


# --- Step 7: Evaluate the Final Model on the Validation Set ---
print("\n▶ Step 7: Evaluating the final model on the held-out validation set...")
y_pred_val = best_pipeline.predict(X_val)
y_proba_val = best_pipeline.predict_proba(X_val)[:, 1]

# Calculate metrics, handling the case where a class might be missing in a small validation set
metrics = {
    "val_accuracy": accuracy_score(y_val, y_pred_val),
    "val_precision": precision_score(y_val, y_pred_val, zero_division=0),
    "val_recall": recall_score(y_val, y_pred_val, zero_division=0),
    "val_f1_score": f1_score(y_val, y_pred_val, zero_division=0)
}
if len(np.unique(y_val)) > 1:
    metrics["val_roc_auc"] = roc_auc_score(y_val, y_proba_val)
else:
    metrics["val_roc_auc"] = float('nan')

print("\n✅ Final Model Performance on Validation Set:")
for name, val in metrics.items():
    print(f"  {name.replace('_', ' ').capitalize():15}: {val:.4f}")

print("\n🎉 Full training process complete.")

# --- Step 8: SHAP Explainability Analysis ---
print("\n▶ Step 8: Running SHAP explainability analysis...")

# SHAP requires the raw model, not the pipeline
# Extract the trained LGBMClassifier from the pipeline
model = best_pipeline.named_steps['model']
# Apply preprocessing to validation data
X_val_processed = best_pipeline.named_steps['preprocessing'].transform(X_val)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_val_processed)

# SHAP summary plot (comment this if running in headless environment)
plt.figure()
shap.summary_plot(shap_values, X_val_processed, show=False)
plt.tight_layout()
shap_summary_path = os.path.join(FINAL_MODEL_DIR, "shap_summary_plot.png")
plt.savefig(shap_summary_path)
print(f"   ✅ SHAP summary plot saved to: {shap_summary_path}")


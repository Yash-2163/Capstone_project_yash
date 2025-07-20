# This is to test the model.pkl file if it's working properly

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_loader import load_data  # Custom function to load local raw data

# Constants
SAMPLE_SIZE = 300           # Number of rows to sample from the dataset for quick testing
RANDOM_STATE = 42           # For reproducibility of the sample

# --- Load full pipeline (preprocessing + model) ---
# The saved pipeline includes all preprocessing steps and the final model
model_pipeline = joblib.load('../final_model/final_model_pipeline.pkl')

# --- Load raw data ---
# This function returns raw features (X) and target variable (y)
X, y = load_data()

# --- Take random subset for testing ---
# Select a random sample from the dataset to quickly validate the model
sample_df = X.sample(n=min(SAMPLE_SIZE, len(X)), random_state=RANDOM_STATE)
y_sample = y.loc[sample_df.index]  # Ensure labels correspond to the sampled rows

# --- Predict using full pipeline (which includes preprocessing) ---
# Since the model_pipeline includes preprocessing, we can pass raw features directly
y_pred = model_pipeline.predict(sample_df)              # Predicted class labels
y_proba = model_pipeline.predict_proba(sample_df)[:, 1] # Predicted probabilities for the positive class

# --- Evaluation ---
# Print standard classification metrics on the sampled data
print("📊 Test Set Performance on Sample:")
print("Accuracy: ", round(accuracy_score(y_sample, y_pred), 4))
print("Precision:", round(precision_score(y_sample, y_pred), 4))
print("Recall:   ", round(recall_score(y_sample, y_pred), 4))
print("F1 Score: ", round(f1_score(y_sample, y_pred), 4))
print("ROC AUC:  ", round(roc_auc_score(y_sample, y_proba), 4))

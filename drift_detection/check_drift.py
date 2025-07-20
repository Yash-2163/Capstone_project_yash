import pandas as pd
import os
import json
import sys  # Import sys module

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from evidently import Report
from evidently.presets import DataDriftPreset
from model_training.data_loader import load_data

# ─── Global Paths ────────────────────────────────────────────────────────
BASE_DIR = "/opt/airflow"
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")

# ─── Step 1: Load reference and new data ────────────────────────────────
# Reference data is the baseline data already stored
reference_data_path = os.path.join(FINAL_MODEL_DIR, "reference_data.csv")
reference_data = pd.read_csv(reference_data_path)

# New data is loaded using your custom load_data function (e.g., from Postgres)
new_data, y = load_data()

# Normalize column names to lowercase to ensure consistent matching
reference_data.columns = reference_data.columns.str.lower()
new_data.columns = new_data.columns.str.lower()

# ─── Step 2: Align schemas (drop ID cols) ────────────────────────────────
# Remove identifier columns if present
ID_COLS = ["prospect_id", "lead_number"]
reference_data = reference_data.drop(columns=ID_COLS, errors="ignore")
new_data = new_data.drop(columns=ID_COLS, errors="ignore")

# Add the target column back to new data
new_data['converted'] = y

# Keep only the columns that are common in both datasets
common_cols = reference_data.columns.intersection(new_data.columns)
reference_data = reference_data[common_cols]
new_data = new_data[common_cols]

# ─── Step 2.1: Convert 'converted' column to int (or numeric) ────────────
# Ensure the target column is numeric in both datasets
for df_name, df in [("reference_data", reference_data), ("new_data", new_data)]:
    if 'converted' in df.columns:
        df['converted'] = pd.to_numeric(df['converted'], errors='coerce').fillna(0).astype(int)
    else:
        raise KeyError(f"'converted' column missing in {df_name}")

# ─── Step 3: Create and run drift report ─────────────────────────────────
# Generate a drift detection report using Evidently's DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(
    reference_data=reference_data,
    current_data=new_data
)

# ─── Step 4: Extract results dict ────────────────────────────────────────
# Convert the report output to dictionary format for programmatic use
result = snapshot.dict()
print("=== EVIDENTLY OUTPUT ===")
print(json.dumps(result, indent=2))

# ─── Step 5: Determine drift based on DriftedColumnsCount ───────────────
# Check how many columns were detected as drifted
drifted_cols_metric = result.get("metrics", [])[0]
count = drifted_cols_metric.get("value", {}).get("count", 0)
drift_detected = count > 0  # Drift is considered detected if any column has drifted

# ─── Step 6: Write flag ─────────────────────────────────────────────────
# Store result to a file which Airflow DAG will later check
flag_path = os.path.join(FINAL_MODEL_DIR, "drift_detected.txt")
with open(flag_path, "w") as f:
    f.write("true" if drift_detected else "false")

print(f"Data drift detected? {drift_detected}")

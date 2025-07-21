# Filename: dags/scripts/check_drift.py

import pandas as pd
import boto3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def check_drift():
    """
    Compares new data from Redshift with reference data from S3 for drift.
    Pushes the result ('true' or 'false') to Airflow XComs.
    """
    # --- Configuration ---
    S3_BUCKET = "sagemakerbucket2163"
    REFERENCE_DATA_KEY = "reference/training_data.csv"
    # In a real scenario, this would be the path to the new data from Redshift
    NEW_DATA_KEY = "reference/sample_data1.csv" # Using a sample file for this example
    
    # --- Setup Connections ---
    s3_client = boto3.client('s3')

    # --- Load Data ---
    print("Loading reference data...")
    ref_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=REFERENCE_DATA_KEY)
    ref_df = pd.read_csv(ref_obj['Body'])

    print("Loading new data...")
    new_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=NEW_DATA_KEY)
    new_df = pd.read_csv(new_obj['Body'])

    # --- Perform Drift Detection ---
    print("Generating data drift report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_df, current_data=new_df)
    
    drift_results = drift_report.as_dict()
    is_drift_detected = drift_results['metrics'][0]['result']['dataset_drift']
    
    print(f"Drift detected: {is_drift_detected}")

    # --- Push result to XComs ---
    # Airflow will automatically capture the last line printed to stdout
    # and push it to XComs if do_xcom_push=True in the operator.
    # A more explicit way is to write to a specific file Airflow watches.
    # For BashOperator, we'll write to a file that the next task can read.
    # A better approach is to use the TaskFlow API or a PythonOperator.
    
    # For now, we'll simulate pushing to XComs by writing to a file.
    # The DAG's BranchPythonOperator will need to be adapted to read this.
    # Let's assume the DAG is adapted to use XComs properly.
    
    # This is how you'd push to XComs from a PythonOperator
    # from airflow.models.taskinstance import TaskInstance
    # ti = TaskInstance(...) # This context is provided automatically
    # ti.xcom_push(key='drift_detected', value=str(is_drift_detected).lower())
    
    # For BashOperator, we can print the value and have the DAG capture it
    # Or write to a known file path. Let's assume the DAG handles this.
    # The provided DAG has been updated to use XComs.
    
    # To make this script work with the BashOperator, we need to push to XComs.
    # The easiest way is to use the Airflow API if available, or write to the designated file.
    # Let's write to the default XCom return file.
    import json
    with open('/airflow/xcom/return.json', 'w') as f:
        json.dump({'drift_detected': str(is_drift_detected).lower()}, f)

if __name__ == "__main__":
    check_drift()


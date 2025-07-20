# dags/retrain_dag.py 2

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# Default arguments applied to all tasks unless overridden
default_args = {
    'owner': 'yash',
    'start_date': datetime(2024, 1, 1),  # DAG will not run for dates before this
    'retries': 1  # Number of retries for failed tasks
}

# Python function used by BranchPythonOperator to decide the path
def decide_branch():
    flag_path = '/opt/airflow/final_model/drift_detected.txt'  # File that indicates whether drift was detected
    try:
        with open(flag_path, 'r') as f:
            drift_flag = f.read().strip().lower()  # Read and normalize the value (e.g., 'True' -> 'true')
    except FileNotFoundError:
        # If drift flag file is missing, assume no drift
        drift_flag = 'false'
    
    # Based on the value in the file, return the next task's ID
    return 'retrain_model' if drift_flag == 'true' else 'skip_retrain'

# DAG definition block
with DAG(
    dag_id='model_retraining_on_drift',  # Unique DAG name
    default_args=default_args,
    schedule_interval='@daily',  # Runs every day
    catchup=False  # Don't run DAGs for past dates
) as dag:

    # Task 1: Run the drift detection script
    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='PYTHONPATH="/opt/airflow" python /opt/airflow/drift_detection/check_drift.py'
        # Ensures the script runs with correct module path
    )

    # Task 2: Decide the next step based on drift detection
    branch_on_drift = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=decide_branch  # This will return either 'retrain_model' or 'skip_retrain'
    )

    # Task 3a: If drift detected, retrain the model
    retrain_model = BashOperator(
        task_id='retrain_model',
        bash_command='python /opt/airflow/model_training/airflow_train.py'
    )

    # Task 3b: If no drift, skip retraining (no-op)
    skip_retrain = EmptyOperator(
        task_id='skip_retrain'
    )

    # Final task to mark completion
    end = EmptyOperator(
        task_id='end'
    )

    # DAG dependencies: define the task flow
    detect_drift >> branch_on_drift
    branch_on_drift >> retrain_model >> end
    branch_on_drift >> skip_retrain >> end

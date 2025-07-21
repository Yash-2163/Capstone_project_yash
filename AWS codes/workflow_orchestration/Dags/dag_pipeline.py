# Filename: dags/lead_conversion_retraining_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# --- Configuration Variables ---
# S3 locations
BUCKET_NAME = "sagemakerbucket2163" # Your S3 bucket
CODE_URI = f"s3://{BUCKET_NAME}/code/train.tar.gz" # Path to your packaged training code
MODEL_OUTPUT_PATH = f"s3://{BUCKET_NAME}/model-artifacts/" # Where SageMaker saves the final model

# SageMaker Role ARN
SAGEMAKER_ROLE_ARN = "arn:aws:iam::443630454547:role/service-role/AmazonSageMaker-ExecutionRole-20250710T140390"

def decide_branching(**kwargs):
    """
    Reads a flag file from the previous task to decide whether to retrain.
    The drift detection script is expected to create a file named 'drift_detected.txt'.
    """
    # In a real MWAA environment, you'd use XComs to pass this value.
    # For simplicity with BashOperator, we use a flag file.
    # The check_drift.py script will write "true" or "false" to this file.
    # This function would need to read that flag from a shared location like S3.
    # For this example, we'll assume the drift script sets an XCom value.
    ti = kwargs['ti']
    drift_detected = ti.xcom_pull(task_ids='detect_data_drift', key='drift_detected')
    
    if drift_detected == 'true':
        return 'trigger_sagemaker_retraining'
    else:
        return 'skip_retraining'

# --- SageMaker Training Job Configuration ---
training_job_config = {
    "AlgorithmSpecification": {
        "TrainingImage": "683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        "TrainingInputMode": "File",
    },
    "RoleArn": SAGEMAKER_ROLE_ARN,
    "OutputDataConfig": {"S3OutputPath": MODEL_OUTPUT_PATH},
    "ResourceConfig": {
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 10,
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    "HyperParameters": {
        "sagemaker_program": "train_aws.py",
        "sagemaker_submit_directory": CODE_URI,
        # Pass Redshift credentials via Airflow Connections, not hardcoded
    },
    "VpcConfig": {
        "SecurityGroupIds": ["sg-0c1a7cffa8cbd775c"],
        "Subnets": ["subnet-02568e11d9a9d81d9", "subnet-05c554493e8ab9bea", "subnet-0315d3065cef77e11"],
    },
}

# --- The Airflow DAG Definition ---
with DAG(
    dag_id='lead_conversion_retraining_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'sagemaker', 'lead-conversion'],
) as dag:

    start = EmptyOperator(task_id='start')

    # This task runs the drift detection script.
    # It assumes check_drift.py is in the dags/scripts folder of your S3 bucket.
    detect_data_drift = BashOperator(
        task_id='detect_data_drift',
        bash_command=(
            "pip install evidently pandas boto3 "
            "&& python /usr/local/airflow/dags/scripts/check_drift.py"
        )
    )

    branching = BranchPythonOperator(
        task_id='branch_on_drift_result',
        python_callable=decide_branching,
    )

    trigger_sagemaker_retraining = SageMakerTrainingOperator(
        task_id='trigger_sagemaker_retraining',
        config=training_job_config,
        aws_conn_id="aws_default",
        wait_for_completion=True,
    )

    skip_retraining = EmptyOperator(task_id='skip_retraining')
    
    end = EmptyOperator(task_id='end', trigger_rule='none_failed_min_one_success')

    start >> detect_data_drift >> branching
    branching >> [trigger_sagemaker_retraining, skip_retraining]
    trigger_sagemaker_retraining >> end
    skip_retraining >> end

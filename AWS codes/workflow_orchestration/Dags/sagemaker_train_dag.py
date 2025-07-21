from airflow import DAG
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1)
}

dag = DAG(
    dag_id='sagemaker_training_from_redshift',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)

training_job_name = "lead-conversion-job"

training_config = {
    "AlgorithmSpecification": {
        "TrainingInputMode": "File",
        "TrainingImage": "811284229777.dkr.ecr.ap-south-1.amazonaws.com/xgboost:1.3-1"
    },

    "RoleArn": "arn:aws:iam::443630454547:role/service-role/AmazonSageMaker-ExecutionRole-20250710T140390",

    "OutputDataConfig": {
        "S3OutputPath": "s3://sagemakerbucket2163/outputs/"
    },

    "ResourceConfig": {
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 5
    },

    "StoppingCondition": {
        "MaxRuntimeInSeconds": 3600
    },

    "TrainingJobName": training_job_name,

    "HyperParameters": {
        "sagemaker_program": "train_AWS.py",
        "sagemaker_submit_directory": "s3://sagemakerbucket2163/code/train.tar.gz",
        "sagemaker_container_log_level": "20",
        "sagemaker_region": "ap-south-1",
        "sagemaker_job_name": training_job_name,

        # ✅ Replacing ENV vars with hyperparameters
        "PREPROCESSING_PIPELINE_URI": "s3://sagemakerbucket2163/artifacts/preprocessing_pipeline.pkl",
        "REDSHIFT_HOST": "capstonetest.443630454547.ap-south-1.redshift-serverless.amazonaws.com",
        "REDSHIFT_PORT": "5439",
        "REDSHIFT_DB": "dev",
        "REDSHIFT_USER": "yash",
        "REDSHIFT_PASSWORD": "Yashrajput_2163"
    },

    "EnableNetworkIsolation": False,

    "VpcConfig": {
        "SecurityGroupIds": ["sg-0c1a7cffa8cbd775c"],
        "Subnets": [
            "subnet-02568e11d9a9d81d9",
            "subnet-05c554493e8ab9bea",
            "subnet-0315d3065cef77e11"
        ]
    }
}

sagemaker_training = SageMakerTrainingOperator(
    task_id="run_sagemaker_training",
    config=training_config,
    aws_conn_id="aws_default",
    wait_for_completion=True,
    dag=dag
)

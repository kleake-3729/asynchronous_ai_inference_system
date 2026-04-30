from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException
from datetime import datetime
import sys, os
import json
import joblib
import boto3

# S3 bucket variables 
BUCKET_NAME = "kl-hw-3-bucket" 

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.breast_cancer_data import load_data
from ml_pipeline.breast_cancer_model import train_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Pipeline: train model -> evaluate model -> promote model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper(data_path: str, model_path: str, metadata_path: str, **kwargs):
        df = load_data(data_path)
        date_time = datetime.now()
        run_id = kwargs['run_id']
        model_version = run_id
        acc = train_model(df, model_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Load existing metrics or initialize
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new accuracy
        data.append({"model_version": model_version, "dataset": data_path, "model_type": "logistic_regression", "accuracy": acc})

        # Save back to file
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[train_model] Saved metadata to {metadata_path}")

        return acc
        
    def eval_model_wrapper(model_path: str, metrics_path: str, **kwargs):

        ti = kwargs["ti"]

        # Pull accuracy from train task (XCom)
        acc = ti.xcom_pull(task_ids="train_model")

        # Load trained model
        clf = joblib.load(model_path)

        print(f"[eval_model] Loaded model from {model_path}")
        print(f"[eval_model] Accuracy from training: {acc:.4f}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Load existing metrics or initialize
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new accuracy
        data.append({"accuracy": acc})

        # Save back to file
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[eval_model] Saved accuracy to {metrics_path}")

        return acc
        
    def promote_model_wrapper(metrics_path:str, metadata_path: str, model_path: str, **kwargs):
        
        ti = kwargs["ti"]
        
        run_id = kwargs['run_id']

        # Pull accuracy from train task (XCom)
        acc = ti.xcom_pull(task_ids="train_model")
        
        threshold = 0.94
        
        if acc >= threshold: 
            s3 = boto3.client("s3")
            metrics_key = f"models/{run_id}/metrics.json"
            metadata_key = f"models/{run_id}/metadata.json"
            model_key = f"models/{run_id}/breast_cancer.pkl"
            s3.upload_file(metrics_path, BUCKET_NAME, metrics_key)
            s3.upload_file(metadata_path, BUCKET_NAME, metadata_key)
            s3.upload_file(model_path, BUCKET_NAME, model_key)
        
        else: 
            raise AirflowFailException(f"Model accuracy metric {acc} below threshold {threshold}")
        
        return acc
        
        

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "model_path": "models/breast_cancer.pkl", 
            "metadata_path": "models/metadata.json"
        },
    )
    
    eval_task = PythonOperator(
        task_id="eval_model",
        python_callable=eval_model_wrapper,
        op_kwargs={
            "model_path": "models/breast_cancer.pkl",
            "metrics_path": "models/metrics.json",
    },
    )
    
    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
        op_kwargs={
            "model_path": "models/breast_cancer.pkl",
            "metrics_path": "models/metrics.json",
            "metadata_path": "models/metadata.json",
    },
    )

    train_task >> eval_task >> promote_task
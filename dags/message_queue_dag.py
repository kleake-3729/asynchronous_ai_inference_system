from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException
from datetime import datetime
import sys, os
import json
import joblib
import boto3
import time
import uuid
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
sqs = boto3.client("sqs")
QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/006539580860/async-ai-system-queue"


# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.breast_cancer_data import load_data
from ml_pipeline.writer import get_test_data

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="message_queue_dag",
    default_args=default_args,
    description="Pipeline: add messages to queue",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def add_queue_message_wrapper(data_path: str, queue_url:str):
        df = load_data(data_path)
        X_test = get_test_data(df, queue_url)
        return X_test
        
        
        
        

    add_message_queue_task = PythonOperator(
        task_id="add_message_queue",
        python_callable=add_queue_message_wrapper,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "queue_url": QUEUE_URL, 
        },
    )
    

    add_message_queue_task 
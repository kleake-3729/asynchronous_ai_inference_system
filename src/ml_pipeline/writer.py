import json
import time
import uuid
import boto3
import argparse
import os
import joblib
import pandas as pd
#import settings
from sklearn.model_selection import train_test_split
#from breast_cancer_data import load_data

sqs = boto3.client("sqs")


def get_test_data(df, queue_url):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for index, row in X_test.iterrows():
        row_info = {"record_id": str(index), "features": str(row)}
        
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(row_info))
        print(f"Sent {index}")



#if __name__ == "__main__":
    #df = load_data("data/breast_cancer.csv")
    #get_test_data(df, settings.QUEUE_URL)
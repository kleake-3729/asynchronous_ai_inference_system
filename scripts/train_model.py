import sys, os

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.breast_cancer_data import load_data
from ml_pipeline.breast_cancer_model import train_model

if __name__ == "__main__":
    df = load_data("data/breast_cancer.csv")
    train_model(df, "models/breast_cancer.pkl", "data/test_data", "test", "models/metadata.json")
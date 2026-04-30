# src/app/api.py
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
# Explicit request schema for Iris dataset (4 features)
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
# Explicit request schema for breast cancer dataset 
class BreastCancerRequest(BaseModel):
    mean_radius: float
    mean_texture: float 
    mean_perimeter: float 
    mean_area:float
    mean_smoothness:float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float 
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float 
    perimeter_error: float
    area_error: float
    smoothness_error: float 
    compactness_error:float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter:float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

def create_app(model_path: str = "models/breast_cancer.pkl", metadata_path: str = "models/metadata.json"):
    """
    Creates a FastAPI app that serves predictions for the breast cancer model.

    """
    # Helpful guard so students get a clear error if they forgot to train first
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the DAG or scripts/train_model.py)."
        )

    model = joblib.load(model_path)
    app = FastAPI(title="Breast Cancer Model API")

    # Map numeric predictions to class names
    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {
            "message": "Breast cancer model is ready for inference!",
            "classes": target_names,
        }
    
    @app.get("/model/info")
    def get_metrics():
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new accuracy
        current_model = data[-1]
        return {
            "message": "The current model metadata information is below!",
            "model version": current_model,
        }

    @app.post("/predict")
    def predict(request: BreastCancerRequest):
        # Convert request into the correct shape (1 x 4)
        X = np.array([
            [request.mean_radius, request.mean_texture,request.mean_perimeter, request.mean_area, 
            request.mean_smoothness,request.mean_compactness, request.mean_concavity, request.mean_concave_points, request.mean_symmetry,
            request.mean_fractal_dimension,request.radius_error, request.texture_error, request.perimeter_error, request.area_error,
            request.smoothness_error, request.compactness_error, request.concavity_error, request.concave_points_error, request.symmetry_error, 
            request.fractal_dimension_error, request.worst_radius, request.worst_texture, request.worst_perimeter, request.worst_area, request.worst_smoothness, 
            request.worst_compactness,request.worst_concavity, request.worst_concave_points, request.worst_symmetry, request.worst_fractal_dimension]
        ])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}

    # return the FastAPI app
    return app

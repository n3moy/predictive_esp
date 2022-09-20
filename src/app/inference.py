import os
import click
import yaml
import joblib
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException


load_dotenv()
app = FastAPI()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")


class Model:
    def __init__(self, model_name: str, model_stage: str):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

    def predict(self, data: pd.DataFrame):
        predictions = self.model.predict(data)
        return predictions


model = Model("LogisticRegression", "Staging")


@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):

    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        return list(model.predict(data))
    else:
        raise HTTPException(
            status_code=400,
            detail="Incorrect file format")


if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    exit(1)

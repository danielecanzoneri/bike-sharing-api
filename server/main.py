import io
import os
import pandas as pd
from typing import Annotated

from fastapi import FastAPI, UploadFile, HTTPException
from sqlalchemy import create_engine

from model import load_model, save_model
from model import train as model_train

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError("MODEL_PATH environment variable not set")

# Load the model if it exists
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None

app = FastAPI()
engine = create_engine(DATABASE_URL, echo=True)


@app.post("/load")
def load_dataset(file: UploadFile):
    """Load a dataset from a CSV file and store it in the database."""

    # Check that file is a CSV
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=415, detail="Only CSV files are allowed.")
    
    contents = file.file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the file: {e}")
    
    # Store the DataFrame in the database
    try:
        df.to_sql('bike_sharing', con=engine, if_exists='replace', index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing the data in the database: {e}")


@app.get("/train")
def train_model():
    """Train the model using the dataset stored in the database. The model used is a simple Decision Tree Regressor."""
    
    try:
        df = pd.read_sql_table('bike_sharing', con=engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset not present: {e}")

    try:
        global model  # Use the global model variable to store the trained model

        X, y = model_train.preprocess_dataset(df)
        model = model_train.train(X, y)

        # Save the trained model to a file
        save_model(model, MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {e}")
    
    return {"message": "Model trained successfully"}
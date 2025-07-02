import io
import os
from enum import Enum
import pandas as pd
from typing import Annotated

from fastapi import FastAPI, UploadFile, HTTPException
from sqlalchemy import create_engine, text

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


class AvgStatsType(str, Enum):
    hr = "hour"
    weekday = "day"

@app.get("/stats/avg")
def average(type: AvgStatsType = AvgStatsType.hr):
    """Get aggregated statistics (average users for each hour/day)"""

    # Validate column exists in the table
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'bike_sharing' AND column_name = :col
            """),
            {"col": type.name}
        )
        if not result.fetchone():
            raise HTTPException(status_code=400, detail=f"Dataset does not contain '{type.name}' column.")

        # Compute average using SQL
        avg_query = text(f"""
            SELECT {type.name}, AVG(cnt) as avg_users
            FROM bike_sharing
            GROUP BY {type.name}
            ORDER BY {type.name}
        """)
        avg_result = conn.execute(avg_query)
        avg_users = {row[0]: round(row[1], 4) for row in avg_result}

    if type == AvgStatsType.weekday:
        # Convert the weekday number to a string representation
        weekday_map = {
            0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
            4: 'Thursday', 5: 'Friday', 6: 'Saturday',
        }
        avg_users = {weekday_map[k]: v for k, v in avg_users.items()}

    return avg_users
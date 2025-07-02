import io
import os
from enum import Enum
import uuid
import pandas as pd
from typing import Annotated

from fastapi import FastAPI, Response, UploadFile, HTTPException
from sqlalchemy import create_engine, text

from model import load_model, save_model, preprocess_dataset
from model import train as model_train

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

MODEL_PATH = "/model-data/model.pkl"

# Load the model if it exists
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None

app = FastAPI()
engine = create_engine(DATABASE_URL, echo=True)


def parse_csv(file):
    # Check that file is a CSV
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=415, detail="Only CSV files are allowed.")
    
    contents = file.file.read()
    try:
        return pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the file: {e}")

@app.post("/load")
def load_dataset(file: UploadFile):
    """Load a dataset from a CSV file and store it in the database."""
    
    df = parse_csv(file)

    # Store the DataFrame in the database
    try:
        df.to_sql('bike_sharing', con=engine, if_exists='replace', index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing the data in the database: {e}")
    
    return {"message": "Dataset loaded successfully."}


@app.get("/train")
def train_model():
    """Train the model using the dataset stored in the database. The model used is a simple Decision Tree Regressor."""
    
    try:
        df = pd.read_sql_table('bike_sharing', con=engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset not present: {e}")

    try:
        global model  # Use the global model variable to store the trained model

        X = preprocess_dataset(df)
        y = df["cnt"].copy()
        model = model_train.train(X, y)

        # Save the trained model to a file
        save_model(model, MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {e}")
    
    return {"message": "Model trained successfully."}


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


@app.get("/stats/total")
def total_users():
    """Get total number of users for each month."""

    with engine.connect() as conn:
        # Compute total using SQL
        total_query = text("""
            SELECT mnth, SUM(cnt) as total_users
            FROM bike_sharing
            GROUP BY mnth
            ORDER BY mnth
        """)
        total_result = conn.execute(total_query)

        # Convert the month number to a string representation
        month_map = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December',
        }
        total_users = {month_map[row[0]]: row[1] for row in total_result}

    return total_users


predictions = {}

@app.post("/predict")
def predict(file: UploadFile):
    """Predict the number of users given a CSV file with the same columns as the dataset."""

    if not model:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    
    df = parse_csv(file)

    # Predict the number of users
    try:
        X = preprocess_dataset(df)
        y = model.predict(X.values)
        print(predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    # Add the predictions to the DataFrame
    df["cnt"] = y

    # Generate an id for the prediction and store it in the predictions dictionary
    prediction_id = str(uuid.uuid4())
    predictions[prediction_id] = df

    # Return the prediction id
    return {"prediction_id": prediction_id}


@app.get("/predict/{prediction_id}")
def get_prediction(prediction_id: str):
    """Get the prediction for a given prediction id."""

    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    return Response(content=predictions[prediction_id].to_csv(index=False), media_type="text/csv")
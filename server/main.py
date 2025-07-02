import io
import os
import pandas as pd
from typing import Annotated

from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")


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

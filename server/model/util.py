import joblib
import pandas as pd

def load_model(path: str):
    """
    Load a trained model from a file.
    Args:
        path (str): The file path from which to load the model.
    Returns:
        model: The loaded model.
    """
    
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    
    return model

def save_model(model, path: str):
    """
    Save the trained model to a file.
    Args:
        model: The trained model to save.
        path (str): The file path where the model will be saved.
    """
    
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def preprocess_dataset(df: pd.DataFrame):
    """
    Preprocess the dataset by removing unnecessary attributes.
    Args:
        df (pd.DataFrame): The input DataFrame containing the bike sharing dataset.
    Returns:
        X (pd.DataFrame): The preprocessed DataFrame with unnecessary columns removed.
    """

    # There are two datasets: one with data aggregated by hours and another by day.
    columns = [
        "season", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    if not "hr" in df.columns: # Day dataset
        columns.remove("hr")
    
    if not all(col in df.columns for col in columns):
        raise ValueError("Dataset does not contain all required columns.")
    
    return df[columns].copy()
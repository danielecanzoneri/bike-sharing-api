import joblib

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
import pandas as pd


def preprocess_dataset(df: pd.DataFrame):
    """
    Preprocess the dataset by removing unnecessary attributes.
    Args:
        df (pd.DataFrame): The input DataFrame containing the bike sharing dataset.
    Returns:
        X (pd.DataFrame): The preprocessed DataFrame with unnecessary columns removed.
        y (pd.Series): The target variable (count of bike rentals).
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
    
    X = df[columns].copy()
    y = df["cnt"].copy()
    
    return X, y


def train(X: pd.DataFrame, y: pd.Series):
    """
    Train a simple Decision Tree Regressor model.
    Args:
        X (pd.DataFrame): The input features for training.
        y (pd.Series): The target variable for training.
    Returns:
        model: The trained Decision Tree Regressor model.
    """
    from sklearn.tree import DecisionTreeRegressor
    
    model = DecisionTreeRegressor(
        max_depth=10, min_samples_leaf=3, min_samples_split=10, # Best parameters found by grid search
        random_state=42,
    )
    model.fit(X, y)
    
    return model

import pandas as pd


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

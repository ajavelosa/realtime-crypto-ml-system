import pandas as pd


class BaselineModel:
    def __init__(self):
        """
        Initialize the baseline model.

        Args:
            X: The features.
            y: The target.
        """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the baseline model. We will pass since the baseline model
        will not be based on machine learning.
        """
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target using the baseline model.

        Args:
            X: The features.

        Returns:
            The predicted target.
        """
        return X['close']

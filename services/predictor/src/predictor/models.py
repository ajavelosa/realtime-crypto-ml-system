import os

import mlflow
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_absolute_error


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
        Predict the target using the baseline model. The baseline model
        will predict the future close price as the current close price.

        Args:
            X: The features.

        Returns:
            The predicted target.
        """
        return X['close']


def get_model_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_candidates: int,
) -> list[str]:
    """
    Uses lazypredict to fit N models with default hyperparameters for the given
    (X_train, y_train), and evaluate them with (X_test, y_test)

    It returns a list of model names, from best to worst.

    Args:
        X_train: pd.DataFrame, the training data
        y_train: pd.Series, the target variable
        X_test: pd.DataFrame, the test data
        y_test: pd.Series, the target variable
        n_candidates: int, the number of candidates to return

    Returns:
        list[str], the list of model names from best to worst
    """
    # unset the MLFLOW_TRACKING_URI
    # This is a temporary hack to avoid LazyPredict from
    # setting its own MLFLOW_TRACKING_URI. We want to use
    # the parent MLFlow run's tracking URI.
    # TODO: find a better way to do this.
    mlflow_tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    del os.environ['MLFLOW_TRACKING_URI']

    # fit N models with default hyperparameters
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=False,
        custom_metric=mean_absolute_error,
    )
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    # reset the index so that the model names are in the first column
    models.reset_index(inplace=True)

    # log table to mlflow experiment
    mlflow.log_table(models, 'model_candidates_with_default_hyperparameters.json')

    # set the MLFLOW_TRACKING_URI back to its original value
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

    # list of top n_candidates model names
    model_candidates = models['Model'].tolist()[:n_candidates]

    return model_candidates

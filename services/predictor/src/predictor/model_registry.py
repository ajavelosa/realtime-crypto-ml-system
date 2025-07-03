"""
Model registry for the predictor service.

This module provides functionality to load and push models to the MLflow model registry.

Key Features:
- Loading models from the MLflow model registry
- Pushing models to the MLflow model registry
- Inferring model signatures
"""

from typing import Optional

import mlflow
import pandas as pd
from loguru import logger
from mlflow.models import Model
from mlflow.tracking import MlflowClient


def get_model_name(
    pair: str,
    candle_seconds: int,
    prediction_horizon_seconds: int,
) -> str:
    """
    Get the name of the model in the MLFlow model registry.
    """
    return f'{pair.replace("/", "-")}_{candle_seconds}s_{prediction_horizon_seconds}s'

# TODO: Create a custom type to annotate the output of this function
def load_model(
    model_name: str,
    model_version: Optional[str] = "latest",
) -> tuple[Model, list[str], str]:
    """
    Load the model from the MLFlow model registry.
    """
    model = mlflow.sklearn.load_model(
        model_uri=f'models:/{model_name}/{model_version}'
    )

    # Get the model info which contains the signature to extract the features.
    model_info = mlflow.models.get_model_info(
        model_uri=f'models:/{model_name}/{model_version}'
    )

    features = model_info.signature.inputs.input_names()

    client = MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=['None'])[0].version

    return model, features, model_version

def push_model(
    model,
    X_test: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Pushes the given `model` to the MLflow model registry using the given `model_name`.

    Args:
        model: The model to push to the model registry.
        X_test: The test data to use to infer the model signature.
        model_name: The name of the model to push to the model registry.
    """
    # Infer the model signature
    y_pred = model.predict(X_test)
    signature = mlflow.models.infer_signature(X_test, y_pred)

    # Log the sklearn model and register as version 1
    logger.info(f'Pushing model {model_name} to the model registry')
    mlflow.sklearn.log_model(  # type: ignore
        sk_model=model,
        name=model_name,
        signature=signature,
        registered_model_name=model_name,
    )
    logger.info(f'Model {model_name} pushed to the model registry')

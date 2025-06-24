import mlflow
import pandas as pd
from loguru import logger
from mlflow.models.signature import infer_signature


def get_model_name(
    pair: str,
    candle_seconds: int,
    prediction_horizon_seconds: int,
) -> str:
    """
    Get the name of the model in the MLFlow model registry.
    """
    return f'{pair.replace("/", "-")}_{candle_seconds}s_{prediction_horizon_seconds}s'


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
    signature = infer_signature(X_test, y_pred)

    # Log the sklearn model and register as version 1
    logger.info(f'Pushing model {model_name} to the model registry')
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        signature=signature,
        registered_model_name=model_name,
    )
    logger.info(f'Model {model_name} pushed to the model registry')

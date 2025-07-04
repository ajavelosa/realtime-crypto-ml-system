"""
Model registry for the predictor service.

This module provides functionality to load and push models to the MLflow model registry.

Key Features:
- Loading models from the MLflow model registry
- Pushing models to the MLflow model registry
- Inferring model signatures
"""

from typing import Any, Optional

import mlflow
import mlflow.exceptions
import mlflow.models
import mlflow.sklearn
import pandas as pd
from loguru import logger
from mlflow.models import MetricThreshold, Model
from mlflow.tracking import MlflowClient

client = MlflowClient()

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
    model_alias: str = 'champion',
) -> tuple[Model, list[str], str]:
    """
    Load the model from the MLFlow model registry.

    Args:
        model_name: Name of the model in the registry
        model_alias: Alias to use (defaults to 'champion', falls back to 'latest')

    Returns:
        tuple: (model, features, model_version)
    """
    # Try to get the model by alias, otherwise use latest
    try:
        model_version_info = client.get_model_version_by_alias(
            name=model_name,
            alias=model_alias,
        )
        model_version = model_version_info.version
        logger.info(f'Loaded model {model_name} with alias "{model_alias}" (version {model_version})')
    except Exception as e:
        logger.warning(f'No model found with alias {model_alias} for {model_name}. Using latest. {e}')
        model_version = 'latest'

    # Load the model and get its features
    model_uri = f'models:/{model_name}/{model_version}'
    model = mlflow.sklearn.load_model(model_uri)  # type: ignore

    # Get features from model signature
    model_info = mlflow.models.get_model_info(model_uri)  # type: ignore
    features = model_info.signature.inputs.input_names()

    return model, features, str(model_version)  # type: ignore

def get_champion_model_mae(
    model_name: str,
    alias: str,
) -> Optional[float]:
    """
    Get the champion model from the MLFlow model registry.
    Returns None if no champion model exists yet.
    """
    try:
        champion_model = client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )
    except Exception as e:
        logger.warning(f'No champion model found for {model_name}. {e}')
        return None

    champion_model_mae = client.get_metric_history(
        run_id=champion_model.run_id,  # type: ignore
        key='mean_absolute_error',
    )

    return champion_model_mae[-1].value

def validate_and_push_model_to_registry(
    model_info: Any,  # MLflow model info object
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_mae_baseline: float,
    max_percent_diff_wrt_baseline: float,
    model_name: str,
    pair: str,
) -> None:
    """
    Validate the model and update the champion model if it performs better.

    Args:
        model_info: MLflow model info object
        X_test: Test features
        y_test: Test target
        test_mae_baseline: Baseline model MAE
        max_percent_diff_wrt_baseline: Maximum allowed percentage difference from baseline
        model_name: Name of the model
        pair: Trading pair

    Returns:
        None
    """

    # Evaluate the model and generate model metrics
    result = mlflow.evaluate(
        model=model_info.model_uri,
        # Combine X_test and y_test into a single DataFrame because the
        # function expects a single DataFrame with a target column.
        data=pd.concat([X_test, y_test], axis=1),
        targets='target',
        model_type='regressor',
        evaluators='default',
        evaluator_config={
            'default': {}
        },
    )

    try:
        mlflow.validate_evaluation_results(
            candidate_result=result,
            validation_thresholds={
                'mean_absolute_error': MetricThreshold(
                    # The threshold is the maximum percentage difference
                    # between the model's MAE and the baseline MAE.
                    # We add 1 because the metric is a percentage difference
                    # and we want to allow for a 10% difference.
                    threshold=(1 + max_percent_diff_wrt_baseline) * test_mae_baseline,
                    greater_is_better=False,
                ),
            }
        )
        is_model_passing_validation = True

    except mlflow.exceptions.MlflowException as e:
        logger.error(f'Model validation failed: {e}')
        is_model_passing_validation = False

    mlflow.set_tags({'passed_validation': str(is_model_passing_validation)})

    new_model_mae = result.metrics["mean_absolute_error"]

    # Log the model validation results
    logger.info(f"Model {model_name} for {pair} validation: {'PASSED' if is_model_passing_validation else 'FAILED'} (MAE: {new_model_mae:.4f} vs baseline: {test_mae_baseline:.4f})")

    # If the model scores better than the best model, update the best model
    champion_model_mae = get_champion_model_mae(
        model_name=model_name,
        alias='champion',
    )

    # If no champion model exists yet, or if the new model is better, update the champion
    if champion_model_mae is None or new_model_mae < champion_model_mae:
        logger.info(f'Updating the champion model to {model_name}')
        mv = client.create_model_version(
            name=model_name,
            source=model_info.model_uri,
            run_id=model_info.run_id,
        )

        client.set_registered_model_alias(
            name=model_name,
            alias='champion',
            version=mv.version,
        )

        logger.info(f'Successfully set alias "champion" for model {model_name} version {mv.version}')

        if champion_model_mae is None:
            logger.info(f"Set first champion model: {model_name} version {mv.version} (MAE: {new_model_mae:.4f})")
        else:
            improvement = (champion_model_mae / new_model_mae - 1) * 100
            logger.info(f"Updated champion model: {model_name} version {mv.version} (MAE: {new_model_mae:.4f} vs {champion_model_mae:.4f}, {improvement:.2f}% improvement)")

    else:
        logger.info(f'{model_name} is not better than the current champion (MAE: {new_model_mae:.4f} >= {champion_model_mae:.4f})')

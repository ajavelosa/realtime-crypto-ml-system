"""
The training script for the predictor service.

Has the following steps:
1. Load data from RisingWave
2. Add a target column
3. Validate the data
4. Profile the data
5. Split the data into train and test
6. Split features and target
7. Train a dummy baseline model
8. Find the best model candidates
9. Train the best model with hyperparameter search
10. Validate the model
11. Push the model to the model registry
"""

from typing import Optional

import mlflow
import mlflow.data
import mlflow.exceptions
import mlflow.models
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from risingwave import OutputFormat, RisingWave, RisingWaveConnOptions
from sklearn.metrics import mean_absolute_error
from ydata_profiling import ProfileReport

from predictor.config import training_config
from predictor.data_validation import validate_data
from predictor.model_registry import get_model_name, validate_and_push_model_to_registry
from predictor.models import BaselineModel, get_model_candidates, get_model_object

client = MlflowClient()

def load_ts_data_from_risingwave(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    pair: str,
    training_set_size_days: int,
    candle_seconds: int,
    table: str,
) -> pd.DataFrame:
    """
    Fetches technical indicators from RisingWave for the given pair and time range.

    Args:
        - host: The host of the RisingWave instance.
        - port: The port of the RisingWave instance.
        - user: The user of the RisingWave instance.
        - password: The password of the RisingWave instance.
        - database: The database of the RisingWave instance.
        - pair: The pair of the technical indicators to fetch.
        - training_set_size_days: The number of days in the past to fetch the technical indicators for.
        - candle_seconds: The number of seconds in the candle.

    Returns:
        - A pandas DataFrame with the technical indicators.
    """
    logger.info(f'Connecting to RisingWave: {host}:{port}/{database}')
    rw = RisingWave(
        RisingWaveConnOptions.from_connection_info(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
    )
    query = f"""
        SELECT *

        FROM {table}

        WHERE pair = '{pair}'
            AND TO_TIMESTAMP(window_start_ms / 1000) >= NOW() - INTERVAL '{training_set_size_days} DAYS'
            AND candle_seconds = {candle_seconds}
    """

    ts_data = rw.fetch(query, format=OutputFormat.DATAFRAME)
    assert isinstance(ts_data, pd.DataFrame), 'Expected DataFrame from RisingWave'
    logger.info(
        f'Fetched {len(ts_data)} rows for {pair} in the last {training_set_size_days} days from RisingWave'
    )
    return ts_data


def generate_exploratory_data_analysis_report(
    ts_data: pd.DataFrame,
    output_html_path: str,
) -> None:
    """
    Genearates an HTML file exploratory data analysis charts for the given `ts_data` and
    saves it locally to the given `output_html_path`

    Args:
        ts_data: The technical indicators data to profile.
        output_html_path: The path to save the HTML file to.

    Returns:
        None. Writes the HTML file to the given `output_html_path`.
    """
    logger.info('Generating exploratory data analysis report...')
    profile = ProfileReport(
        ts_data,
        tsmode=True,
        sortby='window_start_ms',
        title='Technical indicators EDA',
        minimal=True,
    )
    profile.to_file(output_html_path)


def train(
    mlflow_tracking_uri: str,
    risingwave_host: str,
    risingwave_port: int,
    risingwave_user: str,
    risingwave_password: str,
    risingwave_database: str,
    risingwave_table: str,
    pair: str,
    training_set_size_days: int,
    candle_seconds: int,
    prediction_horizon_seconds: int,
    output_html_path: str,
    train_test_split_ratio: float,
    n_model_candidates: int,
    features: list[str],
    n_rows_to_profile: Optional[int] = None,
    model_name: Optional[str] = None,
    hyperparam_search_trials: Optional[int] = None,
    hyperparam_splits: Optional[int] = None,
    max_percent_diff_wrt_baseline: float = 0.10,
    max_percentage_rows_with_null_values: float = 0.05,
) -> None:
    """
    Train a predictor model for the given pair and data. If the model is good, push it
    to the MLFlow model registry.

    Args:
        mlflow_tracking_uri: The URI of the MLFlow server.
        risingwave_host: The host of the RisingWave instance.
        risingwave_port: The port of the RisingWave instance.
        risingwave_user: The user of the RisingWave instance.
        risingwave_password: The password of the RisingWave instance.
        risingwave_database: The database of the RisingWave instance.
        risingwave_table: The table to fetch the technical indicators from.
        pair: The pair of the technical indicators to fetch.
        training_set_size_days: The number of days in the past to fetch the technical indicators for.
        candle_seconds: The number of seconds in the candle.
        prediction_horizon_seconds: The number of seconds in the prediction horizon.
        output_html_path: The path to save the HTML file to.
        train_test_split_ratio: The ratio of the training set to the test set.
        n_model_candidates: The number of model candidates to find.
        n_rows_to_profile: The number of rows to profile.
        model_name: The name of the model to train. If None, we will find the best model candidates.
    Returns:
        None. Writes the model parameters, metrics, and artifacts to MLFlow.
    """
    logger.info(
        f'Starting training for {pair} for the last {training_set_size_days} days.'
    )

    # Set MLflow tracking URI
    logger.info(f'Setting MLFlow tracking URI to {mlflow_tracking_uri}.')
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    logger.info('Setting MLFlow experiment')
    mlflow.set_experiment(
        get_model_name(
            pair=pair,
            candle_seconds=candle_seconds,
            prediction_horizon_seconds=prediction_horizon_seconds,
        )
    )

    # Things we want to log to MLFlow:
    # - Training data
    # - Parameters
    # - EDA report
    # - Model performance metrics

    with mlflow.start_run():
        logger.info('Starting training run...')

        # Input to the training process
        mlflow.log_param('features', features)
        mlflow.log_param('pair', pair)
        mlflow.log_param('training_data_horizon_days', training_set_size_days)
        mlflow.log_param('candle_seconds', candle_seconds)
        mlflow.log_param('prediction_horizon_seconds', prediction_horizon_seconds)
        mlflow.log_param('train_test_split_ratio', train_test_split_ratio)
        mlflow.log_param('data_profiling_n_rows', n_rows_to_profile)
        if model_name:
            mlflow.log_param('model_name', model_name)
        mlflow.log_param(
            'max_percentage_diff_mae_wrt_baseline', max_percent_diff_wrt_baseline
        )

        # Step 1: Load data from RisingWave
        logger.info('Loading data from RisingWave...')
        ts_data = load_ts_data_from_risingwave(
            host=risingwave_host,
            port=risingwave_port,
            user=risingwave_user,
            password=risingwave_password,
            database=risingwave_database,
            pair=pair,
            training_set_size_days=training_set_size_days,
            candle_seconds=candle_seconds,
            table=risingwave_table,
        )

        # Only keep the features we want to use
        ts_data = ts_data[features]

        # Ensure ts_data is a DataFrame (convert from numpy array if needed)
        if not isinstance(ts_data, pd.DataFrame):
            ts_data = pd.DataFrame(ts_data)

        # Step 2: Add a target column
        ts_data['target'] = ts_data['close'].shift(
            -prediction_horizon_seconds // candle_seconds
        )

        # Drop rows with missing values
        ts_data = ts_data.dropna(subset=['target'])

        # Log training dataset to MLFlow
        dataset = mlflow.data.from_pandas(ts_data)  # type: ignore[attr-defined]
        mlflow.log_input(dataset, context='training')

        # Log dataset size
        mlflow.log_param('ts_data_shape', ts_data.shape)

        # Step 3: Validate the data
        ts_data = validate_data(
            ts_data=ts_data,
            prediction_horizon_seconds=prediction_horizon_seconds,
            candle_seconds=candle_seconds,
            max_percentage_rows_with_null_values=max_percentage_rows_with_null_values,
        )

        # Step 4: Profile the data
        ts_data_to_profile = (
            ts_data.head(n_rows_to_profile) if n_rows_to_profile else ts_data
        )
        logger.info('Generating EDA report...')
        generate_exploratory_data_analysis_report(
            ts_data=ts_data_to_profile,
            output_html_path=output_html_path,
        )
        logger.info('Logging EDA report to MLFlow...')
        mlflow.log_artifact(local_path=output_html_path, artifact_path='eda_report')
        logger.info('EDA report logged to MLFlow.')

        # Step 5: Split the data into train and test

        # We want to split the data such that earlier data is in the training set
        # and later data is in the test set. We do this because we want to predict
        # the future price of the asset, and we want to use the past data to train
        # the model.

        logger.info('Splitting data into train and test...')
        train_data_size = int(len(ts_data) * train_test_split_ratio)
        train_data = ts_data.iloc[:train_data_size]
        test_data = ts_data.iloc[train_data_size:]

        # Log train and test data size
        logger.info('Logging train and test data shape to MLFlow...')
        mlflow.log_param('train_data_shape', train_data.shape)
        mlflow.log_param('test_data_shape', test_data.shape)

        # Step 6: Split features and target
        logger.info('Splitting features and target...')
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        mlflow.log_param('X_train_shape', X_train.shape)
        mlflow.log_param('y_train_shape', y_train.shape)
        mlflow.log_param('X_test_shape', X_test.shape)
        mlflow.log_param('y_test_shape', y_test.shape)

        # Log train and test data to MLFlow
        train_dataset = mlflow.data.from_pandas(X_train)  # type: ignore[attr-defined]
        mlflow.log_input(train_dataset, context='training')
        test_dataset = mlflow.data.from_pandas(X_test)  # type: ignore[attr-defined]
        mlflow.log_input(test_dataset, context='test')

        # Step 7: Train a dummy baseline model
        logger.info('Creating a dummy baseline model...')

        baseline_model = BaselineModel()
        y_pred = baseline_model.predict(X_test)
        test_mae_baseline = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric('baseline_model_test_mae', test_mae_baseline)
        logger.info(f'Baseline model test MAE: {test_mae_baseline:.4f} for {pair}')

        # Step 8: Find the best model candidates, if model_name is not provided.
        if model_name is None:
            logger.info('Training a lazy model...')

            model_names = get_model_candidates(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                n_candidates=n_model_candidates,
            )

        # TODO: Train multiple models with the count
        # with the count being n_model_candidates
        # We need to split the test data further into
        # 2 sets so that we can validate the top models
        # against the first set and baseline the top
        # model (winner) against the second set.

        # Loop over the available models until we are
        # able to find one in our registry.
        for model_name in model_names:
            try:
                model = get_model_object(model_name)
            except NotImplementedError:
                logger.error(f'Model {model_name} not found. Choosing the next best model...')
                continue
            else:
                break

        # Step 9: Train the best model with hyperparameter search
        logger.info(f'Training the {model_name} model with hyperparameter search...')
        model.fit(
            X_train,
            y_train,
            hyperparam_search_trials=hyperparam_search_trials,
            hyperparam_splits=hyperparam_splits,
        )

        # Step 10: Validate the model
        logger.info(f'Validating the {model_name} model...')
        y_pred = model.predict(X_test)

        # Get the model name
        model_name = get_model_name(
            pair=pair,
            candle_seconds=candle_seconds,
            prediction_horizon_seconds=prediction_horizon_seconds,
        )

        # Infer signature and log the model
        signature = mlflow.models.infer_signature(X_test, y_pred)  # type: ignore

        model_info = mlflow.sklearn.log_model(  # type: ignore
            sk_model=model,
            name=model_name,
            signature=signature,
            registered_model_name=model_name,
        )

        # Step 11: Validate the model and update champion if better
        validate_and_push_model_to_registry(
            model_info=model_info,
            X_test=X_test,
            y_test=y_test,
            test_mae_baseline=test_mae_baseline,
            max_percent_diff_wrt_baseline=max_percent_diff_wrt_baseline,
            model_name=model_name,
            pair=pair,
        )


if __name__ == '__main__':
    train(
        mlflow_tracking_uri=training_config.mlflow_tracking_uri,
        risingwave_host=training_config.risingwave_host,
        risingwave_port=training_config.risingwave_port,
        risingwave_user=training_config.risingwave_user,
        risingwave_password=training_config.risingwave_password,
        risingwave_database=training_config.risingwave_database,
        risingwave_table=training_config.risingwave_table,
        pair=training_config.pair,
        training_set_size_days=training_config.training_set_size_days,
        candle_seconds=training_config.candle_seconds,
        prediction_horizon_seconds=training_config.prediction_horizon_seconds,
        output_html_path='./eda_report.html',
        n_rows_to_profile=training_config.n_rows_to_profile,
        train_test_split_ratio=training_config.train_test_split_ratio,
        n_model_candidates=training_config.n_model_candidates,
        features=training_config.features,
        hyperparam_search_trials=training_config.hyperparam_search_trials,
        hyperparam_splits=training_config.hyperparam_splits,
        max_percent_diff_wrt_baseline=training_config.max_percent_diff_wrt_baseline,
        max_percentage_rows_with_null_values=training_config.max_percentage_rows_with_null_values,
    )

"""
The training script for the predictor service.

Has the following steps:
1. Fetch data from RisingWave
2. Add a target column
3. Validate the data
4. Profile the data
5. Split the data into train and test
6. Create a baseline model
7. XGBoost model with default hyperparameters
8. Hyperparameter tuning with Optuna
9. Validate the final model
10. Push the model to MLFlow
11. Log the model on MLFlow
"""

from typing import Optional

import mlflow
import pandas as pd
from loguru import logger
from risingwave import OutputFormat, RisingWave, RisingWaveConnOptions
from ydata_profiling import ProfileReport

from predictor.data_validation import validate_data
from predictor.model_registry import get_model_name


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
    pair: str,
    training_set_size_days: int,
    candle_seconds: int,
    prediction_horizon_seconds: int,
    output_html_path: str,
    n_rows_to_profile: Optional[int] = None,
    table: str = None,
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
        pair: The pair of the technical indicators to fetch.
        training_set_size_days: The number of days in the past to fetch the technical indicators for.
        candle_seconds: The number of seconds in the candle.
        prediction_horizon_seconds: The number of seconds in the prediction horizon.
        output_html_path: The path to save the HTML file to.
        n_rows_to_profile: The number of rows to profile.
        table: The table to fetch the technical indicators from.
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
            table=table,
        )

        # Step 2: Add a target column
        ts_data['target'] = ts_data['close'].shift(
            -prediction_horizon_seconds // candle_seconds
        )
        ts_data = ts_data.dropna()

        # Log training dataset to MLFlow
        dataset = mlflow.data.from_pandas(ts_data)
        mlflow.log_input(dataset, context='training')

        # Log dataset size
        mlflow.log_param('ts_data_shape', ts_data.shape)

        # Step 3: Validate the data
        ts_data = validate_data(
            ts_data=ts_data,
            prediction_horizon_seconds=prediction_horizon_seconds,
            candle_seconds=candle_seconds,
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


if __name__ == '__main__':
    train(
        mlflow_tracking_uri='http://localhost:5000',
        risingwave_host='localhost',
        risingwave_port=4567,
        risingwave_user='root',
        risingwave_password='123456',
        risingwave_database='dev',
        pair='BTC/USD',
        training_set_size_days=30,
        candle_seconds=60,
        prediction_horizon_seconds=300,  # 5 minutes
        output_html_path='./eda_report.html',
        n_rows_to_profile=1000,
        table='public.technical_indicators',
    )

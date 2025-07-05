"""
Predictor service for real-time cryptocurrency price prediction.

This module provides functionality to load trained models from MLflow and generate
real-time predictions based on incoming technical indicator data from RisingWave.

The predictor continuously monitors the input table for new data and generates
predictions that are written to an output table, enabling real-time trading
signals and market analysis.

Key Features:
- Model loading from MLflow registry with version control
- Real-time data streaming from RisingWave
- Automated prediction generation on new data
- Configurable prediction horizons and trading pairs
- Integration with technical indicator pipelines

Example:
    The predictor can be configured to monitor cryptocurrency trading pairs data with
    a given candle size and generate predictions for a given prediction horizon,
    writing results to a predictions table for downstream trading systems or analysis
    dashboards.
"""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger
from risingwave import (
    OutputFormat,
    RisingWave,
    RisingWaveConnOptions,
)

from predictor.config import predictor_config
from predictor.model_registry import (
    get_model_name,
    load_model,
)


def predict(
    mlflow_tracking_uri: str,
    pair: str,
    candle_seconds: int,
    prediction_horizon_seconds: int,
    risingwave_host: str,
    risingwave_port: int,
    risingwave_user: str,
    risingwave_password: str,
    risingwave_database: str,
    risingwave_schema: str,
    risingwave_input_table: str,
    risingwave_output_table: str,
    model_alias: str = 'champion',
):
    """
    Generates a new prediction as soon as new data is available in the `risingwave_input_table`.

    Steps:
    1. Load the model from the MLFlow registry with the given `model_alias` if provided. Otherwise, use the latest model.
    2. Start listening to the `risingwave_input_table` for new data.
    3. For each new or updated row, generate a new prediction.
    4. Write the prediction to the `risingwave_output_table`.

    Args:
        mlflow_tracking_uri: The URI of the MLFlow tracking server.
        pair: The pair to predict.
        candle_seconds: The number of seconds in a candle.
        prediction_horizon_seconds: The number of seconds to predict.
        risingwave_host: The host of the RisingWave instance.
        risingwave_port: The port of the RisingWave instance.
        risingwave_user: The user of the RisingWave instance.
        risingwave_password: The password of the RisingWave instance.
        risingwave_database: The database of the RisingWave instance.
        risingwave_schema: The schema of the RisingWave instance.
        risingwave_input_table: The input table of the RisingWave instance.
        risingwave_output_table: The output table of the RisingWave instance.
        model_alias: The alias of the model to use for prediction.
    """
    # Step 1: Load the model from the MLFlow registry with the given `model_alias` if provided. Otherwise, use the latest model.
    model_name = get_model_name(pair, candle_seconds, prediction_horizon_seconds)
    model, features, model_version = load_model(model_name, model_alias)
    logger.info(f'Loaded model: {model_name} with features: {features}')

    # Step 2: Start listening to the `risingwave_input_table` for new data.
    rw = RisingWave(
        RisingWaveConnOptions.from_connection_info(
            host=risingwave_host,
            port=risingwave_port,
            user=risingwave_user,
            password=risingwave_password,
            database=risingwave_database,
        )
    )

    def prediction_handler(data: pd.DataFrame) -> Any:
        """
        Maps the data changes to fresh predictions using the loaded model. These
        predictions are then written to the `risingwave_output_table`.
        """
        logger.info('Reading new data...')

        # We only read Insert and Updates.
        data = data[data['op'].isin(['Insert', 'UpdateInsert'])] # type: ignore

        # Keep only the data for the given pair.
        data = data[data['pair'] == pair] # type: ignore

        # Keep only the relevant features.
        data = data[features] # type: ignore

        # We only read recent data.
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  # current time in milliseconds UTC
        data = data[
            data['window_start_ms'] > (ts_ms - 1000 * candle_seconds * 5)
        ]  # type: ignore

        if data.empty:
            logger.info('No data to predict')
            return

        logger.info(f'Received new data: {data.shape[0]} row/s updated')
        logger.info(f'Data:\n {data.head(5)}')

        # Step 3: For each new or updated row, generate a new prediction.
        predictions: pd.Series = model.predict(data) # type: ignore

        # Write the predictions to the `risingwave_output_table`.
        # Update ts_ms to the current time.
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        predicted_ts_ms = (
            data['window_start_ms'] + (candle_seconds + prediction_horizon_seconds) * 1000 # predicted time in milliseconds UTC
        ).to_list()

        output = pd.DataFrame({
            'pair': pair,
            'ts_ms': ts_ms,
            'model_name': model_name,
            'model_version': model_version,
            'predicted_ts_ms': predicted_ts_ms,
            'predicted_price': predictions,
        })

        # Step 4: Write the prediction to the `risingwave_schema.risingwave_output_table`.
        logger.info(f'Writing predictions to {risingwave_schema}.{risingwave_output_table}')
        logger.info(f'Output:\n {output.head(5)}')
        rw.insert(
            table_name=risingwave_output_table,
            schema_name=risingwave_schema,
            data=output,
        )
        logger.info(f'{data.shape[0]} row/s written to {risingwave_schema}.{risingwave_output_table} successfully.')

    rw.on_change(
        subscribe_from=risingwave_input_table,
        schema_name=risingwave_schema,
        handler=prediction_handler,
        output_format=OutputFormat.DATAFRAME,
    )


if __name__ == '__main__':
    predict(
        mlflow_tracking_uri=predictor_config.mlflow_tracking_uri,
        pair=predictor_config.pair,
        candle_seconds=predictor_config.candle_seconds,
        prediction_horizon_seconds=predictor_config.prediction_horizon_seconds,
        risingwave_host=predictor_config.risingwave_host,
        risingwave_port=predictor_config.risingwave_port,
        risingwave_user=predictor_config.risingwave_user,
        risingwave_password=predictor_config.risingwave_password,
        risingwave_database=predictor_config.risingwave_database,
        risingwave_schema=predictor_config.risingwave_schema,
        risingwave_input_table=predictor_config.risingwave_input_table,
        risingwave_output_table=predictor_config.risingwave_output_table,
        model_alias=predictor_config.model_alias,
    )

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_configuration import ExpectationConfiguration


def validate_data(
    ts_data: pd.DataFrame,
    prediction_horizon_seconds: int,
    candle_seconds: int,
    max_percentage_rows_with_null_values: float,
) -> pd.DataFrame:
    """
    Validates the data by checking for missing values and duplicates. We
    will use the great_expectations library to validate the data.

    Args:
        - ts_data: The time series data to validate.
        - prediction_horizon_seconds: The number of seconds in the future to predict.
        - candle_seconds: The number of seconds in the candle.

    Returns:
        - The validated time series data.
    """
    # Ensure that prediction horizon seconds is a multiple of candle seconds
    if prediction_horizon_seconds % candle_seconds != 0:
        raise ValueError(
            'Prediction horizon seconds must be a multiple of candle seconds'
        )

    # Check for missing values
    ts_data_without_nans = ts_data.dropna()
    perc_row_with_null_values = (len(ts_data) - len(ts_data_without_nans)) / len(
        ts_data
    )
    if perc_row_with_null_values > max_percentage_rows_with_null_values:
        raise Exception(
            'ts_data has too many rows with null values. Aborting the training script'
        )

    # We proceed with the dataset without nans
    ts_data = ts_data_without_nans

    # Create a great_expectations context
    context = gx.get_context()
    gx_df = gx.from_pandas(ts_data)

    # Create a new expectation suite
    expectation_suite = context.add_or_update_expectation_suite(
        expectation_suite_name='ts_data_suite',
    )

    # Add expectations to the suite
    # Make sure that there are no duplicate timestamps; each timestamp
    # represents a unique candle.
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type='expect_column_values_to_be_unique',
            kwargs={'column': 'window_start_ms'},
        )
    )

    # Make sure that there are no null values in the close column.
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type='expect_column_values_to_not_be_null',
            kwargs={'column': 'close'},
        )
    )

    # Make sure that all close values are positive.
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type='expect_column_values_to_be_between',
            kwargs={'column': 'close', 'min_value': 0, 'max_value': None},
        )
    )

    # Make sure that there are no null values in the target column. We
    # don't want to train on data that we don't have a target for.
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type='expect_column_values_to_not_be_null',
            kwargs={'column': 'target'},
        )
    )

    # Make sure that the target column is a float.
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type='expect_column_values_to_be_of_type',
            kwargs={'column': 'target', 'type_': 'float64'},
        )
    )

    # Validate the data.
    if not gx_df.validate().success:
        raise ValueError('Data validation failed')

    return ts_data


def generate_data_drift_report(
    ts_data: pd.DataFrame,
    model_name: str,
):
    """ """
    # Use the mlflow sdk to get the experiment name/id for the last model in the model registry
    # TODO

    # Download the ts_data used by the model
    # TODO

    # Now you have the current run `ts_data` and the `ts_data` used by the last model in the model registry.

    # Use a library like Evidently to generate a data drift report. See Github: https://github.com/evidentlyai/evidently.
    # '''
    # Run the Data Drift evaluation preset that will test for shift in column distributions. Take the first 60 rows of the dataframe as "current" data and the following as reference. Get the output in Jupyter notebook:

    # report = Report([
    #     DataDriftPreset(method="psi")
    # ],
    # include_tests="True")
    # my_eval = report.run(iris_frame.iloc[:60], iris_frame.iloc[60:])
    # my_eval
    # '''

    # Save the report to the MLflow experiment
    # TODO

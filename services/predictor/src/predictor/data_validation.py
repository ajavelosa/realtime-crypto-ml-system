import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_configuration import ExpectationConfiguration


def validate_data(
    ts_data: pd.DataFrame,
    prediction_horizon_seconds: int,
    candle_seconds: int,
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

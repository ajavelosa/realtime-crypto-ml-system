from datetime import timedelta

from loguru import logger
from quixstreams import Application
from quixstreams.models import TopicConfig

from candles.config import config


def init_candle(trade: dict) -> dict:
    """
    Initializes a new candle for a given symbol.

    Args:
        trade (dict): The trade to initialize the candle with.

    Returns:
        dict: The initial candle state.
    """
    return {
        'open': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'close': trade['price'],
        'volume': trade['quantity'],
        'pair': trade['product_id'],
    }


def update_candle(candle: dict, trade: dict) -> dict:
    """
    Updates the candle's open, high, low, close, and volume values
    from trade's within the current candle window.

    Args:
        candle (dict): The current candle state.
        trade (dict): The trade to update the candle with.

    Returns:
        dict: The updated candle state.
    """
    # open price does not change so we don't need to update it
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['close'] = trade['price']
    candle['volume'] += trade['quantity']

    return candle


def run(
    # kafka parameters
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    # candle parameters
    candle_seconds: int,
    emit_intermediate_candles: bool = True,
):
    """
    Transforms a stream of trades into a stream of candles.

    Steps:
    1. Ingests raw trades from the `kafka input topic`.
    2. Calculates the open, high, low, close, and volume for each candle by symbol.
    4. Outputs the candles to the `kafka output topic`.

    Args:
        kafka_broker_address (str): The address of the kafka broker.
        kafka_input_topic (str): The topic to ingest raw trades from.
        kafka_output_topic (str): The topic to output candles to.
        candle_seconds (int): The duration of the candles in seconds.

    Returns:
        None
    """

    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='earliest',
        auto_create_topics=True,
    )

    # input topic
    trades_topic = app.topic(kafka_input_topic, value_deserializer='json')

    # output topic
    candles_topic = app.topic(
        kafka_output_topic,
        value_serializer='json',
        key_serializer='json',
        config=TopicConfig(
            num_partitions=1,
            replication_factor=1,
        ),
    )

    # 1. Ingest raw trades from the `kafka input topic`
    # with a streaming dataframe.
    sdf = app.dataframe(topic=trades_topic)

    # 2. Calculate the open, high, low, close, and volume for each candle by symbol.

    # Transform the input data into candles
    sdf = sdf.tumbling_window(duration_ms=timedelta(seconds=candle_seconds)).reduce(
        reducer=update_candle, initializer=init_candle
    )

    # we emit all intermediate candles to make the system more responsive
    if emit_intermediate_candles:
        sdf = sdf.current()
    else:
        sdf = sdf.final()

    # Extract open, high, low, close, volume, pair, window_start, and window_end from the dataframe
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    sdf['pair'] = sdf['value']['pair']

    # Extract window_start and window_end from the dataframe
    sdf['window_start'] = sdf['start']
    sdf['window_end'] = sdf['end']

    # Keep only the required columns
    sdf = sdf[
        ['open', 'high', 'low', 'close', 'volume', 'pair', 'window_start', 'window_end']
    ]

    sdf['candle_seconds'] = candle_seconds

    # Log the output data
    sdf = sdf.update(lambda value: logger.debug(f'Candle: {value}'))

    # 3. Output the candles to the `kafka output topic`.
    # Push the candles to the output topic with a string key
    sdf.to_topic(topic=candles_topic, key=lambda value: value['pair'])

    # Run the streaming dataframe application
    app.run()


if __name__ == '__main__':
    try:
        run(
            kafka_broker_address=config.kafka_broker_address,
            kafka_input_topic=config.kafka_input_topic,
            kafka_output_topic=config.kafka_output_topic,
            kafka_consumer_group=config.kafka_consumer_group,
            candle_seconds=config.candle_seconds,
            emit_intermediate_candles=True,
        )
    except KeyboardInterrupt:
        logger.info('Keyboard interrupt. Exiting gracefully...')

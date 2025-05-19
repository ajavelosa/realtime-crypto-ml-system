from loguru import logger
from quixstreams import Application

from technical_indicators.candle import update_candles_in_state
from technical_indicators.config import config
from technical_indicators.indicators import compute_technical_indicators


def run(
    # kafka parameters
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    # candle parameters
    candle_seconds: int,
):
    """
    Transforms a stream of candles into a stream of technical indicators.

    Steps:
    1. Ingests candles from the `kafka input topic`.
    2. Calculates the technical indicators for each candle by symbol.
    3. Outputs the technical indicators to the `kafka output topic`.

    Args:
        kafka_broker_address (str): The address of the kafka broker.
        kafka_input_topic (str): The topic to ingest candles from.
        kafka_output_topic (str): The topic to output technical indicators to.
        kafka_consumer_group (str): The consumer group to use for the application.
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
    candles_topic = app.topic(kafka_input_topic, value_deserializer='json')

    # output topic
    technical_indicators_topic = app.topic(kafka_output_topic, value_serializer='json')

    # 1. Ingest raw trades from the `kafka input topic`
    # with a streaming dataframe.
    sdf = app.dataframe(topic=candles_topic)

    # 2. Filter only candles with a duration of `candle_seconds`
    sdf = sdf[sdf['candle_seconds'] == candle_seconds]

    # 3. Add candles to the state dictionary
    sdf = sdf.apply(update_candles_in_state, stateful=True)

    # 4. Compute technical indicators from candles in the state dictionary
    sdf = sdf.apply(compute_technical_indicators, stateful=True)

    # Log the output data
    sdf = sdf.update(lambda value: logger.debug(f'Candle: {value}'))

    # 5. Output the technical indicators to the `kafka output topic`.
    sdf.to_topic(topic=technical_indicators_topic)

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
        )
    except KeyboardInterrupt:
        logger.info('Keyboard interrupt. Exiting gracefully...')

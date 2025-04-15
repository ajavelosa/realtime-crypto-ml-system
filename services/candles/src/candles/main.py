from loguru import logger
from quixstreams import Application


def run(
    # kafka parameters
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    # candle parameters
    candle_seconds: int,
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
        auto_offset_reset='earliest',
    )

    # input topic
    trades_topic = app.topic(kafka_input_topic, value_deserializer='json')

    # output topic
    candles_topic = app.topic(kafka_output_topic, value_serializer='json')

    # 1. Ingest raw trades from the `kafka input topic`
    # with a streaming dataframe.
    sdf = app.dataframe(topic=trades_topic)

    # 2. Calculate the open, high, low, close, and volume for each candle by symbol.
    # print the input data
    sdf = sdf.update(lambda message: logger.info(f'Received message: {message}'))

    # TODO: transform the input data into candles

    # 3. Output the candles to the `kafka output topic`.
    # push the candles to the output topic
    sdf.to_topic(topic=candles_topic)

    # run the streaming dataframe application
    app.run()


if __name__ == '__main__':
    run(
        kafka_broker_address='localhost:31234',
        kafka_input_topic='trades',
        kafka_output_topic='candles',
        candle_seconds=60,
    )

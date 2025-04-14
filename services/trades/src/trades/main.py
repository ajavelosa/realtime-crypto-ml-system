# Module to process trades from Kraken and push them to Kafka

import time

from kraken_api import KrakenAPI, Trade
from loguru import logger
from quixstreams import Application

from typing import List


def run(
    kafka_broker_address: str,
    kafka_topic_name: str,
    kraken_api: KrakenAPI,
):
    app = Application(
        broker_address=kafka_broker_address,
        auto_offset_reset="earliest",
    )

    # Define the topic with a JSON value serializer
    topic = app.topic(name=kafka_topic_name, value_serializer="json")

    # Create a producer instance
    with app.get_producer() as producer:

        while True:

            # 1. Fetch the trades from the external API
            events: List[Trade] = kraken_api.get_trades()

            for event in events:
                # 2. Serialize the event to JSON and send it to the topic
                message = topic.serialize(key=event.product_id, value=event.to_dict())

                # 3. Produce the message to the kafka topic
                producer.produce(
                    topic=topic.name,
                    value=message.value,
                    key=message.key,
                )

                logger.info(f'Trade {event.to_dict()} pushed to Kafka')

                time.sleep(1)

if __name__ == "__main__":

    kraken_api = KrakenAPI(product_ids=["BTC/EUR"])

    run(
        kafka_broker_address="localhost:31234",
        kafka_topic_name="trades",
        kraken_api=kraken_api,
    )

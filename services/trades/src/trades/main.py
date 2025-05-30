# Module to process trades from Kraken and push them to Kafka

import time
from typing import List

from loguru import logger
from quixstreams import Application
from quixstreams.models import TopicConfig

from trades.config import config
from trades.kraken_rest_api import KrakenRestAPI
from trades.kraken_websocket_api import KrakenWebsocketAPI
from trades.trade import Trade


def run(
    kafka_broker_address: str,
    kafka_topic_name: str,
    kraken_api: KrakenWebsocketAPI | KrakenRestAPI,
):
    app = Application(
        broker_address=kafka_broker_address,
        auto_offset_reset='earliest',
        auto_create_topics=True,
    )

    # Define the topic with a JSON value serializer
    topic = app.topic(
        name=kafka_topic_name,
        value_serializer='json',
        key_serializer='json',
        config=TopicConfig(
            num_partitions=2,
            replication_factor=1,
        ),
    )

    # Create a producer instance
    with app.get_producer() as producer:
        while not kraken_api.is_done():
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


if __name__ == '__main__':
    if config.live_or_historical == 'live':
        kraken_api = KrakenWebsocketAPI(config.product_ids)
    else:
        kraken_api = KrakenRestAPI(
            product_id=config.product_ids[0],
            last_n_days=config.last_n_days,
        )

    try:
        run(
            kafka_broker_address=config.kafka_broker_address,
            kafka_topic_name=config.kafka_topic_name,
            kraken_api=kraken_api,
        )
    except KeyboardInterrupt:
        logger.info('Keyboard interrupt. Exiting gracefully...')

from quixstreams import Application
import time
from datetime import datetime

from loguru import logger

app = Application(
    broker_address="localhost:31234",
    consumer_group="example",
    auto_offset_reset="earliest",
)

# Define the topic with a JSON value serializer
topic = app.topic(name="my_topic", value_serializer="json")

# Create a producer instance
with app.get_producer() as producer:

    while True:

        # 1. Fetch the trades from the external API
        event = {
            "symbol": "BTC",
            "price": 10000,
            "timestamp": datetime.now().isoformat(),
        }

        # 2. Serialize the event to JSON and send it to the topic
        message = topic.serialize(key=event["symbol"], value=event)

        # 3. Produce the message to the kafka topic
        producer.produce(
            topic=topic.name,
            value=message.value,
            key=message.key,
        )
        logger.info(f"Produced message to topic: {message.value}")

        time.sleep(1)

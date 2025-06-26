from pathlib import Path

from loguru import logger
from risingwave import RisingWave, RisingWaveConnOptions

# Manual dialect registration - required because sqlalchemy-risingwave doesn't auto-register
try:
    from sqlalchemy import dialects

    dialects.registry.register(
        'risingwave', 'sqlalchemy_risingwave.psycopg2', 'RisingWaveDialect_psycopg2'
    )
    logger.info('Successfully registered RisingWave SQLAlchemy dialect')
except ImportError as e:
    logger.error(f'Failed to import sqlalchemy-risingwave: {e}')
except Exception as e:
    logger.error(f'Failed to register RisingWave dialect: {e}')


def create_table_in_risingwave(
    table_name: str,
    kafka_topic: str,
    kafka_broker_address: str,
    host: str,
    port: int,
    user: str,
    database: str,
    password: str,
) -> None:
    """
    Creates a RisingWave table connected to a Kafka topic for real-time data ingestion.

    Args:
        table_name: Name of the table to create
        kafka_topic: Name of the Kafka topic to connect to
        kafka_broker_address: Address of the Kafka broker
        risingwave_host: RisingWave server hostname (Kubernetes service name)
        risingwave_port: RisingWave server port
        risingwave_user: RisingWave user credentials
        risingwave_database: RisingWave database name
        risingwave_password: RisingWave password
    """
    # Ensure we're using the correct Kafka broker address
    kafka_broker_address = 'kafka-e11b-kafka-bootstrap.kafka.svc.cluster.local:9092'

    query = (
        (Path(__file__).parent.parent.parent / 'query.sql')
        .read_text()
        .format(
            risingwave_table_name=table_name,
            kafka_topic=kafka_topic,
            kafka_broker_address=kafka_broker_address,
        )
    )

    logger.info(f'Creating table with Kafka broker: {kafka_broker_address}')
    logger.info(f'Connecting to RisingWave at: {host}:{port}')

    client = None
    try:
        client = RisingWave(
            RisingWaveConnOptions.from_connection_info(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
            )
        )
        client.execute(query)
        logger.info('Table created successfully')
    except Exception as e:
        logger.error(f'Failed to create table: {str(e)}')
        raise
    finally:
        if client is not None:
            try:
                # Try to close the connection, but don't fail if it doesn't work
                client.close()
            except Exception as e:
                logger.warning(f'Error while closing connection: {str(e)}')

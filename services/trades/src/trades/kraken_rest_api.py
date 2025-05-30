import json
import time

import requests
from loguru import logger

from trades.trade import Trade


class KrakenRestAPI:
    URL = 'https://api.kraken.com/0/public/Trades'

    def __init__(self, product_id: str, last_n_days: int = 1):
        self.url = self.URL
        self.product_id = product_id
        self.last_n_days = last_n_days
        self._is_done = False

        # Get current time in nanoseconds and subtract last_n_days (also in nanoseconds)
        self.since_timestamp_ns = int(
            time.time_ns() - int(last_n_days * 24 * 60 * 60 * 1000000000)
        )

    def get_trades(self) -> list[Trade]:
        """
        Sends a GET request to the Kraken REST API to get the
        trades for the product_id and last_n_days.

        Returns:
            list[Trade]: List of trades for the product_id and for
            the last_n_days.
        """

        # Step 1: Set the right headers and parameters for the request
        headers = {'Accept': 'application/json'}
        params = {
            'pair': self.product_id,
            'since': self.since_timestamp_ns,
        }
        # Step 2: Send a GET request to the Kraken REST API
        try:
            response = requests.request('GET', self.URL, headers=headers, params=params)

        except requests.exceptions.SSLError as e:
            logger.error(f'The Kraken REST API is not available. Error: {e}')

            # wait 10 seconds and try again
            # It would be better to make this source stateful and recoverable, so if
            # the container goes down and gets restarted by kubernetes, it can recover
            # from the last known state.

            # TODO: Implement this as a stateful Quix Streams source data source so we
            # don't have to sleep here.
            logger.error('Sleeping for 10 seconds and trying again...')
            time.sleep(10)
            return []

        # Step 3: Parse the response
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse the response as JSON: {e}')
            return []

        # Step 4: Parse the trades
        try:
            trades = data['result'][self.product_id]
        except KeyError as e:
            logger.error(f'The response does not contain the trades: {e}')
            return []

        # Step 5: Convert the trades to Trade objects
        trades = [
            Trade.from_kraken_rest_api_response(
                product_id=self.product_id,
                price=trade[0],
                quantity=trade[1],
                timestamp_sec=trade[2],
            )
            for trade in trades
        ]

        # Update the since_timestamp_ns
        self.since_timestamp_ns = int(float(data['result']['last']))

        # Check if there are more trades to fetch
        if self.since_timestamp_ns > int(time.time_ns() - 1000000000):
            # we got trades until now, so we can stop
            self._is_done = True

        return trades

    def is_done(self) -> bool:
        """
        Returns True if there are no more trades to fetch, False otherwise.
        """
        return self._is_done

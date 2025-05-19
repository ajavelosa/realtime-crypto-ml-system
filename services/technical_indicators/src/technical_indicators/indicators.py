from loguru import logger
from talib import stream


def compute_technical_indicators(candle: dict, state: dict) -> dict:
    """
    Computes technical indicators from the candles in the state dictionary.

    Args:
        candle (dict): Candle to compute the technical indicators from.
        state (dict): State dictionary containing the candles.

    Returns:
        dict: Dictionary containing the technical indicators.
    """
    import numpy as np

    # Extract the candles from the state dictionary
    candles = state.get('candles', default=[])

    logger.debug(f'Number of candles in state: {len(candles)}')

    # Extract the open, high, low, close, and volume from the candles (which is a list of
    # dictionaries) into numpy arrays because TA-Lib expects numpy arrays.
    _open = np.array([c['open'] for c in candles])
    _high = np.array([c['high'] for c in candles])
    _low = np.array([c['low'] for c in candles])
    close = np.array([c['close'] for c in candles])
    _volume = np.array([c['volume'] for c in candles])

    indicators = {}

    # Simple moving average (SMA) for different periods
    indicators['sma_7'] = stream.SMA(close, timeperiod=7)
    indicators['sma_14'] = stream.SMA(close, timeperiod=14)
    indicators['sma_21'] = stream.SMA(close, timeperiod=21)
    indicators['sma_60'] = stream.SMA(close, timeperiod=60)

    breakpoint()

    # Return the latest candle with the indicators
    return {
        **candle,  # Spread the current candle
        **indicators,  # Spread the indicators
    }

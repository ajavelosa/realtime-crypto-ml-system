def get_model_name(
    pair: str,
    candle_seconds: int,
    prediction_horizon_seconds: int,
) -> str:
    """
    Get the name of the model in the MLFlow model registry.
    """
    return f'{pair.replace("/", "-")}_{candle_seconds}s_{prediction_horizon_seconds}s'

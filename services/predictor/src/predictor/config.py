from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='services/predictor/settings.env', env_file_encoding='utf-8'
    )

    mlflow_tracking_uri: str = 'http://localhost:5000'
    risingwave_host: str = 'localhost'
    risingwave_port: int = 4567
    risingwave_user: str = 'root'
    risingwave_password: str = '123456'
    risingwave_database: str = 'dev'
    risingwave_table: str = 'public.technical_indicators'
    pair: str = 'BTC/USD'
    training_set_size_days: int = 10
    candle_seconds: int = 60
    prediction_horizon_seconds: int = 300  # 5 minutes
    output_html_path: str = './eda_report.html'
    n_rows_to_profile: int = 1000
    train_test_split_ratio: float = 0.8
    n_model_candidates: int = 10
    features: list[str] = [
        'open',
        'high',
        'low',
        'close',
        'window_start_ms',
        'volume',
        'sma_7',
        'sma_14',
        'sma_21',
        'sma_60',
        'ema_7',
        'ema_14',
        'ema_21',
        'ema_60',
        'rsi_7',
        'rsi_14',
        'rsi_21',
        'rsi_60',
        'macd_7',
        'macdsignal_7',
        'macdhist_7',
        'obv',
    ]


training_config = TrainingConfig()

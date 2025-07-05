CREATE TABLE IF NOT EXISTS {risingwave_output_table} (
    pair VARCHAR,
    ts_ms BIGINT,

    -- useful for monitoring the model performance
    model_name VARCHAR,
    model_version INT,

    predicted_ts_ms BIGINT,
    predicted_price FLOAT,

    PRIMARY KEY (pair, ts_ms, model_name, model_version, predicted_ts_ms)
);
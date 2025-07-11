SELECT
    pair,
    predicted_ts_ms,
    predicted_price

FROM {view_name}

WHERE pair = '{pair}'
  AND datestr = TO_CHAR(NOW(), 'YYYY-MM-DD');

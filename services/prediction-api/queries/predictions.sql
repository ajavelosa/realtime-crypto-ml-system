-- Get the most recent prediction for a given pair
-- Only get predictions from the last 10 seconds to ensure freshness
SELECT
    predicted_ts_ms,
    predicted_price

FROM public.{table_name}
WHERE pair = '{pair}'
AND predicted_ts_ms > (EXTRACT(EPOCH FROM NOW()) * 1000) - 10000
ORDER BY predicted_ts_ms DESC
LIMIT 1

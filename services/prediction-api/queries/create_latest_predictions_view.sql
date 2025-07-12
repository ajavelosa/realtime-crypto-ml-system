-- Create a materialized view that contains the latest predictions with date field
-- Uses MAX aggregation for better streaming performance
DROP MATERIALIZED VIEW IF EXISTS :view_name;

CREATE MATERIALIZED VIEW :view_name AS

WITH max_predictions_per_date AS (
    SELECT
        pair,
        TO_CHAR(TO_TIMESTAMP(ts_ms / 1000) AT TIME ZONE 'UTC', 'YYYY-MM-DD') as datestr,
        MAX(predicted_ts_ms) as max_predicted_ts_ms

    FROM :table_name

    GROUP BY pair, TO_CHAR(TO_TIMESTAMP(ts_ms / 1000) AT TIME ZONE 'UTC', 'YYYY-MM-DD')
),

max_ts_for_ties AS (
    SELECT
        p.pair,
        p.predicted_ts_ms,
        TO_CHAR(TO_TIMESTAMP(ts_ms / 1000) AT TIME ZONE 'UTC', 'YYYY-MM-DD') as datestr,
        MAX(p.ts_ms) as max_ts_ms

    FROM :table_name p
    INNER JOIN max_predictions_per_date mp
        ON p.pair = mp.pair
        AND TO_CHAR(TO_TIMESTAMP(ts_ms / 1000) AT TIME ZONE 'UTC', 'YYYY-MM-DD') = mp.datestr
        AND p.predicted_ts_ms = mp.max_predicted_ts_ms

    GROUP BY p.pair, p.predicted_ts_ms, TO_CHAR(TO_TIMESTAMP(ts_ms / 1000) AT TIME ZONE 'UTC', 'YYYY-MM-DD')
)

SELECT
    p.pair,
    p.predicted_ts_ms,
    p.predicted_price,
    mt.datestr

FROM :table_name p

INNER JOIN max_ts_for_ties mt
    ON p.pair = mt.pair
    AND p.predicted_ts_ms = mt.predicted_ts_ms
    AND p.ts_ms = mt.max_ts_ms
    AND TO_CHAR(TO_TIMESTAMP(ts_ms / 1000) AT TIME ZONE 'UTC', 'YYYY-MM-DD') = mt.datestr;

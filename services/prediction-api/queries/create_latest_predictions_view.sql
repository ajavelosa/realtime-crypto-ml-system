-- Create a materialized view that contains the latest predictions with date field
-- Uses MAX aggregation for better streaming performance
-- Macro: {DATE_FUNC} = TO_CHAR(TO_TIMESTAMP(ts_ms / 1000), 'YYYY-MM-DD')
CREATE MATERIALIZED VIEW {view_name} AS

WITH max_predictions_per_date AS (
    SELECT
        pair,
        {DATE_FUNC} as datestr,
        MAX(predicted_ts_ms) as max_predicted_ts_ms

    FROM public.{table_name}

    GROUP BY pair, {DATE_FUNC}
),

max_ts_for_ties AS (
    SELECT
        p.pair,
        p.predicted_ts_ms,
        {DATE_FUNC} as datestr,
        MAX(p.ts_ms) as max_ts_ms

    FROM public.{table_name} p
    INNER JOIN max_predictions_per_date mp
        ON p.pair = mp.pair
        AND {DATE_FUNC} = mp.datestr
        AND p.predicted_ts_ms = mp.max_predicted_ts_ms

    GROUP BY p.pair, p.predicted_ts_ms, {DATE_FUNC}
)

SELECT
    p.pair,
    p.predicted_ts_ms,
    p.predicted_price,
    mt.datestr

FROM public.{table_name} p

INNER JOIN max_ts_for_ties mt
    ON p.pair = mt.pair
    AND p.predicted_ts_ms = mt.predicted_ts_ms
    AND p.ts_ms = mt.max_ts_ms
    AND {DATE_FUNC} = mt.datestr;

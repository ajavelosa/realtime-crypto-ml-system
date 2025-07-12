use axum::{
    extract::{ Query, State },
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use log::info;

use crate::AppState;

#[derive(Deserialize)]
pub struct PredictionParams {
    pair: String,
}

#[derive(Serialize)]
pub struct PredictionResponse {
    pair: String,
    predicted_ts_ms: i64,
    predicted_price: f64,
    status: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    error: String,
}

pub async fn get_prediction(
    params: Query<PredictionParams>,
    State(app_state): State<AppState>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let pair = &params.pair;
    info!("Requested prediction for pair: {}", pair);

    // Use the existing database connection pool
    let pool = app_state.pool;
    let config = app_state.config;

    // Load SQL query from external file and substitute parameters
    let query_template = include_str!("../../queries/latest_predictions.sql");
    let query = query_template
        .replace("{view_name}", &config.pg_view_name)
        .replace("{pair}", pair);

    info!("Executing query for pair: {}", pair);

    // Query the predictions for the given pair
    let row: (String, i64, f64) = match sqlx::query_as(&query)
        .fetch_one(&pool)
        .await
    {
        Ok(row) => row,
        Err(e) => {
            let error_msg = format!("Database query failed: {}", e);
            info!("Error: {}", error_msg);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: error_msg,
                }),
            ));
        }
    };

    // Build successful response with prediction data
    let response = PredictionResponse {
        pair: row.0,  // Use the pair from the query result
        predicted_ts_ms: row.1,
        predicted_price: row.2,
        status: "success".to_string(),
    };

    info!("Returning prediction to the client: predicted_price={}, predicted_ts_ms={}",
          response.predicted_price, response.predicted_ts_ms);

    Ok(Json(response))
}

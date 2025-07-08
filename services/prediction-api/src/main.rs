use axum::{
    extract::Query,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;

#[derive(Deserialize)]
struct PredictionParams {
    pair: String,
}

#[derive(Serialize)]
struct PredictionResponse {
    pair: String,
    predicted_ts_ms: i64,
    predicted_price: f64,
    status: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// This is how you denote the entrypoint of a Rust application
#[tokio::main]
async fn main() {
    // build our application with a route
    let app = Router::new()
        // `GET /health` goes to `health`
        .route("/health", get(health))
        // Add an endpoint to get predictions
        // We will use the pair as a query parameter
        // Example: http://localhost:3001/predictions?pair=BTC/USD
        .route("/predictions", get(get_predictions));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with a static string
async fn health() -> &'static str {
    "I am healthy, bruh!"
}

async fn get_predictions(
    params: Query<PredictionParams>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let pair = &params.pair;

    // 1. Connect to the database on RisingWave
    let pool = match PgPoolOptions::new()
        .max_connections(5)
        .connect("postgresql://root:123456@localhost:4567/dev")
        .await
    {
        Ok(pool) => pool,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Database connection failed: {}", e),
                }),
            ));
        }
    };

    let query = format!(r#"
        SELECT
            predicted_ts_ms,
            predicted_price

        FROM predictions
        WHERE pair = '{}'
        AND predicted_ts_ms > (EXTRACT(EPOCH FROM NOW()) * 1000) - 10000
        ORDER BY predicted_ts_ms DESC LIMIT 1"#,
        pair
    );

    // 2. Query the predictions for the given pair
    let row: (i64, f64) = match sqlx::query_as(&query)
        .fetch_one(&pool)
        .await
    {
        Ok(row) => row,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Database query failed: {}", e),
                }),
            ));
        }
    };

    let response = PredictionResponse {
        pair: pair.clone(),
        predicted_ts_ms: row.0,
        predicted_price: row.1,
        status: "success".to_string(),
    };

    Ok(Json(response))
}

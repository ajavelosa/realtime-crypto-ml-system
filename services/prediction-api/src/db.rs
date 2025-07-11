use sqlx::{PgPool, postgres::PgPoolOptions};
use log::info;

pub async fn get_pool(
    host: &str,
    port: &u16,
    database: &str,
    user: &str,
    password: &str,
) -> PgPool {
    let database_url = format!("postgres://{}:{}@{}:{}/{}", user, password, host, port, database);

    PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Failed to create connection pool")
}

pub async fn create_latest_predictions_view(
    pool: &PgPool,
    table_name: &str,
    view_name: &str,
) -> Result<(), String> {
    info!("Creating latest_predictions materialized view...");

    // First, drop the existing view if it exists (both types for safety)
    let drop_materialized_view_query = format!("DROP MATERIALIZED VIEW IF EXISTS {}", view_name);
    match sqlx::query(&drop_materialized_view_query).execute(pool).await {
        Ok(_) => info!("Dropped existing materialized view '{}'", view_name),
        Err(e) => info!("Note: Could not drop existing materialized view (may not exist): {}", e),
    }

    let drop_view_query = format!("DROP VIEW IF EXISTS {}", view_name);
    match sqlx::query(&drop_view_query).execute(pool).await {
        Ok(_) => info!("Dropped existing view '{}'", view_name),
        Err(e) => info!("Note: Could not drop existing view (may not exist): {}", e),
    }

    // Load and process the SQL template
    let create_view_query = include_str!("../queries/create_latest_predictions_view.sql");
    let create_view_query = create_view_query
        .replace("{table_name}", table_name)
        .replace("{view_name}", view_name)
        .replace("{DATE_FUNC}", "TO_CHAR(TO_TIMESTAMP(ts_ms / 1000), 'YYYY-MM-DD')");

    // Execute the view creation
    match sqlx::query(&create_view_query).execute(pool).await {
        Ok(_) => {
            info!("Latest predictions materialized view '{}' created successfully", view_name);
            Ok(())
        },
        Err(e) => {
            let error_msg = format!("Failed to create materialized view '{}': {}", view_name, e);
            info!("Error: {}", error_msg);
            Err(error_msg)
        }
    }
}

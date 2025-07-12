use sqlx::{PgPool, postgres::PgPoolOptions};

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

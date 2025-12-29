use std::net::SocketAddr;

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::get,
    Json, Router,
};
use respondedorbot::models::Update;
use respondedorbot::webhook::{
    app_state, handle_get, handle_post, log_bot_config, AppState, WebhookQuery,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    log_bot_config();

    let app_state = app_state();
    let app = Router::new()
        .route("/", get(handle_get_http).post(handle_post_http))
        .with_state(app_state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().expect("invalid bind address");
    tracing::info!("listening on {}", addr);

    axum_server::bind(addr)
        .serve(app.into_make_service())
        .await
        .expect("server failed");
}

async fn handle_get_http(
    State(state): State<AppState>,
    Query(query): Query<WebhookQuery>,
) -> (StatusCode, String) {
    let response = handle_get(&state, query).await;
    (response.status, response.body)
}

async fn handle_post_http(
    State(state): State<AppState>,
    Query(query): Query<WebhookQuery>,
    headers: HeaderMap,
    body: Result<Json<Update>, axum::extract::rejection::JsonRejection>,
) -> (StatusCode, String) {
    let secret_header = headers
        .get("X-Telegram-Bot-Api-Secret-Token")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());
    let update = body.ok().map(|Json(value)| value);

    let response = handle_post(&state, query, secret_header, update).await;
    (response.status, response.body)
}

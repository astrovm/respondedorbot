use std::net::SocketAddr;

use respondedorbot::webhook::{app_state, build_axum_router, log_bot_config};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    log_bot_config();

    let app_state = app_state();
    let app = build_axum_router(app_state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().expect("invalid bind address");
    tracing::info!("listening on {}", addr);

    axum_server::bind(addr)
        .serve(app.into_make_service())
        .await
        .expect("server failed");
}

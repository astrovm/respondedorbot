use crate::models::Update;
use crate::webhook::{
    app_state, handle_get, handle_post, log_bot_config, AppState, WebhookQuery, WebhookResponse,
};
use serde_urlencoded::from_str;
use worker::{event, Env, Request, Response, Result, Router};

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: worker::Context) -> Result<Response> {
    log_bot_config();

    let state = app_state(&env);

    Router::with_data(state)
        .get_async("/", handle_get_request)
        .post_async("/", handle_post_request)
        .run(req, env)
        .await
}

async fn handle_get_request(req: Request, ctx: worker::RouteContext<AppState>) -> Result<Response> {
    let query = match parse_query(&req) {
        Ok(query) => query,
        Err(response) => return Ok(response),
    };

    let response = handle_get(&ctx.data, query).await;
    build_response(response)
}

async fn handle_post_request(
    mut req: Request,
    ctx: worker::RouteContext<AppState>,
) -> Result<Response> {
    let query = match parse_query(&req) {
        Ok(query) => query,
        Err(response) => return Ok(response),
    };
    let secret_header = req
        .headers()
        .get("X-Telegram-Bot-Api-Secret-Token")
        .ok()
        .flatten();
    let update = req.json::<Update>().await.ok();

    let response = handle_post(&ctx.data, query, secret_header, update).await;
    build_response(response)
}

fn parse_query(req: &Request) -> std::result::Result<WebhookQuery, Response> {
    let url = req
        .url()
        .map_err(|err| Response::error(format!("Invalid request URL: {err}"), 400).unwrap())?;
    let query = url.query().unwrap_or_default();
    from_str(query).map_err(|_| Response::error("Invalid query parameters", 400).unwrap())
}

fn build_response(webhook_response: WebhookResponse) -> Result<Response> {
    Ok(Response::ok(webhook_response.body)?.with_status(webhook_response.status.as_u16()))
}

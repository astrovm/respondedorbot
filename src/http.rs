use http::Method;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::time::Duration;
use url::Url;

use futures::{future, pin_mut};
use wasm_bindgen::JsValue;
use worker::{Fetch, Headers, Request, RequestInit};

const AUTHORIZATION: &str = "authorization";
pub const CONTENT_TYPE: &str = "content-type";

#[derive(Debug)]
pub enum HttpError {
    Message(String),
    Worker(worker::Error),
    Timeout,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Message(msg) => write!(f, "{msg}"),
            HttpError::Worker(err) => write!(f, "{err}"),
            HttpError::Timeout => write!(f, "request timed out"),
        }
    }
}

impl std::error::Error for HttpError {}

impl From<worker::Error> for HttpError {
    fn from(value: worker::Error) -> Self {
        Self::Worker(value)
    }
}

pub type HttpResult<T> = Result<T, HttpError>;

#[derive(Clone)]
pub struct HttpClient;

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    pub fn new() -> Self {
        Self
    }

    pub fn get(&self, url: impl Into<String>) -> HttpRequestBuilder {
        HttpRequestBuilder::new(Method::GET, url.into())
    }

    pub fn post(&self, url: impl Into<String>) -> HttpRequestBuilder {
        HttpRequestBuilder::new(Method::POST, url.into())
    }
}

pub struct HttpRequestBuilder {
    method: Method,
    url: String,
    headers: Vec<(String, String)>,
    query: Vec<(String, String)>,
    body: Option<RequestBody>,
    bearer_token: Option<String>,
    timeout: Option<Duration>,
}

enum RequestBody {
    Json(serde_json::Value),
    Bytes(Vec<u8>),
}

impl HttpRequestBuilder {
    fn new(method: Method, url: String) -> Self {
        Self {
            method,
            url,
            headers: Vec::new(),
            query: Vec::new(),
            body: None,
            bearer_token: None,
            timeout: None,
        }
    }

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    pub fn headers(mut self, headers: http::HeaderMap) -> Self {
        for (name, value) in headers.iter() {
            if let Ok(val) = value.to_str() {
                self.headers
                    .push((name.as_str().to_string(), val.to_string()));
            }
        }
        self
    }

    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    pub fn bearer_auth(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    pub fn json<T: Serialize + ?Sized>(mut self, value: &T) -> Self {
        self.body = serde_json::to_value(value).ok().map(RequestBody::Json);
        self
    }

    pub fn body(mut self, bytes: Vec<u8>) -> Self {
        self.body = Some(RequestBody::Bytes(bytes));
        self
    }

    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    fn build_url(&self) -> HttpResult<Url> {
        let mut url = Url::parse(&self.url)
            .map_err(|err| HttpError::Message(format!("invalid url {}: {}", self.url, err)))?;
        if !self.query.is_empty() {
            let mut pairs = url.query_pairs_mut();
            for (key, value) in &self.query {
                pairs.append_pair(key, value);
            }
        }
        Ok(url)
    }

    pub async fn send(self) -> HttpResult<HttpResponse> {
        let url = self.build_url()?;

        let mut init = RequestInit::new();
        let headers = Headers::new();
        for (key, value) in &self.headers {
            headers.set(key, value)?;
        }
        if let Some(token) = self.bearer_token.as_ref() {
            headers.set(AUTHORIZATION, &format!("Bearer {}", token))?;
        }
        if let Some(body) = self.body {
            match body {
                RequestBody::Json(value) => {
                    if !headers.has(CONTENT_TYPE)? {
                        headers.set(CONTENT_TYPE, "application/json")?;
                    }
                    init.with_body(Some(JsValue::from_str(&value.to_string())));
                }
                RequestBody::Bytes(bytes) => {
                    let array = js_sys::Uint8Array::from(bytes.as_slice());
                    init.with_body(Some(array.into()));
                }
            }
        }
        init.with_headers(headers);
        init.with_method(method_to_worker(&self.method));

        let request = Request::new_with_init(url.as_ref(), &init)?;
        let fetch = Fetch::Request(request);

        let response = if let Some(duration) = self.timeout {
            let send = fetch.send();
            pin_mut!(send);
            let delay = worker::Delay::from(duration);
            pin_mut!(delay);
            match future::select(send, delay).await {
                future::Either::Left((resp, _)) => resp?,
                future::Either::Right((_, _)) => return Err(HttpError::Timeout),
            }
        } else {
            fetch.send().await?
        };

        Ok(HttpResponse::from_worker(response))
    }
}

pub struct HttpResponse {
    inner: worker::Response,
}

impl HttpResponse {
    fn from_worker(inner: worker::Response) -> Self {
        Self { inner }
    }

    pub async fn json<T: DeserializeOwned>(mut self) -> HttpResult<T> {
        self.inner.json::<T>().await.map_err(HttpError::from)
    }

    pub async fn text(mut self) -> HttpResult<String> {
        self.inner.text().await.map_err(HttpError::from)
    }

    pub async fn bytes(mut self) -> HttpResult<Vec<u8>> {
        self.inner.bytes().await.map_err(HttpError::from)
    }

    pub fn header(&self, key: &str) -> Option<String> {
        self.inner.headers().get(key).ok().flatten()
    }
}

fn method_to_worker(method: &Method) -> worker::Method {
    match *method {
        Method::GET => worker::Method::Get,
        Method::POST => worker::Method::Post,
        Method::PUT => worker::Method::Put,
        Method::DELETE => worker::Method::Delete,
        Method::HEAD => worker::Method::Head,
        Method::OPTIONS => worker::Method::Options,
        Method::PATCH => worker::Method::Patch,
        _ => worker::Method::Get,
    }
}

pub async fn get_with_ssl_fallback(http: &HttpClient, url: &str) -> HttpResult<HttpResponse> {
    match http.get(url).send().await {
        Ok(resp) => Ok(resp),
        Err(_) => Err(HttpError::Message("request failed".to_string())),
    }
}

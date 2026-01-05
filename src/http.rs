use http::Method;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::time::Duration;
use url::Url;

#[cfg(target_arch = "wasm32")]
use futures::{future, pin_mut};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use worker::{Fetch, Headers, Request, RequestInit};

#[cfg(target_arch = "wasm32")]
const AUTHORIZATION: &str = "authorization";
pub const CONTENT_TYPE: &str = "content-type";

#[derive(Debug)]
pub enum HttpError {
    Message(String),
    #[cfg(not(target_arch = "wasm32"))]
    Request(reqwest::Error),
    #[cfg(target_arch = "wasm32")]
    Worker(worker::Error),
    #[cfg(target_arch = "wasm32")]
    Timeout,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Message(msg) => write!(f, "{msg}"),
            #[cfg(not(target_arch = "wasm32"))]
            HttpError::Request(err) => write!(f, "{err}"),
            #[cfg(target_arch = "wasm32")]
            HttpError::Worker(err) => write!(f, "{err}"),
            #[cfg(target_arch = "wasm32")]
            HttpError::Timeout => write!(f, "request timed out"),
        }
    }
}

impl std::error::Error for HttpError {}

#[cfg(not(target_arch = "wasm32"))]
impl From<reqwest::Error> for HttpError {
    fn from(value: reqwest::Error) -> Self {
        HttpError::Request(value)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<worker::Error> for HttpError {
    fn from(value: worker::Error) -> Self {
        HttpError::Worker(value)
    }
}

pub type HttpResult<T> = Result<T, HttpError>;

#[derive(Clone)]
pub struct HttpClient {
    #[cfg(not(target_arch = "wasm32"))]
    inner: reqwest::Client,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: reqwest::Client::new(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {}
        }
    }

    pub fn get(&self, url: impl Into<String>) -> HttpRequestBuilder {
        HttpRequestBuilder::new(self.clone(), Method::GET, url.into())
    }

    pub fn post(&self, url: impl Into<String>) -> HttpRequestBuilder {
        HttpRequestBuilder::new(self.clone(), Method::POST, url.into())
    }
}

pub struct HttpRequestBuilder {
    #[cfg(not(target_arch = "wasm32"))]
    client: HttpClient,
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
    fn new(client: HttpClient, method: Method, url: String) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = client;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            client,
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

    pub fn query<K, V>(mut self, params: &[(K, V)]) -> Self
    where
        K: AsRef<str>,
        V: ToString,
    {
        for (k, v) in params {
            self.query.push((k.as_ref().to_string(), v.to_string()));
        }
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            let url = self.build_url()?;
            let mut builder = self
                .client
                .inner
                .request(self.method.clone(), url);
            for (key, value) in self.headers {
                builder = builder.header(key, value);
            }
            if let Some(token) = self.bearer_token {
                builder = builder.bearer_auth(token);
            }
            if let Some(body) = self.body {
                match body {
                    RequestBody::Json(value) => {
                        builder = builder.json(&value);
                    }
                    RequestBody::Bytes(bytes) => {
                        builder = builder.body(bytes);
                    }
                }
            }
            if let Some(duration) = self.timeout {
                builder = builder.timeout(duration);
            }
            let response = builder.send().await?;
            Ok(HttpResponse::from_reqwest(response))
        }

        #[cfg(target_arch = "wasm32")]
        {
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
}

pub struct HttpResponse {
    #[cfg(not(target_arch = "wasm32"))]
    inner: reqwest::Response,
    #[cfg(target_arch = "wasm32")]
    inner: worker::Response,
}

impl HttpResponse {
    #[cfg(not(target_arch = "wasm32"))]
    fn from_reqwest(inner: reqwest::Response) -> Self {
        Self { inner }
    }

    #[cfg(target_arch = "wasm32")]
    fn from_worker(inner: worker::Response) -> Self {
        Self { inner }
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    pub async fn json<T: DeserializeOwned>(mut self) -> HttpResult<T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.json::<T>().await.map_err(HttpError::from)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.json::<T>().await.map_err(HttpError::from)
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    pub async fn text(mut self) -> HttpResult<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.text().await.map_err(HttpError::from)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.text().await.map_err(HttpError::from)
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    pub async fn bytes(mut self) -> HttpResult<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(HttpError::from)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.bytes().await.map_err(HttpError::from)
        }
    }

    pub fn header(&self, key: &str) -> Option<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner
                .headers()
                .get(key)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.to_string())
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.headers().get(key).ok().flatten()
        }
    }
}

#[cfg(target_arch = "wasm32")]
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
        Err(_) => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let insecure = reqwest::Client::builder()
                    .danger_accept_invalid_certs(true)
                    .build()
                    .map_err(HttpError::from)?;
                let response = insecure.get(url).send().await.map_err(HttpError::from)?;
                Ok(HttpResponse::from_reqwest(response))
            }
            #[cfg(target_arch = "wasm32")]
            {
                Err(HttpError::Message("request failed".to_string()))
            }
        }
    }
}

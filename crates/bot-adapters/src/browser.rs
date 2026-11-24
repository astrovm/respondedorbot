//! Headless Chromium rendering for pages that build their content with
//! JavaScript. Chromium reaches the network only through a local proxy that
//! applies the same public-address checks as `web_fetch`, so a page cannot
//! make it load internal services.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use std::{env, fs, process, thread};

use serde_json::{Value, json};
use thiserror::Error;
use url::{Position, Url};

use crate::web_fetch::{FETCH_MAX_BYTES, HostResolver, SystemHostResolver, public_addresses};

/// How long one render may take, page load and scripts included.
pub const RENDER_TIME_LIMIT: Duration = Duration::from_secs(20);
/// How long scripts may keep loading data after the page's load event.
const SCRIPT_TIME_BUDGET: Duration = Duration::from_secs(8);
/// A loaded page with no request in flight for this long is done.
const NETWORK_QUIET: Duration = Duration::from_millis(500);
/// Time kept free before the hard limit to read the page and close Chromium.
const READ_MARGIN: Duration = Duration::from_secs(3);
/// How long a closed Chromium gets to exit before it is stopped.
const EXIT_GRACE: Duration = Duration::from_secs(3);
/// The largest DevTools message read. The page's HTML is cut to
/// `FETCH_MAX_BYTES` characters in the browser, and JSON escapes can make
/// each character up to six bytes.
const MAX_MESSAGE_BYTES: usize = FETCH_MAX_BYTES * 6 + 65_536;
/// Gives Chromium its DevTools pipe (it reads fd 3 and writes fd 4) on the
/// child's stdin and stdout, so it opens no port and no other descriptor.
const PIPE_WRAPPER: &str = "exec \"$@\" 3<&0 4>&1 0</dev/null 1>/dev/null 2>/dev/null";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(8);
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_HEAD_BYTES: usize = 65_536;
const REJECTED: &[u8] = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const CHROMIUM_FLAGS: &[&str] = &[
    "--headless",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-extensions",
    "--mute-audio",
    "--hide-scrollbars",
    "--window-size=1280,2400",
    "--blink-settings=imagesEnabled=false",
    // Chromium sends loopback requests around a proxy unless told not to.
    "--proxy-bypass-list=<-loopback>",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--remote-debugging-pipe",
];

pub trait PageRenderer {
    /// Loads `url`, runs its scripts and returns the resulting HTML.
    fn render(&self, url: &str) -> Result<String, RenderError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RenderError {
    #[error("browser could not start: {0}")]
    Start(String),
    #[error("browser failed: {0}")]
    Failed(String),
}

/// Opens TCP connections for the proxy once it has checked the address.
pub trait Dialer {
    fn dial(&self, address: SocketAddr) -> io::Result<TcpStream>;
}

pub struct TcpDialer;

impl Dialer for TcpDialer {
    fn dial(&self, address: SocketAddr) -> io::Result<TcpStream> {
        TcpStream::connect_timeout(&address, CONNECT_TIMEOUT)
    }
}

/// An HTTP proxy on loopback that only connects to public addresses.
pub struct PublicProxy {
    address: SocketAddr,
}

impl PublicProxy {
    pub fn start<R, D>(resolver: R, dialer: D) -> io::Result<Self>
    where
        R: HostResolver + Send + Sync + 'static,
        D: Dialer + Send + Sync + 'static,
    {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let address = listener.local_addr()?;
        let ports = Arc::new((resolver, dialer));
        thread::Builder::new()
            .name("browser-proxy".to_owned())
            .spawn(move || {
                for client in listener.incoming().flatten() {
                    let ports = Arc::clone(&ports);
                    let _ = thread::Builder::new()
                        .name("browser-proxy-conn".to_owned())
                        .spawn(move || serve(client, &ports.0, &ports.1));
                }
            })?;
        Ok(Self { address })
    }

    #[must_use]
    pub const fn address(&self) -> SocketAddr {
        self.address
    }
}

fn serve<R: HostResolver, D: Dialer>(
    mut client: TcpStream,
    resolver: &R,
    dialer: &D,
) -> io::Result<()> {
    client.set_read_timeout(Some(IDLE_TIMEOUT))?;
    let (head, early_bytes) = read_head(&mut client)?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split(' ');
    let method = request_line.next().unwrap_or_default();
    let target = request_line.next().unwrap_or_default();
    let version = request_line.next().unwrap_or_default();
    if method.eq_ignore_ascii_case("CONNECT") {
        let Some(mut upstream) = open(resolver, dialer, &format!("https://{target}")) else {
            return client.write_all(REJECTED);
        };
        client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
        upstream.write_all(&early_bytes)?;
        return relay(client, upstream);
    }
    let headers = lines
        .filter(|line| !line.is_empty())
        .filter_map(|line| line.split_once(':'))
        .collect::<Vec<_>>();
    let url = Url::parse(target).ok().filter(|url| url.scheme() == "http");
    let chunked = headers
        .iter()
        .any(|(name, _)| name.trim().eq_ignore_ascii_case("transfer-encoding"));
    let upstream = url
        .as_ref()
        .filter(|_| !chunked)
        .and_then(|url| open(resolver, dialer, url.as_str()));
    let (Some(url), Some(mut upstream)) = (url, upstream) else {
        return client.write_all(REJECTED);
    };
    // One request per connection, so a reused proxy connection can never
    // carry a request to a host that was not checked.
    let mut forwarded = format!(
        "{method} {} {version}\r\n",
        &url[Position::BeforePath..Position::AfterQuery]
    );
    let mut body_length = 0;
    for (name, value) in headers {
        let name = name.trim();
        if [
            "connection",
            "proxy-connection",
            "keep-alive",
            "proxy-authorization",
        ]
        .iter()
        .any(|hop| name.eq_ignore_ascii_case(hop))
        {
            continue;
        }
        if name.eq_ignore_ascii_case("content-length") {
            body_length = value.trim().parse().unwrap_or(0);
        }
        forwarded.push_str(&format!("{name}:{value}\r\n"));
    }
    forwarded.push_str("Connection: close\r\n\r\n");
    upstream.write_all(forwarded.as_bytes())?;
    let early_body = early_bytes.len().min(body_length);
    upstream.write_all(&early_bytes[..early_body])?;
    let mut late_body = (&client).take(u64::try_from(body_length - early_body).unwrap_or(0));
    io::copy(&mut late_body, &mut upstream)?;
    upstream.set_read_timeout(Some(IDLE_TIMEOUT))?;
    io::copy(&mut upstream, &mut client)?;
    Ok(())
}

fn read_head(stream: &mut TcpStream) -> io::Result<(String, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4_096];
    loop {
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let early_bytes = buffer.split_off(end + 4);
            return Ok((String::from_utf8_lossy(&buffer).into_owned(), early_bytes));
        }
        if buffer.len() > MAX_HEAD_BYTES {
            return Err(io::Error::other("request head too large"));
        }
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn open<R: HostResolver, D: Dialer>(resolver: &R, dialer: &D, raw_url: &str) -> Option<TcpStream> {
    let url = Url::parse(raw_url).ok()?;
    let port = url.port_or_known_default()?;
    public_addresses(&url, resolver)?
        .into_iter()
        .find_map(|address| dialer.dial(SocketAddr::new(address, port)).ok())
}

fn relay(client: TcpStream, upstream: TcpStream) -> io::Result<()> {
    upstream.set_read_timeout(Some(IDLE_TIMEOUT))?;
    let mut client_reader = client.try_clone()?;
    let mut upstream_writer = upstream.try_clone()?;
    let upload = thread::spawn(move || {
        let _ = io::copy(&mut client_reader, &mut upstream_writer);
        let _ = upstream_writer.shutdown(Shutdown::Write);
    });
    let (mut upstream_reader, mut client_writer) = (upstream, client);
    let _ = io::copy(&mut upstream_reader, &mut client_writer);
    let _ = client_writer.shutdown(Shutdown::Both);
    let _ = upload.join();
    Ok(())
}

/// Renders pages with a headless Chromium, one at a time, through a
/// [`PublicProxy`].
pub struct ChromiumRenderer {
    command: Vec<OsString>,
    proxy: SocketAddr,
    time_limit: Duration,
    running: Mutex<()>,
}

impl ChromiumRenderer {
    /// `command` is the Chromium program followed by any arguments to put
    /// before the rendering flags.
    #[must_use]
    pub fn new<I, S>(command: I, proxy: SocketAddr, time_limit: Duration) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        Self {
            command: command.into_iter().map(Into::into).collect(),
            proxy,
            time_limit,
            running: Mutex::new(()),
        }
    }
}

impl PageRenderer for ChromiumRenderer {
    fn render(&self, url: &str) -> Result<String, RenderError> {
        let _running = self.running.lock().unwrap_or_else(PoisonError::into_inner);
        let profile = Profile::create().map_err(|error| RenderError::Start(error.to_string()))?;
        let started = Instant::now();
        // `timeout` signals Chromium's whole process group, helpers included.
        let mut command = Command::new("timeout");
        command
            .arg("--kill-after=2")
            .arg(format!("{}s", self.time_limit.as_secs().max(1)))
            .args(["sh", "-c", PIPE_WRAPPER, "chromium-pipe"])
            .args(&self.command)
            .args(CHROMIUM_FLAGS)
            .arg(format!("--proxy-server=http://{}", self.proxy))
            .arg(format!("--user-data-dir={}", profile.0.display()))
            .arg(format!(
                "--disk-cache-dir={}",
                profile.0.join("cache").display()
            ))
            .arg("about:blank");
        let read_by = started + self.time_limit.saturating_sub(READ_MARGIN);
        run_browser(&mut command, Child::wait, |session| {
            session.render(url, read_by)
        })
    }
}

/// Starts the browser, lets `work` drive it over the DevTools pipe, then
/// closes it. A browser that exits early reports its exit status.
fn run_browser(
    command: &mut Command,
    wait: impl FnOnce(&mut Child) -> io::Result<ExitStatus>,
    work: impl FnOnce(&mut DevTools<std::process::ChildStdin>) -> Result<String, DevToolsError>,
) -> Result<String, RenderError> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| RenderError::Start(error.to_string()))?;
    let mut devtools = DevTools::new(child.stdin.take(), read_messages(child.stdout.take()));
    let result = work(&mut devtools);
    devtools.close();
    let status = stop(&mut child, wait);
    match result {
        Ok(html) => Ok(html),
        Err(DevToolsError::Failed(message)) => Err(RenderError::Failed(message)),
        Err(DevToolsError::Closed) => Err(RenderError::Failed(
            status.map_or_else(|error| error.to_string(), |status| status.to_string()),
        )),
    }
}

/// Waits a moment for the browser to exit, then asks `timeout` to stop it.
fn stop(
    child: &mut Child,
    wait: impl FnOnce(&mut Child) -> io::Result<ExitStatus>,
) -> io::Result<ExitStatus> {
    let grace_ends = Instant::now() + EXIT_GRACE;
    while child.try_wait()?.is_none() && Instant::now() < grace_ends {
        thread::sleep(Duration::from_millis(20));
    }
    if child.try_wait()?.is_none() {
        // `timeout` passes the signal on to Chromium.
        let _ = Command::new("sh")
            .args(["-c", "kill -TERM \"$1\"", "stop-browser"])
            .arg(child.id().to_string())
            .status();
    }
    wait(child)
}

/// Reads NUL-separated DevTools messages on a thread, so waits can time out.
/// Messages that are too large or not JSON are skipped.
fn read_messages<R: Read + Send + 'static>(output: Option<R>) -> Receiver<Value> {
    let (sender, receiver) = mpsc::channel();
    let _ = thread::Builder::new()
        .name("browser-devtools".to_owned())
        .spawn(move || -> Option<()> {
            let mut reader = BufReader::new(output?);
            loop {
                let mut message = Vec::new();
                let limit = u64::try_from(MAX_MESSAGE_BYTES).unwrap_or(u64::MAX);
                (&mut reader).take(limit).read_until(0, &mut message).ok()?;
                if message.last() != Some(&0) {
                    if message.len() < MAX_MESSAGE_BYTES {
                        return None;
                    }
                    reader.skip_until(0).ok()?;
                    continue;
                }
                message.pop();
                if let Ok(message) = serde_json::from_slice(&message) {
                    sender.send(message).ok()?;
                }
            }
        });
    receiver
}

#[derive(Debug, PartialEq, Eq)]
enum DevToolsError {
    /// The browser closed the pipe, usually because it exited.
    Closed,
    Failed(String),
}

/// One DevTools connection, with what the page has loaded so far.
struct DevTools<W> {
    input: Option<W>,
    messages: Receiver<Value>,
    next_id: u64,
    responses: HashMap<u64, Value>,
    loaded_at: Option<Instant>,
    in_flight: HashSet<String>,
}

impl<W: Write> DevTools<W> {
    fn new(input: Option<W>, messages: Receiver<Value>) -> Self {
        Self {
            input,
            messages,
            next_id: 0,
            responses: HashMap::new(),
            loaded_at: None,
            in_flight: HashSet::new(),
        }
    }

    /// Loads `url` in a new tab, waits until it settles or `read_by`, and
    /// returns its HTML.
    fn render(&mut self, url: &str, read_by: Instant) -> Result<String, DevToolsError> {
        let target = self.call("Target.createTarget", json!({"url": "about:blank"}), None)?;
        let target = text_at(&target, "/targetId")?;
        let attached = self.call(
            "Target.attachToTarget",
            json!({"targetId": target, "flatten": true}),
            None,
        )?;
        let session = text_at(&attached, "/sessionId")?;
        self.call("Network.enable", json!({}), Some(&session))?;
        self.call("Page.enable", json!({}), Some(&session))?;
        // Load events before this point belong to the blank tab.
        self.loaded_at = None;
        let navigation = self.call("Page.navigate", json!({"url": url}), Some(&session))?;
        if let Some(error) = navigation.get("errorText").and_then(Value::as_str) {
            return Err(DevToolsError::Failed(format!("navigation failed: {error}")));
        }
        self.settle(read_by);
        let expression = format!(
            "(document.documentElement ? document.documentElement.outerHTML : '').slice(0, {FETCH_MAX_BYTES})"
        );
        let evaluated = self.call(
            "Runtime.evaluate",
            json!({"expression": expression, "returnByValue": true}),
            Some(&session),
        )?;
        let html = text_at(&evaluated, "/result/value")?;
        Ok(truncate(html, FETCH_MAX_BYTES))
    }

    /// Waits in real time until the page has loaded and its network has been
    /// quiet for a moment. Scripts get at most `SCRIPT_TIME_BUDGET` after the
    /// load event, and nothing waits past `read_by`, so a page that never
    /// settles is still read.
    fn settle(&mut self, read_by: Instant) {
        loop {
            let ends = self
                .loaded_at
                .map_or(read_by, |loaded| (loaded + SCRIPT_TIME_BUDGET).min(read_by));
            let left = ends.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            let quiet = self.loaded_at.is_some() && self.in_flight.is_empty();
            match self
                .messages
                .recv_timeout(if quiet { NETWORK_QUIET.min(left) } else { left })
            {
                Ok(message) => self.observe(message),
                Err(RecvTimeoutError::Timeout) if !quiet => {}
                Err(_) => return,
            }
        }
    }

    fn call(
        &mut self,
        method: &str,
        params: Value,
        session: Option<&str>,
    ) -> Result<Value, DevToolsError> {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({"id": id, "method": method, "params": params});
        if let Some(session) = session {
            message["sessionId"] = json!(session);
        }
        let mut bytes = message.to_string().into_bytes();
        bytes.push(0);
        self.input
            .as_mut()
            .ok_or(DevToolsError::Closed)?
            .write_all(&bytes)
            .map_err(|_| DevToolsError::Closed)?;
        loop {
            if let Some(response) = self.responses.remove(&id) {
                return match response.get("error") {
                    Some(error) => Err(DevToolsError::Failed(format!(
                        "{method} failed: {}",
                        error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown error")
                    ))),
                    None => Ok(response.get("result").cloned().unwrap_or(Value::Null)),
                };
            }
            let message = self.messages.recv().map_err(|_| DevToolsError::Closed)?;
            self.observe(message);
        }
    }

    fn observe(&mut self, message: Value) {
        if let Some(id) = message.get("id").and_then(Value::as_u64) {
            self.responses.insert(id, message);
            return;
        }
        let request = message
            .pointer("/params/requestId")
            .and_then(Value::as_str)
            .map(str::to_owned);
        match (message.get("method").and_then(Value::as_str), request) {
            (Some("Page.loadEventFired"), _) => self.loaded_at = Some(Instant::now()),
            (Some("Network.requestWillBeSent"), Some(request)) => {
                self.in_flight.insert(request);
            }
            (Some("Network.loadingFinished" | "Network.loadingFailed"), Some(request)) => {
                self.in_flight.remove(&request);
            }
            _ => {}
        }
    }

    /// Asks the browser to quit and closes the pipe.
    fn close(&mut self) {
        if let Some(mut input) = self.input.take() {
            let _ = input.write_all(b"{\"id\":0,\"method\":\"Browser.close\"}\0");
        }
    }
}

fn text_at(value: &Value, pointer: &str) -> Result<String, DevToolsError> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| DevToolsError::Failed(format!("DevTools reply has no {pointer}")))
}

/// `text` cut to at most `limit` bytes on a character boundary.
fn truncate(mut text: String, limit: usize) -> String {
    let mut end = limit.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text
}

/// A fresh Chromium profile directory, deleted when the render ends.
struct Profile(PathBuf);

impl Profile {
    fn create() -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = env::temp_dir().join(format!(
            "respondedorbot-chromium-{}-{}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The process-wide renderer, which runs the `chromium` on `PATH`. `None`
/// when the proxy cannot start.
#[must_use]
pub fn shared_renderer() -> Option<Arc<ChromiumRenderer>> {
    static RENDERER: OnceLock<Option<Arc<ChromiumRenderer>>> = OnceLock::new();
    RENDERER
        .get_or_init(|| {
            PublicProxy::start(SystemHostResolver, TcpDialer)
                .ok()
                .map(|proxy| {
                    Arc::new(ChromiumRenderer::new(
                        ["chromium"],
                        proxy.address(),
                        RENDER_TIME_LIMIT,
                    ))
                })
        })
        .clone()
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::net::IpAddr;
    use std::time::Instant;

    use super::*;

    struct Resolver(Vec<IpAddr>);

    impl HostResolver for Resolver {
        fn addresses(&self, _hostname: &str, _port: u16) -> Result<Vec<IpAddr>, String> {
            Ok(self.0.clone())
        }
    }

    /// Sends every checked address to one local server and records it.
    struct LocalDialer {
        server: SocketAddr,
        dialed: Arc<Mutex<Vec<SocketAddr>>>,
    }

    impl Dialer for LocalDialer {
        fn dial(&self, address: SocketAddr) -> io::Result<TcpStream> {
            self.dialed
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(address);
            TcpStream::connect(self.server)
        }
    }

    const PUBLIC: IpAddr = IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34));

    fn proxy_to(
        server: SocketAddr,
        addresses: Vec<IpAddr>,
    ) -> io::Result<(PublicProxy, Arc<Mutex<Vec<SocketAddr>>>)> {
        let dialed = Arc::new(Mutex::new(Vec::new()));
        let dialer = LocalDialer {
            server,
            dialed: Arc::clone(&dialed),
        };
        let proxy = PublicProxy::start(Resolver(addresses), dialer);
        Ok((proxy?, dialed))
    }

    fn exchange(proxy: &PublicProxy, request: &[u8]) -> io::Result<String> {
        let mut client = TcpStream::connect(proxy.address())?;
        client.write_all(request)?;
        client.shutdown(Shutdown::Write)?;
        let mut response = String::new();
        client.read_to_string(&mut response)?;
        Ok(response)
    }

    fn dialed(record: &Arc<Mutex<Vec<SocketAddr>>>) -> Vec<SocketAddr> {
        record
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    #[test]
    fn tunnels_reach_public_hosts_on_the_checked_address() -> TestResult {
        let server = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let (proxy, record) = proxy_to(server.local_addr()?, vec![PUBLIC])?;
        let upstream = thread::spawn(move || -> io::Result<String> {
            let (mut stream, _) = server.accept()?;
            let mut received = [0_u8; 5];
            stream.read_exact(&mut received)?;
            stream.write_all(b"world")?;
            Ok(String::from_utf8_lossy(&received).into_owned())
        });
        let request = b"CONNECT site.example:8443 HTTP/1.1\r\nHost: site.example:8443\r\n\r\nhello";
        let response = exchange(&proxy, request)?;
        assert_eq!(response, "HTTP/1.1 200 Connection Established\r\n\r\nworld");
        assert_eq!(upstream.join().ok().transpose()?.as_deref(), Some("hello"));
        assert_eq!(dialed(&record), [SocketAddr::new(PUBLIC, 8443)]);
        Ok(())
    }

    const EXPECTED_REQUEST: &str = concat!(
        "POST /path?q=1 HTTP/1.1\r\n",
        "Host: site.example\r\n",
        "Content-Length: 4\r\n",
        "Connection: close\r\n",
        "\r\n",
        "body"
    );

    #[test]
    fn plain_requests_go_out_alone_without_proxy_headers() -> TestResult {
        let server = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let (proxy, record) = proxy_to(server.local_addr()?, vec![PUBLIC])?;
        let upstream = thread::spawn(move || -> io::Result<String> {
            let (mut stream, _) = server.accept()?;
            let mut received = [0_u8; EXPECTED_REQUEST.len()];
            stream.read_exact(&mut received)?;
            stream.write_all(b"HTTP/1.1 200 OK\r\n\r\nok")?;
            Ok(String::from_utf8_lossy(&received).into_owned())
        });
        let request = concat!(
            "POST http://site.example/path?q=1 HTTP/1.1\r\n",
            "Host: site.example\r\n",
            "Proxy-Connection: keep-alive\r\n",
            "Proxy-Authorization: Basic Zm9vOmJhcg==\r\n",
            "Content-Length: 4\r\n",
            "\r\n",
            // The body, then a second request that must not be forwarded.
            "bodyGET http://other.example/ HTTP/1.1\r\n\r\n"
        );
        let response = exchange(&proxy, request.as_bytes())?;
        assert_eq!(response, "HTTP/1.1 200 OK\r\n\r\nok");
        assert_eq!(
            upstream.join().ok().transpose()?.as_deref(),
            Some(EXPECTED_REQUEST)
        );
        assert_eq!(dialed(&record), [SocketAddr::new(PUBLIC, 80)]);
        Ok(())
    }

    #[test]
    fn private_hosts_and_unsupported_requests_are_rejected_without_connecting() -> TestResult {
        let unused = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let private = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let (proxy, record) = proxy_to(unused.local_addr()?, vec![private])?;
        for request in [
            "CONNECT internal.example:443 HTTP/1.1\r\n\r\n",
            "CONNECT 127.0.0.1:5432 HTTP/1.1\r\n\r\n",
            "CONNECT [::1]:443 HTTP/1.1\r\n\r\n",
            "CONNECT localhost:443 HTTP/1.1\r\n\r\n",
            "GET http://internal.example/ HTTP/1.1\r\n\r\n",
            "GET http://169.254.169.254/latest/meta-data HTTP/1.1\r\n\r\n",
            "GET /relative HTTP/1.1\r\n\r\n",
            "GET ftp://files.example/ HTTP/1.1\r\n\r\n",
            "\r\n\r\n",
        ] {
            assert_eq!(
                exchange(&proxy, request.as_bytes())?.as_bytes(),
                REJECTED,
                "{request}"
            );
        }
        assert!(dialed(&record).is_empty());

        let (proxy, record) = proxy_to(unused.local_addr()?, vec![PUBLIC])?;
        let chunked = "POST http://site.example/ HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert_eq!(exchange(&proxy, chunked.as_bytes())?.as_bytes(), REJECTED);
        assert!(dialed(&record).is_empty());

        let (proxy, _) = proxy_to(unused.local_addr()?, Vec::new())?;
        assert_eq!(
            exchange(&proxy, b"CONNECT nowhere.example:443 HTTP/1.1\r\n\r\n")?.as_bytes(),
            REJECTED
        );
        Ok(())
    }

    #[test]
    fn unreachable_hosts_and_broken_requests_get_no_tunnel() -> TestResult {
        let closed = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?.local_addr()?;
        let (proxy, record) = proxy_to(closed, vec![PUBLIC])?;
        assert_eq!(
            exchange(&proxy, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n")?.as_bytes(),
            REJECTED
        );
        assert_eq!(dialed(&record), [SocketAddr::new(PUBLIC, 443)]);

        assert_eq!(
            exchange(&proxy, b"CONNECT site.example:443 HTTP/1.1\r\n")?,
            ""
        );
        assert_eq!(exchange(&proxy, &vec![b'a'; MAX_HEAD_BYTES + 10])?, "");
        assert_eq!(TcpDialer.dial(closed).map(drop).ok(), None);
        Ok(())
    }

    fn fake_chromium(script: &str, time_limit: Duration) -> ChromiumRenderer {
        ChromiumRenderer::new(
            ["sh", "-c", script, "fake-chromium"],
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
            time_limit,
        )
    }

    /// A fake browser answers on fd 4 in the order `DevTools::render` asks.
    const SAY: &str = r#"say() { printf '%s\0' "$1" >&4; }"#;
    const OPENED: &str = r#"
        say '{"id":1,"result":{"targetId":"T"}}'
        say '{"id":2,"result":{"sessionId":"S"}}'
        say '{"id":3,"result":{}}'
        say '{"id":4,"result":{}}'
        say '{"id":5,"result":{"frameId":"F"}}'
        say '{"method":"Network.requestWillBeSent","sessionId":"S","params":{"requestId":"r1"}}'
        say '{"method":"Page.loadEventFired","sessionId":"S","params":{}}'
        say '{"method":"Network.loadingFinished","sessionId":"S","params":{"requestId":"r1"}}'
    "#;

    fn scratch_file(name: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        env::temp_dir().join(format!(
            "respondedorbot-{name}-{}-{}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn renders_through_the_proxy_in_a_profile_that_is_deleted_afterwards() -> TestResult {
        let log = scratch_file("devtools-log");
        let script = format!(
            r#"{SAY}
            for argument; do
                case "$argument" in --user-data-dir=*) profile="${{argument#*=}}" ;; esac
                last="$argument"
            done
            case " $* " in
                *" --headless "*" --remote-debugging-pipe"*" --proxy-server=http://127.0.0.1:9 "*) ;;
                *) exit 3 ;;
            esac
            case " $* " in *" --dump-dom "*) exit 6 ;; esac
            test "$last" = about:blank || exit 5
            test -d "$profile" || exit 4
            {OPENED}
            say "{{\"id\":6,\"result\":{{\"result\":{{\"type\":\"string\",\"value\":\"<title>$profile</title>\"}}}}}}"
            tr '\0' '\n' <&3 > '{}'
            "#,
            log.display()
        );
        let html = fake_chromium(&script, RENDER_TIME_LIMIT).render("https://site.example/app")?;
        let profile = html
            .strip_prefix("<title>")
            .and_then(|rest| rest.strip_suffix("</title>"))
            .map(PathBuf::from);
        assert!(profile.is_some_and(|profile| !profile.exists()), "{html}");
        let requests = fs::read_to_string(&log)?;
        fs::remove_file(&log)?;
        let requests = requests
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<Result<Vec<_>, _>>()?;
        let methods = requests
            .iter()
            .map(|request| request["method"].as_str().unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            methods,
            [
                "Target.createTarget",
                "Target.attachToTarget",
                "Network.enable",
                "Page.enable",
                "Page.navigate",
                "Runtime.evaluate",
                "Browser.close",
            ]
        );
        assert_eq!(requests[1]["params"]["targetId"], "T");
        assert_eq!(requests[4]["params"]["url"], "https://site.example/app");
        assert_eq!(requests[4]["sessionId"], "S");
        assert!(requests[0].get("sessionId").is_none());
        Ok(())
    }

    #[test]
    fn reports_start_failures_and_exit_or_wait_errors() {
        let ignored = |_: &mut DevTools<std::process::ChildStdin>| Err(DevToolsError::Closed);
        let mut missing = Command::new("/nonexistent/respondedorbot-renderer");
        assert!(matches!(
            run_browser(&mut missing, Child::wait, ignored),
            Err(RenderError::Start(_))
        ));

        assert_eq!(
            run_browser(
                &mut Command::new("true"),
                |child| {
                    child.wait()?;
                    Err(io::Error::other("synthetic wait failure"))
                },
                ignored
            ),
            Err(RenderError::Failed("synthetic wait failure".to_owned()))
        );
        assert_eq!(
            run_browser(&mut Command::new("true"), Child::wait, |_| Err(
                DevToolsError::Failed("synthetic".to_owned())
            )),
            Err(RenderError::Failed("synthetic".to_owned()))
        );
    }

    #[test]
    fn failed_slow_stuck_or_huge_renders_are_bounded() {
        assert_eq!(
            fake_chromium("exit 3", RENDER_TIME_LIMIT).render("https://site.example/"),
            Err(RenderError::Failed("exit status: 3".to_owned()))
        );

        let started = Instant::now();
        assert_eq!(
            fake_chromium("sleep 30", Duration::from_secs(1)).render("https://site.example/"),
            Err(RenderError::Failed("exit status: 124".to_owned()))
        );
        assert!(started.elapsed() < Duration::from_secs(10));

        // A browser that ignores the close request is stopped after the grace.
        let stuck = format!(
            r#"{SAY}
            {OPENED}
            say '{{"id":6,"result":{{"result":{{"value":"<p>ok</p>"}}}}}}'
            exec sleep 30
            "#
        );
        let started = Instant::now();
        assert_eq!(
            fake_chromium(&stuck, RENDER_TIME_LIMIT).render("https://site.example/"),
            Ok("<p>ok</p>".to_owned())
        );
        assert!(started.elapsed() < Duration::from_secs(10));

        let huge = format!(
            r#"{SAY}
            {OPENED}
            {{ printf '{{"id":6,"result":{{"result":{{"value":"'
              head -c {} /dev/zero | tr '\0' a
              printf '"}}}}}}\0'
            }} >&4
            cat <&3 >/dev/null
            "#,
            FETCH_MAX_BYTES + 100
        );
        assert_eq!(
            fake_chromium(&huge, RENDER_TIME_LIMIT)
                .render("https://site.example/")
                .map(|html| html.len()),
            Ok(FETCH_MAX_BYTES)
        );
    }

    /// DevTools with `messages` queued. The browser stays connected while
    /// the returned sender lives.
    fn devtools(messages: &[Value]) -> (DevTools<Vec<u8>>, mpsc::Sender<Value>) {
        let (sender, receiver) = mpsc::channel();
        for message in messages {
            let _ = sender.send(message.clone());
        }
        (DevTools::new(Some(Vec::new()), receiver), sender)
    }

    fn written(devtools: &DevTools<Vec<u8>>) -> Vec<Value> {
        devtools
            .input
            .as_deref()
            .unwrap_or_default()
            .split(|byte| *byte == 0)
            .filter(|message| !message.is_empty())
            .filter_map(|message| serde_json::from_slice(message).ok())
            .collect()
    }

    struct BrokenPipe;

    impl Write for BrokenPipe {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn calls_return_results_errors_and_closed_pipes() {
        let (mut tools, browser) = devtools(&[
            json!({"method": "Page.frameNavigated", "params": {}}),
            json!({"id": 1, "result": {"ok": true}}),
            json!({"id": 2}),
            json!({"id": 3, "error": {"message": "boom"}}),
            json!({"id": 4, "error": {}}),
        ]);
        assert_eq!(
            tools.call("A.one", json!({"x": 1}), Some("S")),
            Ok(json!({"ok": true}))
        );
        assert_eq!(tools.call("A.two", json!({}), None), Ok(Value::Null));
        assert_eq!(
            tools.call("A.three", json!({}), None),
            Err(DevToolsError::Failed("A.three failed: boom".to_owned()))
        );
        assert_eq!(
            tools.call("A.four", json!({}), None),
            Err(DevToolsError::Failed(
                "A.four failed: unknown error".to_owned()
            ))
        );
        drop(browser);
        assert_eq!(
            tools.call("A.five", json!({}), None),
            Err(DevToolsError::Closed)
        );
        let sent = written(&tools);
        assert_eq!(
            sent[0],
            json!({"id": 1, "method": "A.one", "params": {"x": 1}, "sessionId": "S"})
        );
        assert_eq!(sent.len(), 5);

        tools.close();
        assert!(tools.input.is_none());
        tools.close();
        assert_eq!(
            tools.call("A.six", json!({}), None),
            Err(DevToolsError::Closed)
        );

        let (_sender, receiver) = mpsc::channel();
        let mut broken = DevTools::new(Some(BrokenPipe), receiver);
        assert_eq!(
            broken.call("A.one", json!({}), None),
            Err(DevToolsError::Closed)
        );
        broken.close();
    }

    /// A pipe end whose bytes the test can still read after it is dropped.
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn closing_asks_the_browser_to_quit_once() {
        let pipe = Shared::default();
        let mut tools = DevTools::new(Some(pipe.clone()), mpsc::channel().1);
        tools.close();
        tools.close();
        assert_eq!(
            pipe.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_slice(),
            b"{\"id\":0,\"method\":\"Browser.close\"}\0"
        );
    }

    fn opened(session_reply: Value) -> Vec<Value> {
        vec![
            json!({"id": 1, "result": {"targetId": "T"}}),
            json!({"id": 2, "result": session_reply}),
            json!({"id": 3, "result": {}}),
            json!({"id": 4, "result": {}}),
        ]
    }

    #[test]
    fn render_reports_missing_replies_and_navigation_errors() {
        let soon = || Instant::now() + Duration::from_millis(50);
        let (mut no_target, _browser) = devtools(&[json!({"id": 1, "result": {}})]);
        assert_eq!(
            no_target.render("https://site.example/", soon()),
            Err(DevToolsError::Failed(
                "DevTools reply has no /targetId".to_owned()
            ))
        );

        let mut replies = opened(json!({"sessionId": "S"}));
        replies.push(json!({"id": 5, "result": {"errorText": "net::ERR_NAME_NOT_RESOLVED"}}));
        assert_eq!(
            devtools(&replies).0.render("https://site.example/", soon()),
            Err(DevToolsError::Failed(
                "navigation failed: net::ERR_NAME_NOT_RESOLVED".to_owned()
            ))
        );

        let mut replies = opened(json!({"sessionId": "S"}));
        replies.push(json!({"id": 5, "result": {}}));
        replies.push(json!({"id": 6, "result": {"result": {"type": "undefined"}}}));
        assert_eq!(
            devtools(&replies).0.render("https://site.example/", soon()),
            Err(DevToolsError::Failed(
                "DevTools reply has no /result/value".to_owned()
            ))
        );

        // A load event from the blank tab does not count for the new page.
        let mut replies = opened(json!({"sessionId": "S"}));
        replies.insert(3, json!({"method": "Page.loadEventFired", "params": {}}));
        replies.push(json!({"id": 5, "result": {}}));
        replies.push(json!({"id": 6, "result": {"result": {"value": "añb"}}}));
        let (mut tools, _browser) = devtools(&replies);
        let started = Instant::now();
        assert_eq!(
            tools.render(
                "https://site.example/",
                started + Duration::from_millis(300)
            ),
            Ok("añb".to_owned())
        );
        assert!(tools.loaded_at.is_none());
        assert!(started.elapsed() >= Duration::from_millis(250));
    }

    #[test]
    fn settling_waits_for_load_and_quiet_but_never_past_the_deadline() {
        // Never loads: read at the deadline.
        let (mut unloaded, _browser) = devtools(&[]);
        let started = Instant::now();
        unloaded.settle(started + Duration::from_millis(200));
        assert!(started.elapsed() >= Duration::from_millis(150));

        // Loaded with a request that never ends: also read at the deadline.
        let (mut busy, _busy_browser) = devtools(&[
            json!({"method": "Network.requestWillBeSent", "params": {"requestId": "beacon"}}),
            json!({"method": "Network.requestWillBeSent", "params": {}}),
            json!({"method": "Page.loadEventFired", "params": {}}),
        ]);
        let started = Instant::now();
        busy.settle(started + Duration::from_millis(300));
        assert!(started.elapsed() >= Duration::from_millis(250));
        assert_eq!(busy.in_flight.len(), 1);

        // Loaded and quiet: read after the quiet period, before the deadline.
        let (mut quiet, _quiet_browser) = devtools(&[
            json!({"method": "Network.requestWillBeSent", "params": {"requestId": "data"}}),
            json!({"method": "Page.loadEventFired", "params": {}}),
            json!({"method": "Network.loadingFailed", "params": {"requestId": "data"}}),
        ]);
        let started = Instant::now();
        quiet.settle(started + Duration::from_secs(10));
        let waited = started.elapsed();
        assert!(waited >= NETWORK_QUIET && waited < Duration::from_secs(5));
        assert!(quiet.in_flight.is_empty());

        // A browser that went away is read at once.
        let mut gone = DevTools::new(Some(Vec::new()), mpsc::channel().1);
        let started = Instant::now();
        gone.settle(started + Duration::from_secs(10));
        assert!(started.elapsed() < Duration::from_secs(1));

        // A deadline already past returns without waiting.
        let (mut late, _late_browser) = devtools(&[]);
        late.settle(Instant::now());
    }

    #[test]
    fn message_reader_skips_oversized_and_invalid_messages() {
        let mut stream = b"{\"id\":1}\0not json\0".to_vec();
        stream.extend(std::iter::repeat_n(b'x', MAX_MESSAGE_BYTES + 10));
        stream.extend_from_slice(b"\0{\"id\":2}\0{\"id\":3,\"cut");
        let messages = read_messages(Some(io::Cursor::new(stream)));
        assert_eq!(messages.recv().ok(), Some(json!({"id": 1})));
        assert_eq!(messages.recv().ok(), Some(json!({"id": 2})));
        assert!(messages.recv().is_err());

        assert!(read_messages(None::<io::Empty>).recv().is_err());

        // The reader stops when nobody listens any more.
        drop(read_messages(Some(io::Cursor::new(
            b"{\"id\":1}\0".to_vec(),
        ))));
    }

    #[test]
    fn truncation_keeps_whole_characters() {
        assert_eq!(truncate("añb".to_owned(), 2), "a");
        assert_eq!(truncate("añb".to_owned(), 3), "añ");
        assert_eq!(truncate("ab".to_owned(), 10), "ab");
    }

    #[test]
    fn the_shared_renderer_is_built_once() {
        let first = shared_renderer();
        assert!(first.is_some());
        assert!(
            first
                .zip(shared_renderer())
                .is_some_and(|(first, second)| Arc::ptr_eq(&first, &second))
        );
    }
}

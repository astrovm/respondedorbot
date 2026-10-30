//! Headless Chromium rendering for pages that build their content with
//! JavaScript. Chromium reaches the network only through a local proxy that
//! applies the same public-address checks as `web_fetch`, so a page cannot
//! make it load internal services.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;
use std::{env, fs, process, thread};

use thiserror::Error;
use url::{Position, Url};

use crate::web_fetch::{FETCH_MAX_BYTES, HostResolver, SystemHostResolver, public_addresses};

/// How long one render may take, page load and scripts included.
pub const RENDER_TIME_LIMIT: Duration = Duration::from_secs(20);
/// How long Chromium lets scripts and timers run after the page starts
/// loading. Virtual time pauses while requests are pending, so data fetched
/// by scripts still arrives.
const SCRIPT_TIME_BUDGET_MS: u32 = 8_000;
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
    "--dump-dom",
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
        // `timeout` signals Chromium's whole process group, helpers included.
        let mut child = Command::new("timeout")
            .arg("--kill-after=2")
            .arg(format!("{}s", self.time_limit.as_secs().max(1)))
            .args(&self.command)
            .args(CHROMIUM_FLAGS)
            .arg(format!("--virtual-time-budget={SCRIPT_TIME_BUDGET_MS}"))
            .arg(format!("--proxy-server=http://{}", self.proxy))
            .arg(format!("--user-data-dir={}", profile.0.display()))
            .arg(format!(
                "--disk-cache-dir={}",
                profile.0.join("cache").display()
            ))
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| RenderError::Start(error.to_string()))?;
        let mut html = Vec::new();
        if let Some(mut stdout) = child.stdout.take() {
            let limit = u64::try_from(FETCH_MAX_BYTES).unwrap_or(u64::MAX);
            let _ = (&mut stdout).take(limit).read_to_end(&mut html);
            let _ = io::copy(&mut stdout, &mut io::sink());
        }
        let status = child
            .wait()
            .map_err(|error| RenderError::Failed(error.to_string()))?;
        if !status.success() {
            return Err(RenderError::Failed(status.to_string()));
        }
        Ok(String::from_utf8_lossy(&html).into_owned())
    }
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

    #[test]
    fn renders_through_the_proxy_in_a_profile_that_is_deleted_afterwards() -> TestResult {
        let script = r#"
            for argument; do
                case "$argument" in --user-data-dir=*) profile="${argument#*=}" ;; esac
                url="$argument"
            done
            case " $* " in
                *" --headless "*" --dump-dom "*" --proxy-server=http://127.0.0.1:9 "*) ;;
                *) exit 3 ;;
            esac
            test -d "$profile" || exit 4
            printf '<title>%s</title><p>%s</p>' "$profile" "$url"
        "#;
        let html = fake_chromium(script, RENDER_TIME_LIMIT).render("https://site.example/app")?;
        let profile = html
            .strip_prefix("<title>")
            .and_then(|rest| rest.split_once("</title>"))
            .map(|(profile, _)| PathBuf::from(profile));
        assert!(html.ends_with("<p>https://site.example/app</p>"), "{html}");
        assert!(profile.is_some_and(|profile| !profile.exists()));
        Ok(())
    }

    #[test]
    fn failed_slow_or_huge_renders_are_bounded() {
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

        let huge = format!("head -c {} /dev/zero | tr '\\0' a", FETCH_MAX_BYTES + 100);
        assert_eq!(
            fake_chromium(&huge, RENDER_TIME_LIMIT)
                .render("https://site.example/")
                .map(|html| html.len()),
            Ok(FETCH_MAX_BYTES)
        );
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

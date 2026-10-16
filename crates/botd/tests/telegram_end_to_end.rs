//! Runs the real `botd` binary against fake Telegram and OpenRouter servers,
//! with real PostgreSQL and Redis, and checks what users would see.

use std::collections::VecDeque;
use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::billing_schema::BillingSchemaRepository;
use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn Error>>;

const TOKEN: &str = "424242:synthetic-e2e-token";
const BOT_USERNAME: &str = "synthetic_e2e_bot";
const AI_REPLY: &str = "Respuesta sintética del modelo";
const STARTING_CREDITS: i32 = 1_000;

#[derive(Debug, Clone)]
struct Recorded {
    path: String,
    body: Value,
}

#[derive(Default)]
struct FakeState {
    requests: Mutex<Vec<Recorded>>,
    pending_updates: Mutex<VecDeque<Value>>,
    next_message_id: AtomicI64,
    stopping: AtomicBool,
}

impl FakeState {
    fn requests(&self) -> Vec<Recorded> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .unwrap_or_default()
    }

    fn telegram_calls(&self, method: &str) -> Vec<Value> {
        let suffix = format!("/bot{TOKEN}/{method}");
        self.requests()
            .into_iter()
            .filter(|request| request.path.split('?').next() == Some(suffix.as_str()))
            .map(|request| request.body)
            .collect()
    }

    fn sent_texts(&self, chat_id: i64) -> Vec<String> {
        ["sendMessage", "editMessageText"]
            .iter()
            .flat_map(|method| self.telegram_calls(method))
            .filter(|body| body["chat_id"].as_i64() == Some(chat_id))
            .filter_map(|body| body["text"].as_str().map(str::to_owned))
            .collect()
    }
}

struct FakeServer {
    address: String,
    state: Arc<FakeState>,
}

impl FakeServer {
    fn start() -> Result<Self, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?.to_string();
        let state = Arc::new(FakeState {
            next_message_id: AtomicI64::new(1_000),
            ..FakeState::default()
        });
        let accept_state = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let state = Arc::clone(&accept_state);
                thread::spawn(move || serve_connection(stream, &state));
            }
        });
        Ok(Self { address, state })
    }

    fn queue_update(&self, update: Value) {
        if let Ok(mut pending) = self.state.pending_updates.lock() {
            pending.push_back(update);
        }
    }

    fn wait_for(&self, what: &str, condition: impl Fn(&FakeState) -> bool) -> TestResult {
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if condition(&self.state) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(50));
        }
        let seen = self
            .state
            .requests()
            .iter()
            .map(|request| format!("{} {}", request.path, request.body))
            .collect::<Vec<_>>()
            .join("\n");
        Err(format!("timed out waiting for {what}; requests:\n{seen}").into())
    }
}

fn serve_connection(stream: TcpStream, state: &FakeState) {
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(read_half);
    let mut writer = stream;
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or_default() == 0 {
        return;
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_owned();
    let mut content_length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or_default() <= 2 {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().unwrap_or_default();
        }
    }
    let mut body = vec![0; content_length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    let body = serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).into_owned()));
    if let Ok(mut requests) = state.requests.lock() {
        requests.push(Recorded {
            path: path.clone(),
            body: body.clone(),
        });
    }
    let (content_type, response) = route(&path, &body, state);
    let _result = write!(
        writer,
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response}",
        response.len()
    );
    let _result = writer.flush();
}

fn route(path: &str, body: &Value, state: &FakeState) -> (&'static str, String) {
    let path_only = path.split('?').next().unwrap_or_default();
    if let Some(method) = path_only.strip_prefix(&format!("/bot{TOKEN}/")) {
        return ("application/json", telegram_response(method, body, state));
    }
    if path_only.ends_with("/chat/completions") {
        return openrouter_completion(body);
    }
    if path_only.ends_with("/generation") {
        return (
            "application/json",
            json!({"data": {"id": "synthetic-generation", "total_cost": 0.000_01,
                "native_tokens_prompt": 12, "native_tokens_completion": 6}})
            .to_string(),
        );
    }
    if path_only.ends_with("/endpoints") {
        return (
            "application/json",
            json!({"data": {"endpoints": []}}).to_string(),
        );
    }
    if path_only.ends_with("/models") {
        let pricing = json!({"prompt": "0.0000003", "completion": "0.0000012"});
        let models = [
            "deepseek/deepseek-v4.1-flash",
            "deepseek/deepseek-v4-flash",
            "microsoft/mai-transcribe-2",
        ]
        .map(|id| json!({"id": id, "pricing": pricing}));
        return ("application/json", json!({"data": models}).to_string());
    }
    ("application/json", json!({"data": []}).to_string())
}

fn telegram_response(method: &str, body: &Value, state: &FakeState) -> String {
    let result = match method {
        "getUpdates" => {
            let updates = state
                .pending_updates
                .lock()
                .map(|mut pending| pending.drain(..).collect::<Vec<_>>())
                .unwrap_or_default();
            if updates.is_empty() && !state.stopping.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(100));
            }
            json!(updates)
        }
        method if method.starts_with("send") || method.starts_with("edit") => {
            let message_id = state.next_message_id.fetch_add(1, Ordering::SeqCst);
            json!({
                "message_id": message_id,
                "date": now(),
                "chat": {"id": body["chat_id"], "type": "private"},
                "from": {"id": 424_242, "is_bot": true, "first_name": "Synthetic",
                    "username": BOT_USERNAME},
                "text": body["text"],
            })
        }
        _ => json!(true),
    };
    json!({"ok": true, "result": result}).to_string()
}

fn openrouter_completion(body: &Value) -> (&'static str, String) {
    let usage = json!({"prompt_tokens": 12, "completion_tokens": 6, "cost": 0.000_01});
    if body["stream"] == true {
        let chunk = json!({"id": "synthetic-generation", "model": "synthetic/model",
            "choices": [{"delta": {"content": AI_REPLY}, "finish_reason": null}]});
        let last = json!({"id": "synthetic-generation", "model": "synthetic/model",
            "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage});
        return (
            "text/event-stream",
            format!("data: {chunk}\n\ndata: {last}\n\ndata: [DONE]\n\n"),
        );
    }
    (
        "application/json",
        json!({"id": "synthetic-generation", "model": "synthetic/model",
            "choices": [{"message": {"content": AI_REPLY}, "finish_reason": "stop"}],
            "usage": usage})
        .to_string(),
    )
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX)
        })
}

fn private_message(update_id: i64, user_id: i64, text: &str) -> Value {
    json!({
        "update_id": update_id,
        "message": {
            "message_id": update_id % 1_000_000,
            "date": now(),
            "chat": {"id": user_id, "type": "private", "first_name": "Synthetic"},
            "from": {"id": user_id, "is_bot": false, "first_name": "Synthetic",
                "username": "synthetic_user", "language_code": "es"},
            "text": text,
            "entities": text.starts_with('/').then(|| json!([{
                "type": "bot_command", "offset": 0,
                "length": text.split_whitespace().next().unwrap_or_default().len()
            }])),
        }
    })
}

struct Workspace(std::path::PathBuf);

impl Workspace {
    fn new() -> Result<Self, std::io::Error> {
        let path = std::env::temp_dir().join(format!("botd-e2e-{}", std::process::id()));
        std::fs::create_dir_all(path.join("workspace"))?;
        std::fs::write(path.join("workspace/SOUL.md"), "synthetic personality")?;
        std::fs::write(path.join("workspace/RULES.md"), "synthetic rules")?;
        Ok(Self(path))
    }
}

impl Workspace {
    fn log(&self) -> std::path::PathBuf {
        self.0.join("botd.log")
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _result = std::fs::remove_dir_all(&self.0);
    }
}

struct Bot(Child);

impl Drop for Bot {
    fn drop(&mut self) {
        let _result = self.0.kill();
        let _result = self.0.wait();
    }
}

fn start_bot(
    server: &FakeServer,
    workspace: &Workspace,
    database_url: &str,
    redis_host: &str,
    redis_port: &str,
) -> Result<Bot, std::io::Error> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_botd"));
    command.env_clear();
    if let Some(profile) = std::env::var_os("LLVM_PROFILE_FILE") {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    let base = format!("http://{}", server.address);
    command
        .current_dir(&workspace.0)
        .env("TELEGRAM_TOKEN", TOKEN)
        .env("TELEGRAM_USERNAME", BOT_USERNAME)
        .env("TELEGRAM_API_URL", &base)
        .env("TELEGRAM_LONG_POLL_SECONDS", "1")
        .env("OPENROUTER_BASE_URL", format!("{base}/api/v1"))
        .env("OPENROUTER_API_KEY", "synthetic-ai-key")
        .env("COINMARKETCAP_KEY", "synthetic-market-key")
        .env("SUPABASE_POSTGRES_URL", database_url)
        .env("REDIS_HOST", redis_host)
        .env("REDIS_PORT", redis_port)
        .env("BOT_INSTANCE_NAME", "synthetic-e2e")
        // Anything that is not one of the fakes fails instead of leaving the machine.
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(workspace.log())?);
    command.spawn().map(Bot)
}

fn help_button_press(update_id: i64, user_id: i64) -> Value {
    json!({
        "update_id": update_id,
        "callback_query": {
            "id": format!("synthetic-callback-{update_id}"),
            "chat_instance": "synthetic-instance",
            "data": "help:ai",
            "from": {"id": user_id, "is_bot": false, "first_name": "Synthetic",
                "language_code": "es"},
            "message": {
                "message_id": 555,
                "date": now(),
                "chat": {"id": user_id, "type": "private", "first_name": "Synthetic"},
                "from": {"id": 424_242, "is_bot": true, "first_name": "Synthetic",
                    "username": BOT_USERNAME},
                "text": "help",
            },
        }
    })
}

fn stop_bot(bot: &mut Bot) -> TestResult {
    let status = Command::new("kill")
        .args(["-INT", &bot.0.id().to_string()])
        .status()?;
    assert!(status.success());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(exit) = bot.0.try_wait()? {
            assert!(exit.success(), "{exit:?}");
            return Ok(());
        }
        assert!(Instant::now() < deadline, "botd did not stop after SIGINT");
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn real_binary_answers_commands_buttons_and_ai_chat_then_shuts_down_cleanly() -> TestResult {
    let Ok(database_url) = std::env::var("TEST_DATABASE_URL") else {
        return Ok(());
    };
    let Ok(redis_port) = std::env::var("TEST_REDIS_PORT") else {
        return Ok(());
    };
    let redis_host = std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned());
    // A user of its own per run, funded directly, so reruns against the same
    // database don't depend on the shared hourly onboarding-grant limit.
    let user_id = 8_300_000_000_000 + now() % 1_000_000_000;
    BillingSchemaRepository::new(&database_url).ensure_schema()?;
    let billing = BillingRepository::new(&database_url);
    let funded = billing.mint_user_credits(
        user_id,
        STARTING_CREDITS,
        None,
        &format!("e2e-mint:{user_id}"),
    )?;
    assert_eq!(funded, i64::from(STARTING_CREDITS));

    let server = FakeServer::start()?;
    let workspace = Workspace::new()?;
    let mut bot = start_bot(&server, &workspace, &database_url, &redis_host, &redis_port)?;
    exercise_main_flows(&server, &mut bot, &billing, user_id).map_err(|error| {
        let log = std::fs::read_to_string(workspace.log()).unwrap_or_default();
        format!("{error}\nbotd stderr:\n{log}").into()
    })
}

fn exercise_main_flows(
    server: &FakeServer,
    bot: &mut Bot,
    billing: &BillingRepository,
    user_id: i64,
) -> TestResult {
    // Update IDs only grow, like Telegram's, so a rerun is never mistaken for
    // updates the shared Redis already saw.
    let first_update = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX / 100)
        })
        * 10;

    server.queue_update(private_message(first_update, user_id, "/help"));
    server.queue_update(private_message(
        first_update + 1,
        user_id,
        "/convertbase 101, 2, 10",
    ));
    server.queue_update(private_message(first_update + 2, user_id, "/ask hola"));

    // Startup publishes the Spanish and English command menus.
    server.wait_for("command menus", |state| {
        state.telegram_calls("setMyCommands").len() >= 2
    })?;
    let menus = server.state.telegram_calls("setMyCommands");
    assert!(menus.iter().any(|menu| {
        menu["commands"]
            .as_str()
            .is_some_and(|commands| commands.contains("preguntame lo que quieras"))
    }));

    // /help answers with the help menu buttons.
    server.wait_for("the help menu", |state| {
        state.telegram_calls("sendMessage").iter().any(|body| {
            body["chat_id"].as_i64() == Some(user_id)
                && body["reply_markup"].to_string().contains("help:ai")
        })
    })?;

    // A plain command answers without the AI.
    server.wait_for("the base conversion", |state| {
        state
            .sent_texts(user_id)
            .iter()
            .any(|text| text.ends_with("101 en base 2 es 5 en base 10"))
    })?;

    // /ask streams the model's answer into the placeholder message.
    server.wait_for("an AI reply", |state| {
        state
            .telegram_calls("editMessageText")
            .iter()
            .any(|body| body["chat_id"].as_i64() == Some(user_id) && body["text"] == AI_REPLY)
    })?;
    // The real scheduler also runs here and may pick up tasks other tests left
    // in the shared Redis, so only count requests for this conversation.
    let completions = server
        .state
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/chat/completions"))
        .collect::<Vec<_>>();
    let ours = completions
        .iter()
        .filter(|request| {
            request.body["messages"]
                .as_array()
                .and_then(|messages| messages.last())
                .is_some_and(|last| last["content"].to_string().contains("hola"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        ours.len(),
        1,
        "{:#?}",
        completions
            .iter()
            .map(|request| request.body["messages"].to_string())
            .collect::<Vec<_>>()
    );
    let prompt = ours[0].body["messages"].to_string();
    assert!(prompt.contains("synthetic personality"), "{prompt}");
    assert!(prompt.contains("synthetic rules"), "{prompt}");
    assert!(prompt.contains("hola"), "{prompt}");

    // The answer was paid for with the user's credits.
    let deadline = Instant::now() + Duration::from_secs(30);
    while billing.get_balance("user", user_id)? >= i64::from(STARTING_CREDITS) {
        assert!(Instant::now() < deadline, "the AI reply was never charged");
        thread::sleep(Duration::from_millis(50));
    }

    // Pressing a help button edits the menu and answers the button press.
    server.queue_update(help_button_press(first_update + 3, user_id));
    server.wait_for("the button press answer", |state| {
        !state.telegram_calls("answerCallbackQuery").is_empty()
            && state
                .telegram_calls("editMessageText")
                .iter()
                .any(|body| body["message_id"] == 555)
    })?;

    server.state.stopping.store(true, Ordering::SeqCst);
    stop_bot(bot)
}

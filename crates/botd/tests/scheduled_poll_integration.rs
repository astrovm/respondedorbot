use std::collections::BTreeMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::openrouter_chat::{
    HttpRequest, HttpResponse, OpenRouterChatError, OpenRouterTransport,
};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_poll_store::RedisPollStore;
use bot_adapters::redis_task_store::{RedisTaskStore, TASK_SCHEDULER_OWNER_KEY};
use bot_adapters::task_record::TaskRecordDocument;
use bot_adapters::telegram_http::{
    HttpResponse as TelegramResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
};
use bot_core::locale::Locale;
use bot_core::polls::{PollAnswer, PollVote};
use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};
use botd::composition::TelegramActionSink;
use botd::native_ai::{ActionTaskMessenger, OpenRouterTaskProvider};
use botd::poll_tools::{PollTool, PollToolContext, TelegramPollSender};
use botd::scheduler::{SchedulerMode, SchedulerSettings, SchedulerStep, TaskScheduler};
use botd::task_executor::{
    NativeTaskExecutor, StderrTaskDiagnostics, TaskBilling, TaskPromptMessage, TaskReserveOutcome,
};
use serde_json::{Value, json};

#[derive(Clone)]
struct Transport(Arc<Mutex<Vec<Value>>>, bool);

impl OpenRouterTransport for Transport {
    fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
        let body: Value = serde_json::from_str(&request.body)
            .map_err(|error| OpenRouterChatError::Transport(error.to_string()))?;
        assert!(body["tools"].as_array().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| tool["function"]["name"] == "create_poll")
        }));
        let mut events = self
            .0
            .lock()
            .map_err(|error| OpenRouterChatError::Transport(error.to_string()))?;
        let repairing = events.iter().any(|event| event["kind"] == "ai");
        if repairing {
            assert_eq!(
                body["messages"][2]["content"],
                "situación ficticia: encontrás una billetera en la calle"
            );
            assert_eq!(body["messages"][3]["tool_call_id"], "poll-call");
            assert_eq!(body["messages"][3]["content"], "hay opciones repetidas");
        }
        events.push(json!({"kind": "ai"}));
        let options = if self.1 && !repairing {
            json!(["Devolverla", "Devolverla"])
        } else {
            json!(["Devolverla", "Quedártela"])
        };
        Ok(HttpResponse {
            status_code: 200,
            body: json!({
                "id": "synthetic-generation", "model": "synthetic/model",
                "choices": [{"message": {
                    "content": "situación ficticia: encontrás una billetera en la calle",
                    "tool_calls": [{"id": "poll-call", "type": "function", "function": {
                        "name": "create_poll", "arguments": json!({
                            "question": "Encontrás una billetera. ¿Qué hacés?",
                            "options": options
                        }).to_string()
                    }}]
                }, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 34}
            })
            .to_string(),
            headers: BTreeMap::new(),
        })
    }
}

impl TelegramTransport for Transport {
    fn send(&self, request: &TelegramRequest) -> Result<TelegramResponse, TransportFailureKind> {
        let mut events = self.0.lock().map_err(|_| TransportFailureKind::Request)?;
        let payload = request.json_payload.clone().unwrap_or(Value::Null);
        events.push(json!({"kind": request.endpoint, "payload": payload}));
        let poll_number = events
            .iter()
            .filter(|event| event["kind"] == "sendPoll")
            .count();
        Ok(TelegramResponse {
            status_code: 200,
            body: json!({"ok": true, "result": {
                "message_id": poll_number as i64 + 100,
                "poll": {"id": format!("synthetic-poll-{poll_number}")}
            }})
            .to_string(),
        })
    }
}

struct Billing(Arc<Mutex<Vec<Value>>>);

impl TaskBilling for Billing {
    type Error = String;

    fn reserve(
        &mut self,
        _task: &ScheduledTask,
        _execution_id: &str,
        _prompt: &[TaskPromptMessage],
    ) -> Result<TaskReserveOutcome, String> {
        self.0
            .lock()
            .map_err(|error| error.to_string())?
            .push(json!({"kind": "reserve"}));
        Ok(TaskReserveOutcome::Authorized)
    }

    fn settle(
        &mut self,
        _task: &ScheduledTask,
        _execution_id: &str,
        segments: &[Value],
        _reason: &'static str,
    ) -> Result<(), String> {
        let mut events = self.0.lock().map_err(|error| error.to_string())?;
        assert_eq!(
            segments.len(),
            events.iter().filter(|event| event["kind"] == "ai").count()
        );
        events.push(json!({"kind": "settle"}));
        Ok(())
    }

    fn refund(
        &mut self,
        _task: &ScheduledTask,
        _execution_id: &str,
        _reason: &'static str,
    ) -> Result<(), String> {
        Err("unexpected refund".to_owned())
    }
}

#[test]
fn weekly_poll_runs_through_scheduler_delivery_vote_tracking_and_restart()
-> Result<(), Box<dyn Error>> {
    run_weekly_poll(false)?;
    run_weekly_poll(true)
}

fn run_weekly_poll(repair: bool) -> Result<(), Box<dyn Error>> {
    let Some(port) = std::env::var("TEST_REDIS_PORT").ok() else {
        return Ok(());
    };
    let endpoint = RedisEndpoint {
        host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
        port: port.parse()?,
        password: std::env::var("TEST_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
    };
    let now = i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())?;
    let id = format!(
        "synthetic-weekly-poll-{}-{now}-{repair}",
        std::process::id()
    );
    let chat_id = -9_000_000_000 - i64::from(std::process::id()) * 2 - i64::from(repair);
    let task = ScheduledTask {
        id: TaskId::new(&id)?, chat_id: chat_id.to_string(),
        text: "hacer una encuesta semanal en el chat planteando una situación ficticia con dilema moral para que voten".to_owned(),
        user_name: "synthetic-user".to_owned(), user_id: Some(4242),
        schedule: TaskSchedule::IntervalSeconds { seconds: 604_800 },
        timezone_offset: -3, locale: "es".to_owned(), schedule_anchor_at: Some(now),
        next_run_at: Some(now), last_execution_id: None,
    };
    let store = RedisTaskStore::new(&endpoint)?;
    store.delete(TASK_SCHEDULER_OWNER_KEY)?;
    store.save_task(
        &TaskRecordDocument {
            task: task.clone(),
            run_date: None,
            extra: BTreeMap::new(),
        },
        700_000,
    )?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let make_executor = || -> Result<_, Box<dyn Error>> {
        let transport = Transport(Arc::clone(&events), repair);
        Ok(NativeTaskExecutor::new(
            OpenRouterTaskProvider::new(
                transport.clone(),
                "synthetic-key",
                "https://synthetic.invalid/api/v1",
                "synthetic/model",
                "synthetic persona",
            ),
            Billing(Arc::clone(&events)),
            ActionTaskMessenger::new(TelegramActionSink::new(
                transport.clone(),
                "synthetic-token",
            ))
            .with_polls(Box::new(PollTool::new(
                TelegramPollSender::with_transport(transport, "synthetic-token"),
                RedisPollStore::new(&endpoint)?,
                move || now,
                PollToolContext {
                    chat_id: 0,
                    reply_to_message_id: None,
                    locale: Locale::En,
                },
            ))),
            RedisTaskStore::new(&endpoint)?,
            StderrTaskDiagnostics,
        ))
    };
    let mut scheduler = TaskScheduler::new(
        RedisTaskStore::new(&endpoint)?,
        make_executor()?,
        SchedulerMode::Authoritative,
        SchedulerSettings::default(),
        format!("synthetic-owner-{id}"),
    )?;
    let SchedulerStep::Observed { failures, .. } = scheduler.step(now)? else {
        return Err("scheduler did not acquire ownership".into());
    };
    assert!(failures.is_empty(), "{failures:?}");
    let next = store.load_task(&id)?.ok_or("weekly task missing")?;
    assert_eq!(next.task.next_run_at, Some(now + 604_800));
    assert!(next.task.last_execution_id.is_some());
    assert!(scheduler.shutdown()?);
    let mut restarted = TaskScheduler::new(
        RedisTaskStore::new(&endpoint)?,
        make_executor()?,
        SchedulerMode::Authoritative,
        SchedulerSettings::default(),
        format!("synthetic-restarted-owner-{id}"),
    )?;
    let SchedulerStep::Observed { failures, .. } = restarted.step(now)? else {
        return Err("restarted scheduler did not acquire ownership".into());
    };
    assert!(failures.is_empty(), "{failures:?}");
    assert!(restarted.shutdown()?);

    let captured = events.lock().map_err(|error| error.to_string())?.clone();
    assert_eq!(
        captured
            .iter()
            .map(|event| event["kind"].as_str().unwrap_or_default())
            .collect::<Vec<_>>(),
        if repair {
            vec!["reserve", "ai", "ai", "settle", "sendMessage", "sendPoll"]
        } else {
            vec!["reserve", "ai", "settle", "sendMessage", "sendPoll"]
        }
    );
    let payload = &captured.last().ok_or("poll delivery missing")?["payload"];
    assert_eq!(payload["chat_id"], chat_id);
    assert_eq!(
        payload["options"],
        json!([{"text": "Devolverla"}, {"text": "Quedártela"}])
    );
    assert_eq!(payload["is_anonymous"], false);
    assert_eq!(payload["allows_multiple_answers"], false);
    assert!(payload.get("reply_parameters").is_none());
    let polls = RedisPollStore::new(&endpoint)?;
    assert!(polls.record_answer(&PollAnswer {
        poll_id: "synthetic-poll-1".to_owned(),
        voter_id: "4242".to_owned(),
        vote: PollVote {
            name: "Synthetic Voter".to_owned(),
            option_ids: vec![0]
        },
    })?);
    let recent = polls.recent_polls(chat_id)?;
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].0.question, "Encontrás una billetera. ¿Qué hacés?");
    assert_eq!(recent[0].1[0].option_ids, vec![0]);
    assert!(store.cancel_task(&id, &chat_id.to_string())?);
    Ok(())
}

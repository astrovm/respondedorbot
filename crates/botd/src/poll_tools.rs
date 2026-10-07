//! Native AI tools to send Telegram polls and read their votes.

use bot_adapters::redis_poll_store::RedisPollStore;
use bot_adapters::telegram_http::{
    ReqwestTelegramTransport, TelegramTransport, TransportFailureKind,
};
use bot_adapters::telegram_polls::{SendPollError, SentPoll, send_poll_with};
use bot_core::locale::Locale;
use bot_core::polls::{PollAnswer, PollRecord, PollRequest, PollState, PollVote, render_polls};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::dispatcher::PollUpdateSink;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub trait PollSender {
    fn send_poll(
        &mut self,
        chat_id: i64,
        reply_to_message_id: Option<i64>,
        poll: &PollRequest,
    ) -> Result<SentPoll, SendPollError>;
}

pub struct TelegramPollSender<Transport = ReqwestTelegramTransport> {
    transport: Transport,
    token: String,
}

impl TelegramPollSender {
    pub fn new(token: &str) -> Result<Self, TransportFailureKind> {
        Ok(Self::with_transport(
            ReqwestTelegramTransport::new()?,
            token,
        ))
    }
}

impl<Transport> TelegramPollSender<Transport> {
    #[must_use]
    pub fn with_transport(transport: Transport, token: &str) -> Self {
        Self {
            transport,
            token: token.to_owned(),
        }
    }
}

impl<Transport: TelegramTransport> PollSender for TelegramPollSender<Transport> {
    fn send_poll(
        &mut self,
        chat_id: i64,
        reply_to_message_id: Option<i64>,
        poll: &PollRequest,
    ) -> Result<SentPoll, SendPollError> {
        send_poll_with(
            &self.transport,
            &self.token,
            chat_id,
            reply_to_message_id,
            poll,
        )
    }
}

pub trait PollStore {
    fn save_poll(&mut self, record: &PollRecord) -> Result<(), String>;

    fn recent_polls(&mut self, chat_id: i64) -> Result<Vec<(PollRecord, Vec<PollVote>)>, String>;
}

impl PollStore for RedisPollStore {
    fn save_poll(&mut self, record: &PollRecord) -> Result<(), String> {
        Self::save_poll(self, record).map_err(crate::error_text)
    }

    fn recent_polls(&mut self, chat_id: i64) -> Result<Vec<(PollRecord, Vec<PollVote>)>, String> {
        Self::recent_polls(self, chat_id).map_err(crate::error_text)
    }
}

impl PollUpdateSink for RedisPollStore {
    fn record_answer(&mut self, answer: &PollAnswer) -> Result<bool, String> {
        Self::record_answer(self, answer).map_err(crate::error_text)
    }

    fn apply_state(&mut self, state: PollState) -> Result<bool, String> {
        Self::apply_state(self, state).map_err(crate::error_text)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PollToolContext {
    pub chat_id: i64,
    pub reply_to_message_id: Option<i64>,
    pub locale: Locale,
}

/// Handles both `create_poll` and `get_polls`.
pub struct PollTool<Sender, Store, Now> {
    sender: Sender,
    store: Store,
    now: Now,
    context: PollToolContext,
}

impl<Sender, Store, Now> PollTool<Sender, Store, Now> {
    #[must_use]
    pub const fn new(sender: Sender, store: Store, now: Now, context: PollToolContext) -> Self {
        Self {
            sender,
            store,
            now,
            context,
        }
    }
}

impl<Sender, Store, Now> PollTool<Sender, Store, Now>
where
    Sender: PollSender,
    Store: PollStore,
    Now: FnMut() -> i64,
{
    fn create(&mut self, poll: &PollRequest) -> ToolExecutionResult {
        let locale = self.context.locale;
        let sent = match self.sender.send_poll(
            self.context.chat_id,
            self.context.reply_to_message_id,
            poll,
        ) {
            Ok(sent) => sent,
            Err(SendPollError::Transport(kind)) => {
                return ToolExecutionResult::with_diagnostics(
                    match locale {
                        Locale::Es => {
                            "no sé si la encuesta llegó al chat; no la mandes de nuevo sin preguntar"
                        }
                        Locale::En => {
                            "I do not know if the poll reached the chat; do not send it again without asking"
                        }
                    },
                    vec![format!("sendPoll transport failed: {kind:?}")],
                );
            }
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    match locale {
                        Locale::Es => format!("Telegram no aceptó la encuesta: {error}"),
                        Locale::En => format!("Telegram did not accept the poll: {error}"),
                    },
                    vec![format!("sendPoll failed: {error}")],
                );
            }
        };
        let record = PollRecord {
            poll_id: sent.poll_id,
            chat_id: self.context.chat_id,
            message_id: sent.message_id,
            question: poll.question.clone(),
            options: poll.options.clone(),
            anonymous: poll.anonymous,
            multiple_answers: poll.multiple_answers,
            created_at: (self.now)(),
            closed: false,
            counts: None,
            total_voters: None,
        };
        let output = match locale {
            Locale::Es => format!(
                "Mandé la encuesta \"{}\" al chat. No repitas las opciones, ya se ven",
                poll.question
            ),
            Locale::En => format!(
                "Sent the poll \"{}\" to the chat. Do not repeat the options, people can see them",
                poll.question
            ),
        };
        let mut result = ToolExecutionResult::confirmed_output(output);
        if let Err(error) = self.store.save_poll(&record) {
            result
                .diagnostics
                .push(format!("poll sent but not stored: {error}"));
        }
        result
    }

    fn list(&mut self) -> ToolExecutionResult {
        let locale = self.context.locale;
        match self.store.recent_polls(self.context.chat_id) {
            Ok(polls) => ToolExecutionResult::output(render_polls(&polls, locale)),
            Err(error) => ToolExecutionResult::with_diagnostics(
                tool_output::failed(locale, "get_polls"),
                vec![format!("poll lookup failed: {error}")],
            ),
        }
    }
}

impl<Sender, Store, Now> ExternalToolExecutor for PollTool<Sender, Store, Now>
where
    Sender: PollSender,
    Store: PollStore,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        match request {
            ExternalToolRequest::CreatePoll(poll) => self.create(&poll),
            ExternalToolRequest::GetPolls => self.list(),
            _ => ToolExecutionResult::output(tool_output::incompatible(
                self.context.locale,
                "create_poll",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sender {
        reply: Result<SentPoll, SendPollError>,
        sent: Vec<(i64, Option<i64>, PollRequest)>,
    }

    impl PollSender for Sender {
        fn send_poll(
            &mut self,
            chat_id: i64,
            reply_to_message_id: Option<i64>,
            poll: &PollRequest,
        ) -> Result<SentPoll, SendPollError> {
            self.sent.push((chat_id, reply_to_message_id, poll.clone()));
            self.reply.clone()
        }
    }

    #[derive(Default)]
    struct Store {
        fail: bool,
        saved: Vec<PollRecord>,
    }

    impl PollStore for Store {
        fn save_poll(&mut self, record: &PollRecord) -> Result<(), String> {
            if self.fail {
                return Err("synthetic Redis failure".to_owned());
            }
            self.saved.push(record.clone());
            Ok(())
        }

        fn recent_polls(
            &mut self,
            _chat_id: i64,
        ) -> Result<Vec<(PollRecord, Vec<PollVote>)>, String> {
            if self.fail {
                return Err("synthetic Redis failure".to_owned());
            }
            Ok(self
                .saved
                .iter()
                .map(|record| {
                    (
                        record.clone(),
                        vec![PollVote {
                            name: "Ana".to_owned(),
                            option_ids: vec![0],
                        }],
                    )
                })
                .collect())
        }
    }

    fn poll() -> PollRequest {
        PollRequest {
            question: "¿Asado?".to_owned(),
            options: vec!["Sí".to_owned(), "No".to_owned()],
            anonymous: false,
            multiple_answers: false,
        }
    }

    fn poll_tool(
        reply: Result<SentPoll, SendPollError>,
        fail: bool,
        locale: Locale,
    ) -> PollTool<Sender, Store, fn() -> i64> {
        PollTool::new(
            Sender {
                reply,
                sent: Vec::new(),
            },
            Store {
                fail,
                saved: Vec::new(),
            },
            || 1_700_000_000,
            PollToolContext {
                chat_id: -100,
                reply_to_message_id: Some(7),
                locale,
            },
        )
    }

    fn sent() -> Result<SentPoll, SendPollError> {
        Ok(SentPoll {
            message_id: 9,
            poll_id: "p1".to_owned(),
        })
    }

    #[test]
    fn creates_stores_and_lists_polls_with_voters() {
        let mut tool = poll_tool(sent(), false, Locale::Es);
        let result = tool.execute(ExternalToolRequest::CreatePoll(poll()), "call");
        assert_eq!(
            result.output,
            "Mandé la encuesta \"¿Asado?\" al chat. No repitas las opciones, ya se ven"
        );
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some(result.output.as_str())
        );
        assert_eq!(tool.sender.sent, vec![(-100, Some(7), poll())]);
        assert_eq!(tool.store.saved[0].poll_id, "p1");
        assert_eq!(tool.store.saved[0].message_id, 9);
        assert_eq!(tool.store.saved[0].created_at, 1_700_000_000);
        assert_eq!(
            tool.execute(ExternalToolRequest::GetPolls, "call").output,
            "¿Asado?\n- Sí (1 voto): Ana\n- No (0 votos)\nTotal: 1 voto, abierta"
        );
        let mut english = poll_tool(sent(), false, Locale::En);
        assert!(
            english
                .execute(ExternalToolRequest::CreatePoll(poll()), "call")
                .output
                .starts_with("Sent the poll \"¿Asado?\"")
        );
    }

    #[test]
    fn storage_failures_keep_the_sent_poll_and_report_diagnostics() {
        let mut tool = poll_tool(sent(), true, Locale::En);
        let result = tool.execute(ExternalToolRequest::CreatePoll(poll()), "call");
        assert!(result.output.starts_with("Sent the poll"));
        assert!(result.diagnostics[0].contains("synthetic Redis failure"));
        let listed = tool.execute(ExternalToolRequest::GetPolls, "call");
        assert_eq!(listed.output, tool_output::failed(Locale::En, "get_polls"));
        assert!(listed.diagnostics[0].contains("synthetic Redis failure"));
    }

    #[test]
    fn telegram_failures_are_explained_without_resending() {
        for locale in [Locale::Es, Locale::En] {
            let mut timeout = poll_tool(
                Err(SendPollError::Transport(TransportFailureKind::Timeout)),
                false,
                locale,
            );
            let result = timeout.execute(ExternalToolRequest::CreatePoll(poll()), "call");
            assert!(result.output.contains("no"), "{}", result.output);
            assert!(result.failure_fallback.is_none());
            assert!(timeout.store.saved.is_empty());

            let mut rejected = poll_tool(
                Err(SendPollError::Rejected {
                    status_code: 400,
                    description: "Bad Request: synthetic".to_owned(),
                }),
                false,
                locale,
            );
            let result = rejected.execute(ExternalToolRequest::CreatePoll(poll()), "call");
            assert!(result.output.contains("Bad Request: synthetic"));
            assert!(rejected.store.saved.is_empty());
        }
    }

    #[test]
    fn other_requests_are_incompatible() {
        let mut tool = poll_tool(sent(), false, Locale::En);
        assert_eq!(
            tool.execute(ExternalToolRequest::TaskList, "call").output,
            "tool 'create_poll' received an incompatible request"
        );
    }

    struct Transport;

    impl TelegramTransport for Transport {
        fn send(
            &self,
            request: &bot_adapters::telegram_http::TelegramRequest,
        ) -> Result<bot_adapters::telegram_http::HttpResponse, TransportFailureKind> {
            assert_eq!(request.token, "synthetic-token");
            Ok(bot_adapters::telegram_http::HttpResponse {
                status_code: 200,
                body: r#"{"ok":true,"result":{"message_id":3,"poll":{"id":"p9"}}}"#.to_owned(),
            })
        }
    }

    #[test]
    fn telegram_sender_posts_through_its_transport() {
        assert!(TelegramPollSender::new("synthetic-token").is_ok());
        let mut sender = TelegramPollSender::with_transport(Transport, "synthetic-token");
        assert_eq!(
            sender.send_poll(-100, None, &poll()),
            Ok(SentPoll {
                message_id: 3,
                poll_id: "p9".to_owned(),
            })
        );
    }

    #[test]
    fn redis_store_backs_tools_and_poll_updates() -> crate::test_env::TestResult {
        crate::test_env::redis_endpoint().map_or(Ok(()), |endpoint| assert_redis_store(&endpoint))
    }

    fn assert_redis_store(
        endpoint: &bot_adapters::redis_connection::RedisEndpoint,
    ) -> crate::test_env::TestResult {
        let mut store = RedisPollStore::new(endpoint)?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let chat_id = -i64::try_from(nonce % 1_000_000_000_000)? - 3;
        let poll_id = format!("tool-poll-{nonce}");
        let record = PollRecord {
            poll_id: poll_id.clone(),
            chat_id,
            message_id: 1,
            question: "q".to_owned(),
            options: vec!["a".to_owned(), "b".to_owned()],
            anonymous: false,
            multiple_answers: false,
            created_at: 1,
            closed: false,
            counts: None,
            total_voters: None,
        };
        PollStore::save_poll(&mut store, &record)?;
        let answer = PollAnswer {
            poll_id: poll_id.clone(),
            voter_id: "1".to_owned(),
            vote: PollVote {
                name: "Ana".to_owned(),
                option_ids: vec![1],
            },
        };
        let state = PollState {
            poll_id,
            options: record.options.clone(),
            counts: vec![0, 1],
            total_voters: 1,
            closed: true,
        };
        let recorded = PollUpdateSink::record_answer(&mut store, &answer)?;
        let applied = PollUpdateSink::apply_state(&mut store, state)?;
        assert!(recorded && applied);
        let polls = PollStore::recent_polls(&mut store, chat_id)?;
        assert!(polls[0].0.closed);
        assert_eq!(polls[0].1[0].name, "Ana");
        Ok(())
    }
}

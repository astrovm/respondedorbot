//! Concrete adapter composition for the native Telegram runtime.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bot_adapters::chat_config::{ChatConfigRepository, ChatConfigRepositoryError};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_message_state::{RedisMessageState, RedisMessageStateError};
use bot_adapters::telegram_actions::{ActionError, ActionOutcome, execute_with};
use bot_adapters::telegram_http::{
    ReqwestTelegramTransport, TelegramTransport, TransportFailureKind,
};
use bot_adapters::telegram_polling::{PollOutcome, PollingError, poll_once_with};
use bot_core::chat_config::ChatConfig;
use bot_core::command_state::{
    BOT_MESSAGE_METADATA_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT, CHAT_STATE_TTL_SECONDS,
    IncomingCommandWritePlan, OutgoingCommandWritePlan,
};
use bot_core::telegram_actions::TelegramAction;
use bot_core::telegram_commands::command_publication_actions;
use num_bigint::{BigInt, BigUint};
use thiserror::Error;

use crate::dispatcher::{
    ActionReceipt, ActionSink, ChatConfigSource, MessageStateSink, NativeDispatcher, RandomSource,
    RuntimeValues,
};
use crate::runtime::{PollingRuntime, UpdateSource};

impl ChatConfigSource for ChatConfigRepository {
    type Error = ChatConfigRepositoryError;

    fn get(&mut self, chat_id: &str) -> Result<ChatConfig, Self::Error> {
        ChatConfigRepository::get(self, chat_id).map(|config| config.unwrap_or_default())
    }
}

pub struct TelegramUpdateSource<Transport> {
    transport: Transport,
    token: String,
    long_poll_seconds: u64,
}

impl<Transport> TelegramUpdateSource<Transport> {
    #[must_use]
    pub fn new(transport: Transport, token: &str, long_poll_timeout: Duration) -> Self {
        Self {
            transport,
            token: token.to_owned(),
            long_poll_seconds: long_poll_timeout.as_secs(),
        }
    }
}

impl<Transport: TelegramTransport> UpdateSource for TelegramUpdateSource<Transport> {
    fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError> {
        poll_once_with(&self.transport, &self.token, offset, self.long_poll_seconds)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TelegramActionSinkError {
    #[error(transparent)]
    Adapter(#[from] ActionError),
    #[error("Telegram action was rate limited")]
    RateLimited { retry_after_seconds: Option<u64> },
    #[error("Telegram action failed with status {status_code:?}: {description}")]
    Failed {
        status_code: Option<u16>,
        description: String,
    },
    #[error("Telegram action transport failed: {0:?}")]
    Transport(TransportFailureKind),
}

pub struct TelegramActionSink<Transport> {
    transport: Transport,
    token: String,
}

pub struct SystemRuntimeValues {
    instance_name: Option<String>,
}

impl SystemRuntimeValues {
    #[must_use]
    pub const fn new(instance_name: Option<String>) -> Self {
        Self { instance_name }
    }
}

impl RuntimeValues for SystemRuntimeValues {
    fn unix_timestamp(&mut self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64)
    }

    fn instance_name(&self) -> Option<&str> {
        self.instance_name.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SystemRandomError {
    #[error("random range must contain at least one value")]
    EmptyRange,
    #[error("random range is too large for this platform")]
    RangeTooLarge,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemRandomSource;

fn random_biguint_below(upper_exclusive: &BigUint) -> Result<BigUint, SystemRandomError> {
    if upper_exclusive == &BigUint::from(0_u8) {
        return Err(SystemRandomError::EmptyRange);
    }
    let bit_count = upper_exclusive.bits();
    let byte_count =
        usize::try_from(bit_count.div_ceil(8)).map_err(|_| SystemRandomError::RangeTooLarge)?;
    let retained_bits = bit_count % 8;
    loop {
        let mut bytes = vec![0_u8; byte_count];
        rand::fill(&mut bytes);
        if retained_bits != 0 {
            let mask = (1_u8 << retained_bits) - 1;
            if let Some(last) = bytes.last_mut() {
                *last &= mask;
            }
        }
        let candidate = BigUint::from_bytes_le(&bytes);
        if &candidate < upper_exclusive {
            return Ok(candidate);
        }
    }
}

impl RandomSource for SystemRandomSource {
    type Error = SystemRandomError;

    fn choice_index(&mut self, upper_exclusive: usize) -> Result<usize, Self::Error> {
        if upper_exclusive == 0 {
            return Err(SystemRandomError::EmptyRange);
        }
        Ok(rand::random_range(0..upper_exclusive))
    }

    fn inclusive_integer(&mut self, start: &BigInt, end: &BigInt) -> Result<BigInt, Self::Error> {
        let width = end - start + BigInt::from(1_u8);
        let Some(upper_exclusive) = width.to_biguint() else {
            return Err(SystemRandomError::EmptyRange);
        };
        let offset = random_biguint_below(&upper_exclusive)?;
        Ok(start + BigInt::from(offset))
    }
}

pub struct RedisCommandState {
    state: RedisMessageState,
}

impl RedisCommandState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisMessageStateError> {
        RedisMessageState::new(endpoint).map(|state| Self { state })
    }
}

impl MessageStateSink for RedisCommandState {
    type Error = RedisMessageStateError;

    fn record_incoming(&mut self, plan: &IncomingCommandWritePlan) -> Result<(), Self::Error> {
        let _stored = self.state.save_message(
            &plan.message,
            CHAT_STATE_TTL_SECONDS,
            CHAT_HISTORY_WRITE_LIMIT,
        )?;
        if let Some(member) = &plan.member {
            self.state.save_chat_member(
                &member.key,
                &member.user_id,
                &member.payload,
                CHAT_STATE_TTL_SECONDS,
            )?;
        }
        Ok(())
    }

    fn record_outgoing(&mut self, plan: &OutgoingCommandWritePlan) -> Result<(), Self::Error> {
        let _stored = self.state.save_message(
            &plan.message,
            CHAT_STATE_TTL_SECONDS,
            CHAT_HISTORY_WRITE_LIMIT,
        )?;
        if let Some(metadata) = &plan.metadata {
            self.state.set_value(
                &metadata.key,
                &metadata.payload,
                BOT_MESSAGE_METADATA_TTL_SECONDS,
            )?;
        }
        Ok(())
    }
}

impl<Transport> TelegramActionSink<Transport> {
    #[must_use]
    pub fn new(transport: Transport, token: &str) -> Self {
        Self {
            transport,
            token: token.to_owned(),
        }
    }
}

impl<Transport: TelegramTransport> ActionSink for TelegramActionSink<Transport> {
    type Error = TelegramActionSinkError;

    fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
        match execute_with(&self.transport, &self.token, action)? {
            ActionOutcome::Completed { message_id } => Ok(ActionReceipt {
                message_id: message_id.map(bot_core::telegram_input::MessageId),
            }),
            ActionOutcome::RateLimited {
                retry_after_seconds,
            } => Err(TelegramActionSinkError::RateLimited {
                retry_after_seconds,
            }),
            ActionOutcome::Failed {
                status_code,
                description,
            } => Err(TelegramActionSinkError::Failed {
                status_code,
                description,
            }),
            ActionOutcome::TransportFailed(failure) => {
                Err(TelegramActionSinkError::Transport(failure))
            }
        }
    }
}

pub fn publish_telegram_commands<Actions: ActionSink>(
    actions: &mut Actions,
) -> Result<(), Actions::Error> {
    for action in command_publication_actions() {
        let _receipt = actions.execute(action)?;
    }
    Ok(())
}

pub type ConcreteNativeRuntime = PollingRuntime<
    TelegramUpdateSource<ReqwestTelegramTransport>,
    NativeDispatcher<
        ChatConfigRepository,
        TelegramActionSink<ReqwestTelegramTransport>,
        RedisCommandState,
        SystemRuntimeValues,
        SystemRandomSource,
    >,
>;

#[derive(Debug, Error)]
pub enum CompositionError {
    #[error("could not construct Telegram polling transport: {0:?}")]
    PollingTransport(TransportFailureKind),
    #[error("could not construct Telegram action transport: {0:?}")]
    ActionTransport(TransportFailureKind),
    #[error("could not construct Redis command state: {0}")]
    RedisState(#[from] RedisMessageStateError),
}

pub fn build_native_runtime(
    token: &str,
    database_url: &str,
    bot_name: &str,
    instance_name: Option<String>,
    redis_endpoint: &RedisEndpoint,
    long_poll_timeout: Duration,
) -> Result<ConcreteNativeRuntime, CompositionError> {
    let polling_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::PollingTransport)?;
    let action_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::ActionTransport)?;
    let source = TelegramUpdateSource::new(polling_transport, token, long_poll_timeout);
    let config = ChatConfigRepository::new(database_url);
    let actions = TelegramActionSink::new(action_transport, token);
    let state = RedisCommandState::new(redis_endpoint)?;
    Ok(PollingRuntime::new(
        source,
        NativeDispatcher::new(
            config,
            actions,
            state,
            SystemRuntimeValues::new(instance_name),
            SystemRandomSource,
            bot_name,
        ),
    ))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_adapters::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };
    use bot_adapters::telegram_polling::{PollFailure, PollOutcome};
    use bot_core::telegram_actions::{SendMessage, TelegramAction};
    use bot_core::telegram_input::ChatId;
    use bot_core::{
        command_state::{
            IncomingCommandState, OutgoingCommandState, prepare_incoming_command_state,
            prepare_outgoing_command_state,
        },
        telegram_input::{MessageId, UserId},
    };
    use num_bigint::BigInt;

    use crate::dispatcher::{
        ActionReceipt, ActionSink, MessageStateSink, RandomSource, RuntimeValues,
    };
    use crate::runtime::UpdateSource;

    use super::{
        RedisCommandState, SystemRandomError, SystemRandomSource, SystemRuntimeValues,
        TelegramActionSink, TelegramActionSinkError, TelegramUpdateSource, build_native_runtime,
        publish_telegram_commands,
    };

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    impl TelegramTransport for Transport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(status_code: u16, body: &str) -> Transport {
        Transport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code,
                body: body.to_owned(),
            }))),
            requests: RefCell::new(Vec::new()),
        }
    }

    fn read_command(reader: &mut BufReader<TcpStream>) -> std::io::Result<Vec<String>> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let count = line
            .trim_end()
            .strip_prefix('*')
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            line.clear();
            reader.read_line(&mut line)?;
            let length = line
                .trim_end()
                .strip_prefix('$')
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let mut bytes = vec![0_u8; length + 2];
            std::io::Read::read_exact(reader, &mut bytes)?;
            values.push(String::from_utf8_lossy(&bytes[..length]).into_owned());
        }
        Ok(values)
    }

    #[test]
    fn polling_source_uses_configured_timeout_token_and_offset() {
        let transport = transport(200, r#"{"ok":true,"result":[]}"#);
        let mut source =
            TelegramUpdateSource::new(transport, "synthetic-token", Duration::from_secs(17));
        assert_eq!(source.poll(Some(42)), Ok(PollOutcome::Updates(Vec::new())));
        let requests = source.transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].token, "synthetic-token");
        assert_eq!(requests[0].timeout, Duration::from_secs(22));
        assert_eq!(
            requests[0]
                .params
                .as_ref()
                .and_then(|params| params.get("offset")),
            Some(&serde_json::json!(42))
        );
    }

    #[test]
    fn polling_transport_failure_is_retryable_not_an_invalid_update() {
        let transport = Transport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        let mut source = TelegramUpdateSource::new(transport, "token", Duration::from_secs(30));
        assert_eq!(
            source.poll(None),
            Ok(PollOutcome::Retry(PollFailure::Transport {
                failure: TransportFailureKind::Timeout
            }))
        );
    }

    #[test]
    fn action_sink_accepts_only_confirmed_delivery() {
        let mut sink = TelegramActionSink::new(
            transport(200, r#"{"ok":true,"result":{"message_id":9}}"#),
            "token",
        );
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Ok(ActionReceipt {
                message_id: Some(bot_core::telegram_input::MessageId(9))
            })
        );

        let mut sink = TelegramActionSink::new(
            transport(
                429,
                r#"{"ok":false,"error_code":429,"parameters":{"retry_after":4}}"#,
            ),
            "token",
        );
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::RateLimited {
                retry_after_seconds: Some(4)
            })
        );
    }

    #[test]
    fn action_sink_preserves_api_and_transport_failures() {
        let mut sink = TelegramActionSink::new(
            transport(400, r#"{"ok":false,"description":"synthetic rejection"}"#),
            "token",
        );
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::Failed {
                status_code: Some(400),
                description: "synthetic rejection".to_owned()
            })
        );

        let transport = Transport {
            response: RefCell::new(Some(Err(TransportFailureKind::Connection))),
            requests: RefCell::new(Vec::new()),
        };
        let mut sink = TelegramActionSink::new(transport, "token");
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::Transport(
                TransportFailureKind::Connection
            ))
        );
    }

    #[test]
    fn command_publication_executes_default_spanish_and_english_in_order() {
        #[derive(Default)]
        struct Published(Vec<TelegramAction>);

        impl ActionSink for Published {
            type Error = std::convert::Infallible;

            fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                self.0.push(action);
                Ok(ActionReceipt { message_id: None })
            }
        }

        let mut published = Published::default();
        assert_eq!(publish_telegram_commands(&mut published), Ok(()));
        let languages = published
            .0
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SetCommands { language_code, .. } => language_code.as_deref(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(languages, ["es", "en"]);
        assert_eq!(published.0.len(), 3);
    }

    #[test]
    fn system_runtime_values_preserve_instance_and_current_epoch_seconds() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs());
        let mut values = SystemRuntimeValues::new(Some("synthetic".to_owned()));
        let actual = values.unix_timestamp();
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs());
        assert!(actual >= before as i64);
        assert!(actual <= after as i64);
        assert_eq!(values.instance_name(), Some("synthetic"));
    }

    #[test]
    fn system_random_source_samples_choices_and_arbitrary_precision_ranges() {
        let mut random = SystemRandomSource;
        for _ in 0..32 {
            let index = random.choice_index(3);
            assert!(index.is_ok_and(|value| value < 3));
        }
        assert_eq!(random.choice_index(0), Err(SystemRandomError::EmptyRange));

        let start = BigInt::parse_bytes(b"100000000000000000000", 10);
        let end = BigInt::parse_bytes(b"100000000000000000002", 10);
        let (Some(start), Some(end)) = (start, end) else {
            return;
        };
        for _ in 0..32 {
            let sampled = random.inclusive_integer(&start, &end);
            assert!(sampled.is_ok_and(|value| value >= start && value <= end));
        }
        assert_eq!(
            random.inclusive_integer(&BigInt::from(2_u8), &BigInt::from(1_u8)),
            Err(SystemRandomError::EmptyRange)
        );
    }

    #[test]
    fn redis_command_state_writes_history_member_and_metadata_contracts()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(
            move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let (index, _) = listener.accept()?;
                let mut reader = BufReader::new(index);
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("FT.CREATE")
                );
                reader.get_mut().write_all(b"-Index already exists\r\n")?;

                let (incoming, _) = listener.accept()?;
                let mut reader = BufReader::new(incoming);
                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("EVAL"));
                assert!(command.iter().any(|value| value == "chat_history:-42"));
                reader.get_mut().write_all(b":1\r\n")?;

                let (member, _) = listener.accept()?;
                let mut reader = BufReader::new(member);
                assert_eq!(read_command(&mut reader)?, ["MULTI"]);
                reader.get_mut().write_all(b"+OK\r\n")?;
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("HSET")
                );
                reader.get_mut().write_all(b"+QUEUED\r\n")?;
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("EXPIRE")
                );
                reader.get_mut().write_all(b"+QUEUED\r\n")?;
                assert_eq!(read_command(&mut reader)?, ["EXEC"]);
                reader.get_mut().write_all(b"*2\r\n:1\r\n:1\r\n")?;

                let (outgoing, _) = listener.accept()?;
                let mut reader = BufReader::new(outgoing);
                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("EVAL"));
                assert!(command.iter().any(|value| value == "bot_99"));
                reader.get_mut().write_all(b":1\r\n")?;

                let (metadata, _) = listener.accept()?;
                let mut reader = BufReader::new(metadata);
                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("SETEX"));
                assert_eq!(
                    command.get(1).map(String::as_str),
                    Some("bot_message_meta:-42:99")
                );
                reader.get_mut().write_all(b"+OK\r\n")?;
                Ok(())
            },
        );

        let mut state = RedisCommandState::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        let incoming = prepare_incoming_command_state(IncomingCommandState {
            chat_id: ChatId(-42),
            message_id: MessageId(7),
            user_id: UserId(88),
            first_name: Some("Synthetic"),
            username: Some("tester"),
            text: "/time",
            is_group: true,
            timestamp: 1_672_531_200,
        })?;
        state.record_incoming(&incoming)?;
        let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id: ChatId(-42),
            incoming_message_id: MessageId(7),
            sent_message_id: Some(MessageId(99)),
            text: "1672531200",
            command: "/time",
            timestamp: 1_672_531_200,
        })?;
        state.record_outgoing(&outgoing)?;

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn concrete_runtime_composes_without_contacting_external_services() {
        let result = build_native_runtime(
            "synthetic-token",
            "postgresql://synthetic.invalid/database",
            "@synthetic_bot",
            Some("synthetic-instance".to_owned()),
            &RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port: 1,
                password: None,
            },
            Duration::from_secs(30),
        );
        assert!(result.is_ok());
    }
}

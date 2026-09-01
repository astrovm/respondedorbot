//! Redacted operational failure delivery.

use std::collections::HashSet;

use bot_adapters::telegram_actions::{ActionOutcome, execute_with};
use bot_adapters::telegram_http::TelegramTransport;
use bot_core::locale::Locale;
use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::ChatId;

pub trait OperationalReporter: Send + Sync + 'static {
    fn report(&self, message: &str) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct NoopOperationalReporter;

impl OperationalReporter for NoopOperationalReporter {
    fn report(&self, _message: &str) -> Result<(), String> {
        Ok(())
    }
}

pub struct TelegramOperationalReporter<Transport> {
    transport: Transport,
    token: String,
    chat_id: ChatId,
    instance_name: String,
    secrets: Vec<String>,
    locale: Locale,
}

impl<Transport> TelegramOperationalReporter<Transport> {
    #[must_use]
    pub fn new(
        transport: Transport,
        token: &str,
        chat_id: i64,
        instance_name: Option<&str>,
        secrets: impl IntoIterator<Item = String>,
        locale: Locale,
    ) -> Self {
        let mut seen = HashSet::new();
        let mut secrets = secrets
            .into_iter()
            .filter(|secret| !secret.is_empty() && seen.insert(secret.clone()))
            .collect::<Vec<_>>();
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        Self {
            transport,
            token: token.to_owned(),
            chat_id: ChatId(chat_id),
            instance_name: instance_name.unwrap_or("unknown").to_owned(),
            secrets,
            locale,
        }
    }

    fn redact(&self, message: &str) -> String {
        self.secrets
            .iter()
            .fold(message.to_owned(), |text, secret| {
                text.replace(secret, "[REDACTED]")
            })
    }
}

impl<Transport> OperationalReporter for TelegramOperationalReporter<Transport>
where
    Transport: TelegramTransport + Send + Sync + 'static,
{
    fn report(&self, message: &str) -> Result<(), String> {
        let text = match self.locale {
            Locale::Es => format!(
                "informe administrativo de {}: {}",
                self.instance_name,
                self.redact(message)
            ),
            Locale::En => format!(
                "admin report from {}: {}",
                self.instance_name,
                self.redact(message)
            ),
        };
        let action = TelegramAction::SendMessage(SendMessage::new(self.chat_id, &text));
        match execute_with(&self.transport, &self.token, action)
            .map_err(|error| error.to_string())?
        {
            ActionOutcome::Completed { .. } => Ok(()),
            ActionOutcome::RateLimited {
                retry_after_seconds,
            } => Err(match self.locale {
                Locale::Es => format!(
                    "el informe administrativo fue limitado (retry_after={retry_after_seconds:?})"
                ),
                Locale::En => {
                    format!("admin report was rate limited (retry_after={retry_after_seconds:?})")
                }
            }),
            ActionOutcome::Failed {
                status_code,
                description,
            } => Err(match self.locale {
                Locale::Es => format!(
                    "falló el informe administrativo con estado {status_code:?}: {description}"
                ),
                Locale::En => {
                    format!("admin report failed with status {status_code:?}: {description}")
                }
            }),
            ActionOutcome::TransportFailed(failure) => Err(match self.locale {
                Locale::Es => {
                    format!("falló el transporte del informe administrativo: {failure:?}")
                }
                Locale::En => format!("admin report transport failed: {failure:?}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use bot_adapters::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };
    use bot_core::locale::Locale;
    use serde_json::Value;

    use super::{OperationalReporter, TelegramOperationalReporter};

    #[derive(Default)]
    struct Transport {
        requests: Mutex<Vec<TelegramRequest>>,
        response: Mutex<Option<Result<HttpResponse, TransportFailureKind>>>,
    }

    impl TelegramTransport for Transport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests
                .lock()
                .map_err(|_| TransportFailureKind::Request)?
                .push(request.clone());
            self.response
                .lock()
                .map_err(|_| TransportFailureKind::Request)?
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(body: &str) -> Transport {
        Transport {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Some(Ok(HttpResponse {
                status_code: 200,
                body: body.to_owned(),
            }))),
        }
    }

    #[test]
    fn sends_redacted_plain_text_to_the_configured_admin() {
        let reporter = TelegramOperationalReporter::new(
            transport(r#"{"ok":true,"result":{"message_id":9}}"#),
            "telegram-secret",
            -42,
            Some("test-instance"),
            ["database-secret".to_owned(), "provider-secret".to_owned()],
            Locale::Es,
        );
        assert!(
            reporter
                .report("worker failed at database-secret using provider-secret")
                .is_ok()
        );
        let requests = reporter.transport.requests.lock();
        assert!(requests.is_ok());
        let Some(request) = requests.ok().and_then(|requests| requests.first().cloned()) else {
            return;
        };
        assert_eq!(request.endpoint, "sendMessage");
        let text = request
            .json_payload
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|payload| payload.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert_eq!(
            text,
            "informe administrativo de test-instance: worker failed at [REDACTED] using [REDACTED]"
        );
        assert!(!text.contains("database-secret"));
        assert!(!text.contains("provider-secret"));
    }

    #[test]
    fn reports_telegram_delivery_failures_without_panicking() {
        let reporter = TelegramOperationalReporter::new(
            transport(r#"{"ok":false,"error_code":403,"description":"forbidden"}"#),
            "token",
            42,
            None,
            [],
            Locale::En,
        );
        let error = reporter.report("failure");
        assert!(error.is_err());
        assert!(error.err().is_some_and(|error| error.contains("forbidden")));
    }
}

//! Redacted operational failure delivery.

use std::collections::HashSet;

use bot_adapters::telegram_actions::{ActionOutcome, execute_with};
use bot_adapters::telegram_http::TelegramTransport;
use bot_core::locale::Locale;
use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::ChatId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalReport {
    spanish: String,
    english: String,
}

impl OperationalReport {
    #[must_use]
    pub fn new(spanish: impl Into<String>, english: impl Into<String>) -> Self {
        Self {
            spanish: spanish.into(),
            english: english.into(),
        }
    }

    #[must_use]
    pub const fn for_locale(&self, locale: Locale) -> &str {
        match locale {
            Locale::Es => self.spanish.as_str(),
            Locale::En => self.english.as_str(),
        }
    }

    #[must_use]
    pub const fn english(&self) -> &str {
        self.english.as_str()
    }
}

pub trait OperationalReporter: Send + Sync + 'static {
    fn report(&self, report: &OperationalReport) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct NoopOperationalReporter;

impl OperationalReporter for NoopOperationalReporter {
    fn report(&self, _report: &OperationalReport) -> Result<(), String> {
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
    fn report(&self, report: &OperationalReport) -> Result<(), String> {
        let message = self.redact(report.for_locale(self.locale));
        let text = match self.locale {
            Locale::Es => format!(
                "informe administrativo de {}: {}",
                self.instance_name, message
            ),
            Locale::En => format!("admin report from {}: {}", self.instance_name, message),
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

    use super::{OperationalReport, OperationalReporter, TelegramOperationalReporter};

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
                .report(&OperationalReport::new(
                    "el proceso falló en database-secret usando provider-secret",
                    "worker failed at database-secret using provider-secret",
                ))
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
            "informe administrativo de test-instance: el proceso falló en [REDACTED] usando [REDACTED]"
        );
        assert!(!text.contains("database-secret"));
        assert!(!text.contains("provider-secret"));
    }

    #[test]
    fn sends_the_english_report_body_without_spanish_fragments() {
        let reporter = TelegramOperationalReporter::new(
            transport(r#"{"ok":true,"result":{"message_id":10}}"#),
            "token",
            42,
            Some("test-instance"),
            [],
            Locale::En,
        );
        assert!(
            reporter
                .report(&OperationalReport::new(
                    "falló el sondeo de Telegram",
                    "Telegram polling failed",
                ))
                .is_ok()
        );
        let requests = reporter.transport.requests.lock();
        let text = requests
            .as_deref()
            .ok()
            .and_then(|requests| requests.first())
            .and_then(|request| request.json_payload.as_ref())
            .and_then(|payload| payload.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert_eq!(
            text,
            "admin report from test-instance: Telegram polling failed"
        );
        assert!(!text.contains("falló"));
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
        let error = reporter.report(&OperationalReport::new("fallo", "failure"));
        assert!(error.is_err());
        assert!(error.err().is_some_and(|error| error.contains("forbidden")));
    }
}

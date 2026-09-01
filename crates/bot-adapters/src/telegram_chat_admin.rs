//! Telegram `getChatMember` adapter for group-configuration authorization.

use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

use crate::telegram_http::{
    TelegramHttpError, TelegramHttpOutcome, TelegramTransport, request_with,
};

const ADMIN_LOOKUP_TIMEOUT_SECONDS: u64 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatAdminLookup {
    pub is_admin: bool,
    pub diagnostic: Option<String>,
}

#[derive(Debug, Error)]
pub enum ChatAdminLookupError {
    #[error(transparent)]
    Http(#[from] TelegramHttpError),
}

#[derive(Deserialize)]
struct Envelope {
    ok: bool,
    #[serde(default)]
    result: Option<Member>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct Member {
    status: String,
}

pub fn lookup_chat_admin_with<Transport: TelegramTransport>(
    transport: &Transport,
    token: &str,
    chat_id: &str,
    user_id: &str,
) -> Result<ChatAdminLookup, ChatAdminLookupError> {
    let outcome = request_with(
        transport,
        token,
        "getChatMember",
        "GET",
        Some(json!({"chat_id":chat_id,"user_id":user_id})),
        None,
        ADMIN_LOOKUP_TIMEOUT_SECONDS,
    )?;
    let TelegramHttpOutcome::Response { status_code, body } = outcome else {
        return Ok(ChatAdminLookup {
            is_admin: false,
            diagnostic: Some("Telegram chat-admin transport failed".to_owned()),
        });
    };
    let envelope = serde_json::from_str::<Envelope>(&body);
    let Ok(envelope) = envelope else {
        return Ok(ChatAdminLookup {
            is_admin: false,
            diagnostic: Some(format!(
                "Telegram chat-admin response was malformed (status {status_code})"
            )),
        });
    };
    if !envelope.ok {
        return Ok(ChatAdminLookup {
            is_admin: false,
            diagnostic: Some(
                envelope.description.unwrap_or_else(|| {
                    format!("Telegram chat-admin lookup failed ({status_code})")
                }),
            ),
        });
    }
    let status = envelope.result.map(|member| member.status.to_lowercase());
    Ok(ChatAdminLookup {
        is_admin: status
            .as_deref()
            .is_some_and(|status| matches!(status, "administrator" | "creator")),
        diagnostic: None,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{ChatAdminLookup, lookup_chat_admin_with};
    use crate::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
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

    fn lookup(body: &str) -> (ChatAdminLookup, Transport) {
        let transport = Transport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code: 200,
                body: body.to_owned(),
            }))),
            requests: RefCell::new(Vec::new()),
        };
        let result = lookup_chat_admin_with(&transport, "token", "-42", "7");
        assert!(result.is_ok());
        (
            result.unwrap_or(ChatAdminLookup {
                is_admin: false,
                diagnostic: None,
            }),
            transport,
        )
    }

    #[test]
    fn recognizes_creator_and_administrator_statuses() {
        for status in ["creator", "administrator"] {
            let (result, transport) = lookup(&format!(
                r#"{{"ok":true,"result":{{"status":"{status}"}}}}"#
            ));
            assert_eq!(
                result,
                ChatAdminLookup {
                    is_admin: true,
                    diagnostic: None
                }
            );
            let requests = transport.requests.borrow();
            assert_eq!(requests[0].endpoint, "getChatMember");
            assert_eq!(
                requests[0].params,
                Some(serde_json::json!({"chat_id":"-42","user_id":"7"}))
            );
        }
    }

    #[test]
    fn members_failures_and_malformed_responses_are_not_authorized() {
        let (member, _) = lookup(r#"{"ok":true,"result":{"status":"member"}}"#);
        assert_eq!(
            member,
            ChatAdminLookup {
                is_admin: false,
                diagnostic: None
            }
        );

        let (failed, _) = lookup(r#"{"ok":false,"description":"synthetic denial"}"#);
        assert_eq!(failed.diagnostic.as_deref(), Some("synthetic denial"));

        let (malformed, _) = lookup("not-json");
        assert!(!malformed.is_admin);
        assert!(malformed.diagnostic.is_some());

        let transport = Transport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        let result = lookup_chat_admin_with(&transport, "token", "-42", "7");
        assert!(result.is_ok_and(|lookup| !lookup.is_admin && lookup.diagnostic.is_some()));
    }
}

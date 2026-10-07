//! Polls the bot sends: validation, Telegram update parsing, and rendering.
//!
//! Telegram only reports votes and voters to the bot that sent the poll, so
//! these records exist only for polls created through the AI tool.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::locale::Locale;
use crate::telegram_input::{format_user_identity, normalize_numeric_id};

pub const MAX_POLL_QUESTION_CHARS: usize = 300;
pub const MAX_POLL_OPTION_CHARS: usize = 100;
pub const MIN_POLL_OPTIONS: usize = 2;
pub const MAX_POLL_OPTIONS: usize = 12;
/// Polls and votes are kept for a month, matching Telegram's longest
/// automatic close period.
pub const POLL_TTL_SECONDS: i64 = 30 * 24 * 60 * 60;
/// Only the newest polls per chat stay listed for the AI.
pub const MAX_POLLS_PER_CHAT: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PollRequest {
    pub question: String,
    pub options: Vec<String>,
    pub anonymous: bool,
    pub multiple_answers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollRequestError {
    MissingQuestion,
    QuestionTooLong,
    TooFewOptions,
    TooManyOptions,
    OptionTooLong,
    DuplicateOptions,
}

impl PollRequestError {
    #[must_use]
    pub const fn message(self, locale: Locale) -> &'static str {
        match (self, locale) {
            (Self::MissingQuestion, Locale::Es) => "falta la pregunta de la encuesta",
            (Self::MissingQuestion, Locale::En) => "the poll question is missing",
            (Self::QuestionTooLong, Locale::Es) => "la pregunta pasa los 300 caracteres",
            (Self::QuestionTooLong, Locale::En) => "the question is over 300 characters",
            (Self::TooFewOptions, Locale::Es) => "una encuesta necesita al menos 2 opciones",
            (Self::TooFewOptions, Locale::En) => "a poll needs at least 2 options",
            (Self::TooManyOptions, Locale::Es) => "Telegram acepta hasta 12 opciones",
            (Self::TooManyOptions, Locale::En) => "Telegram accepts up to 12 options",
            (Self::OptionTooLong, Locale::Es) => "cada opción puede tener hasta 100 caracteres",
            (Self::OptionTooLong, Locale::En) => "each option can be up to 100 characters",
            (Self::DuplicateOptions, Locale::Es) => "hay opciones repetidas",
            (Self::DuplicateOptions, Locale::En) => "some options are repeated",
        }
    }
}

impl PollRequest {
    /// Trims the text and checks Telegram's `sendPoll` limits up front, so the
    /// model gets a clear reason instead of a raw API error.
    pub fn new(
        question: &str,
        options: &[String],
        anonymous: bool,
        multiple_answers: bool,
    ) -> Result<Self, PollRequestError> {
        let question = question.trim();
        if question.is_empty() {
            return Err(PollRequestError::MissingQuestion);
        }
        if question.chars().count() > MAX_POLL_QUESTION_CHARS {
            return Err(PollRequestError::QuestionTooLong);
        }
        let options = options
            .iter()
            .map(|option| option.trim().to_owned())
            .filter(|option| !option.is_empty())
            .collect::<Vec<_>>();
        if options.len() < MIN_POLL_OPTIONS {
            return Err(PollRequestError::TooFewOptions);
        }
        if options.len() > MAX_POLL_OPTIONS {
            return Err(PollRequestError::TooManyOptions);
        }
        if options
            .iter()
            .any(|option| option.chars().count() > MAX_POLL_OPTION_CHARS)
        {
            return Err(PollRequestError::OptionTooLong);
        }
        let mut unique = options
            .iter()
            .map(|option| option.to_lowercase())
            .collect::<Vec<_>>();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != options.len() {
            return Err(PollRequestError::DuplicateOptions);
        }
        Ok(Self {
            question: question.to_owned(),
            options,
            anonymous,
            multiple_answers,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PollRecord {
    pub poll_id: String,
    pub chat_id: i64,
    pub message_id: i64,
    pub question: String,
    pub options: Vec<String>,
    pub anonymous: bool,
    pub multiple_answers: bool,
    pub created_at: i64,
    #[serde(default)]
    pub closed: bool,
    /// Per-option counts from Telegram `poll` updates. Anonymous polls only
    /// have these; public polls fall back to counting stored votes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counts: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_voters: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PollVote {
    pub name: String,
    pub option_ids: Vec<usize>,
}

/// One `poll_answer` update. Empty `option_ids` means the vote was retracted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PollAnswer {
    pub poll_id: String,
    pub voter_id: String,
    pub vote: PollVote,
}

/// Fresh state from a `poll` update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PollState {
    pub poll_id: String,
    pub options: Vec<String>,
    pub counts: Vec<u64>,
    pub total_voters: u64,
    pub closed: bool,
}

#[must_use]
pub fn poll_key(poll_id: &str) -> String {
    format!("poll:{poll_id}")
}

#[must_use]
pub fn poll_votes_key(poll_id: &str) -> String {
    format!("poll_votes:{poll_id}")
}

#[must_use]
pub fn chat_polls_key(chat_id: i64) -> String {
    format!("chat_polls:{chat_id}")
}

#[must_use]
pub fn parse_poll_answer(payload: &Map<String, Value>) -> Option<PollAnswer> {
    let poll_id = payload.get("poll_id")?.as_str()?.to_owned();
    let (voter_id, name) = if let Some(user) = payload.get("user").filter(|user| user.is_object()) {
        let id = user.get("id").and_then(normalize_numeric_id)?;
        (id.to_string(), format_user_identity(user))
    } else {
        let chat = payload.get("voter_chat")?;
        let id = chat.get("id").and_then(normalize_numeric_id)?;
        let title = chat
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        (format!("chat:{id}"), title)
    };
    let option_ids = payload
        .get("option_ids")?
        .as_array()?
        .iter()
        .filter_map(Value::as_u64)
        .filter_map(|id| usize::try_from(id).ok())
        .collect();
    Some(PollAnswer {
        poll_id,
        voter_id,
        vote: PollVote { name, option_ids },
    })
}

#[must_use]
pub fn parse_poll_state(payload: &Map<String, Value>) -> Option<PollState> {
    let poll_id = payload.get("id")?.as_str()?.to_owned();
    let raw_options = payload.get("options")?.as_array()?;
    let mut options = Vec::with_capacity(raw_options.len());
    let mut counts = Vec::with_capacity(raw_options.len());
    for option in raw_options {
        options.push(option.get("text")?.as_str()?.to_owned());
        counts.push(
            option
                .get("voter_count")
                .and_then(Value::as_u64)
                .unwrap_or_default(),
        );
    }
    Some(PollState {
        poll_id,
        options,
        counts,
        total_voters: payload
            .get("total_voter_count")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        closed: payload
            .get("is_closed")
            .and_then(Value::as_bool)
            .unwrap_or_default(),
    })
}

impl PollRecord {
    /// Applies a `poll` update. Users can add options after creation, so the
    /// option list is replaced too.
    pub fn apply_state(&mut self, state: PollState) {
        self.options = state.options;
        self.counts = Some(state.counts);
        self.total_voters = Some(state.total_voters);
        self.closed = state.closed;
    }
}

fn votes_label(count: u64, locale: Locale) -> String {
    match (count, locale) {
        (1, Locale::Es) => "1 voto".to_owned(),
        (1, Locale::En) => "1 vote".to_owned(),
        (count, Locale::Es) => format!("{count} votos"),
        (count, Locale::En) => format!("{count} votes"),
    }
}

fn option_label(options: &[String], index: usize, locale: Locale) -> String {
    options.get(index).cloned().unwrap_or_else(|| match locale {
        Locale::Es => format!("opción {}", index + 1),
        Locale::En => format!("option {}", index + 1),
    })
}

fn render_poll(record: &PollRecord, votes: &[PollVote], locale: Locale) -> String {
    let option_count = record
        .options
        .len()
        .max(record.counts.as_ref().map_or(0, Vec::len));
    let counts = record.counts.clone().unwrap_or_else(|| {
        let mut counts = vec![0_u64; option_count];
        for vote in votes {
            for option in &vote.option_ids {
                if let Some(count) = counts.get_mut(*option) {
                    *count += 1;
                }
            }
        }
        counts
    });
    let total = record.total_voters.unwrap_or_else(|| {
        votes
            .iter()
            .filter(|vote| !vote.option_ids.is_empty())
            .count() as u64
    });
    let mut lines = vec![record.question.clone()];
    for index in 0..option_count {
        let count = counts.get(index).copied().unwrap_or_default();
        let mut line = format!(
            "- {} ({})",
            option_label(&record.options, index, locale),
            votes_label(count, locale)
        );
        if !record.anonymous {
            let mut names = votes
                .iter()
                .filter(|vote| vote.option_ids.contains(&index))
                .map(|vote| vote.name.as_str())
                .collect::<Vec<_>>();
            names.sort_unstable();
            if !names.is_empty() {
                line.push_str(": ");
                line.push_str(&names.join(", "));
            }
        }
        lines.push(line);
    }
    let mut status = vec![format!("Total: {}", votes_label(total, locale))];
    status.push(
        match (record.closed, locale) {
            (true, Locale::Es) => "cerrada",
            (true, Locale::En) => "closed",
            (false, Locale::Es) => "abierta",
            (false, Locale::En) => "open",
        }
        .to_owned(),
    );
    if record.anonymous {
        status.push(
            match locale {
                Locale::Es => "anónima, nadie ve quién votó",
                Locale::En => "anonymous, nobody can see who voted",
            }
            .to_owned(),
        );
    }
    if record.multiple_answers {
        status.push(
            match locale {
                Locale::Es => "varias respuestas",
                Locale::En => "multiple answers",
            }
            .to_owned(),
        );
    }
    lines.push(status.join(", "));
    lines.join("\n")
}

/// Newest first. Each entry is a poll the bot sent and its stored votes.
#[must_use]
pub fn render_polls(polls: &[(PollRecord, Vec<PollVote>)], locale: Locale) -> String {
    if polls.is_empty() {
        return match locale {
            Locale::Es => "No mandé encuestas en este chat todavía",
            Locale::En => "I have not sent any polls in this chat yet",
        }
        .to_owned();
    }
    polls
        .iter()
        .map(|(record, votes)| render_poll(record, votes, locale))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn options(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn record(anonymous: bool) -> PollRecord {
        PollRecord {
            poll_id: "p1".to_owned(),
            chat_id: -100,
            message_id: 5,
            question: "¿Qué hacés?".to_owned(),
            options: options(&["Laburando", "Durmiendo"]),
            anonymous,
            multiple_answers: false,
            created_at: 1_700_000_000,
            closed: false,
            counts: None,
            total_voters: None,
        }
    }

    #[test]
    fn requests_follow_telegram_limits() {
        assert_eq!(
            PollRequest::new("  ¿Pizza?  ", &options(&[" Sí ", "", "No"]), false, true),
            Ok(PollRequest {
                question: "¿Pizza?".to_owned(),
                options: options(&["Sí", "No"]),
                anonymous: false,
                multiple_answers: true,
            })
        );
        let cases = [
            (" ", options(&["a", "b"]), PollRequestError::MissingQuestion),
            (
                &*"q".repeat(301),
                options(&["a", "b"]),
                PollRequestError::QuestionTooLong,
            ),
            ("q", options(&["a", " "]), PollRequestError::TooFewOptions),
            (
                "q",
                (0..13).map(|index| index.to_string()).collect(),
                PollRequestError::TooManyOptions,
            ),
            (
                "q",
                vec!["a".to_owned(), "🦀".repeat(101)],
                PollRequestError::OptionTooLong,
            ),
            (
                "q",
                options(&["Sí", "sí"]),
                PollRequestError::DuplicateOptions,
            ),
        ];
        for (question, options, error) in cases {
            assert_eq!(
                PollRequest::new(question, &options, false, false),
                Err(error)
            );
        }
        assert!(PollRequest::new(&"🦀".repeat(300), &options(&["a", "b"]), true, false).is_ok());
        assert!(
            PollRequest::new(
                "q",
                &(0..12).map(|index| index.to_string()).collect::<Vec<_>>(),
                true,
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn errors_are_localized() {
        for error in [
            PollRequestError::MissingQuestion,
            PollRequestError::QuestionTooLong,
            PollRequestError::TooFewOptions,
            PollRequestError::TooManyOptions,
            PollRequestError::OptionTooLong,
            PollRequestError::DuplicateOptions,
        ] {
            assert_ne!(error.message(Locale::Es), error.message(Locale::En));
        }
    }

    #[test]
    fn keys_are_scoped_by_poll_and_chat() {
        assert_eq!(poll_key("abc"), "poll:abc");
        assert_eq!(poll_votes_key("abc"), "poll_votes:abc");
        assert_eq!(chat_polls_key(-100), "chat_polls:-100");
    }

    #[test]
    fn parses_user_chat_and_retracted_answers() {
        let user = json!({
            "poll_id": "p1",
            "user": {"id": 7, "first_name": "Ana", "username": "ana"},
            "option_ids": [1, -1, "x"]
        });
        assert_eq!(
            parse_poll_answer(user.as_object().unwrap_or(&Map::new())),
            Some(PollAnswer {
                poll_id: "p1".to_owned(),
                voter_id: "7".to_owned(),
                vote: PollVote {
                    name: "Ana (ana)".to_owned(),
                    option_ids: vec![1],
                },
            })
        );
        let chat = json!({
            "poll_id": "p1",
            "voter_chat": {"id": -5, "title": "Canal"},
            "option_ids": []
        });
        assert_eq!(
            parse_poll_answer(chat.as_object().unwrap_or(&Map::new())),
            Some(PollAnswer {
                poll_id: "p1".to_owned(),
                voter_id: "chat:-5".to_owned(),
                vote: PollVote {
                    name: "Canal".to_owned(),
                    option_ids: Vec::new(),
                },
            })
        );
        let untitled = json!({"poll_id": "p1", "voter_chat": {"id": -5}, "option_ids": []});
        assert_eq!(
            parse_poll_answer(untitled.as_object().unwrap_or(&Map::new()))
                .map(|answer| answer.vote.name),
            Some(String::new())
        );
        for malformed in [
            json!({"user": {"id": 7}, "option_ids": []}),
            json!({"poll_id": "p1", "option_ids": []}),
            json!({"poll_id": "p1", "user": {"first_name": "Ana"}, "option_ids": []}),
            json!({"poll_id": "p1", "voter_chat": {"title": "x"}, "option_ids": []}),
            json!({"poll_id": "p1", "user": {"id": 7}}),
        ] {
            assert_eq!(
                parse_poll_answer(malformed.as_object().unwrap_or(&Map::new())),
                None
            );
        }
    }

    #[test]
    fn parses_poll_state_and_rejects_malformed_updates() {
        let update = json!({
            "id": "p1",
            "options": [{"text": "Laburando", "voter_count": 2}, {"text": "Durmiendo"}],
            "total_voter_count": 2,
            "is_closed": true
        });
        let state = parse_poll_state(update.as_object().unwrap_or(&Map::new()));
        assert_eq!(
            state,
            Some(PollState {
                poll_id: "p1".to_owned(),
                options: options(&["Laburando", "Durmiendo"]),
                counts: vec![2, 0],
                total_voters: 2,
                closed: true,
            })
        );
        let bare = json!({"id": "p1", "options": []});
        assert_eq!(
            parse_poll_state(bare.as_object().unwrap_or(&Map::new())).map(|state| state.closed),
            Some(false)
        );
        for malformed in [
            json!({"options": []}),
            json!({"id": "p1"}),
            json!({"id": "p1", "options": [{"voter_count": 1}]}),
        ] {
            assert_eq!(
                parse_poll_state(malformed.as_object().unwrap_or(&Map::new())),
                None
            );
        }
    }

    #[test]
    fn renders_voters_for_public_polls() {
        let votes = vec![
            PollVote {
                name: "Beto".to_owned(),
                option_ids: vec![0],
            },
            PollVote {
                name: "Ana (ana)".to_owned(),
                option_ids: vec![0, 7],
            },
            PollVote {
                name: "Retracted".to_owned(),
                option_ids: Vec::new(),
            },
        ];
        assert_eq!(
            render_polls(&[(record(false), votes)], Locale::Es),
            "¿Qué hacés?\n- Laburando (2 votos): Ana (ana), Beto\n- Durmiendo (0 votos)\n\
             Total: 2 votos, abierta"
        );
    }

    #[test]
    fn renders_telegram_counts_for_anonymous_closed_polls() {
        let mut anonymous = record(true);
        anonymous.multiple_answers = true;
        anonymous.apply_state(PollState {
            poll_id: "p1".to_owned(),
            options: options(&["Laburando", "Durmiendo", "Agregada"]),
            counts: vec![1, 0],
            total_voters: 1,
            closed: true,
        });
        assert_eq!(
            render_polls(&[(anonymous, Vec::new())], Locale::En),
            "¿Qué hacés?\n- Laburando (1 vote)\n- Durmiendo (0 votes)\n- Agregada (0 votes)\n\
             Total: 1 vote, closed, anonymous, nobody can see who voted, multiple answers"
        );
        let mut extra = record(false);
        extra.counts = Some(vec![0, 0, 1]);
        extra.total_voters = Some(1);
        let rendered = render_polls(
            &[
                (
                    extra,
                    vec![PollVote {
                        name: "Ana".to_owned(),
                        option_ids: vec![2],
                    }],
                ),
                (record(true), Vec::new()),
            ],
            Locale::Es,
        );
        assert!(rendered.contains("- opción 3 (1 voto): Ana"), "{rendered}");
        assert!(
            rendered.ends_with("anónima, nadie ve quién votó"),
            "{rendered}"
        );
        let mut english = record(false);
        english.options.clear();
        english.counts = Some(vec![0]);
        english.multiple_answers = true;
        assert_eq!(
            render_polls(&[(english, Vec::new())], Locale::En),
            "¿Qué hacés?\n- option 1 (0 votes)\nTotal: 0 votes, open, multiple answers"
        );
        let mut spanish = record(false);
        spanish.closed = true;
        spanish.multiple_answers = true;
        assert!(
            render_polls(&[(spanish, Vec::new())], Locale::Es)
                .ends_with("cerrada, varias respuestas")
        );
    }

    #[test]
    fn empty_lists_say_no_polls_were_sent() {
        assert_eq!(
            render_polls(&[], Locale::Es),
            "No mandé encuestas en este chat todavía"
        );
        assert_eq!(
            render_polls(&[], Locale::En),
            "I have not sent any polls in this chat yet"
        );
    }

    #[test]
    fn records_round_trip_without_optional_state() {
        let encoded = serde_json::to_value(record(false)).unwrap_or_default();
        assert!(encoded.get("counts").is_none());
        let decoded: Result<PollRecord, _> = serde_json::from_value(json!({
            "poll_id": "p1", "chat_id": 1, "message_id": 2, "question": "q",
            "options": ["a", "b"], "anonymous": false, "multiple_answers": false,
            "created_at": 3
        }));
        assert_eq!(decoded.ok().map(|record| record.closed), Some(false));
    }
}

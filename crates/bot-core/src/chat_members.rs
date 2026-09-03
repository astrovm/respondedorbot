//! Decoding and deterministic AI rendering for known chat members.

use serde::Deserialize;

use crate::locale::Locale;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownChatMember {
    pub user_id: String,
    pub first_name: String,
    pub username: String,
    pub last_seen: i64,
}

#[derive(Debug, Deserialize)]
struct StoredMemberPayload {
    schema_version: u8,
    #[serde(default)]
    first_name: String,
    #[serde(default)]
    username: String,
    #[serde(default)]
    last_seen: i64,
}

#[must_use]
pub fn decode_chat_members(entries: &[(String, String)]) -> Vec<KnownChatMember> {
    entries
        .iter()
        .filter(|(user_id, _)| !user_id.is_empty())
        .filter_map(|(user_id, payload)| {
            let parsed = serde_json::from_str::<StoredMemberPayload>(payload).ok()?;
            (parsed.schema_version == 1).then(|| KnownChatMember {
                user_id: user_id.clone(),
                first_name: parsed.first_name,
                username: parsed.username,
                last_seen: parsed.last_seen,
            })
        })
        .collect()
}

#[must_use]
pub fn render_chat_members(members: &[KnownChatMember], now_unix: i64, locale: Locale) -> String {
    if members.is_empty() {
        return match locale {
            Locale::Es => "no conozco a nadie en este chat todavía".to_owned(),
            Locale::En => "I do not know anyone in this chat yet".to_owned(),
        };
    }
    let lines = members
        .iter()
        .map(|member| {
            let age = now_unix.saturating_sub(member.last_seen);
            let ago = if age < 60 {
                match locale {
                    Locale::Es => "hace unos segundos".to_owned(),
                    Locale::En => "a few seconds ago".to_owned(),
                }
            } else if age < 3_600 {
                match locale {
                    Locale::Es => format!("hace {} min", age / 60),
                    Locale::En => format!("{} min ago", age / 60),
                }
            } else if age < 86_400 {
                match locale {
                    Locale::Es => format!("hace {} h", age / 3_600),
                    Locale::En => format!("{} h ago", age / 3_600),
                }
            } else {
                match locale {
                    Locale::Es => format!("hace {} d", age / 86_400),
                    Locale::En => format!("{} d ago", age / 86_400),
                }
            };
            let name = if member.username.is_empty() {
                member.first_name.clone()
            } else {
                format!("{} (@{})", member.first_name, member.username)
            };
            match locale {
                Locale::Es => format!("- {name} — visto {ago}"),
                Locale::En => format!("- {name} — seen {ago}"),
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    match locale {
        Locale::Es => format!("Miembros conocidos:\n{lines}"),
        Locale::En => format!("Known members:\n{lines}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_only_current_member_payloads() {
        assert_eq!(
            decode_chat_members(&[
                (
                    "7".to_owned(),
                    r#"{"schema_version":1,"first_name":"Ana","username":"ana","last_seen":100}"#
                        .to_owned(),
                ),
                ("8".to_owned(), "invalid".to_owned()),
                (String::new(), "{}".to_owned()),
            ]),
            [KnownChatMember {
                user_id: "7".to_owned(),
                first_name: "Ana".to_owned(),
                username: "ana".to_owned(),
                last_seen: 100,
            },]
        );
    }

    #[test]
    fn renders_all_relative_time_bands_and_empty_state() {
        let members = [
            KnownChatMember {
                user_id: "1".to_owned(),
                first_name: "A".to_owned(),
                username: "a".to_owned(),
                last_seen: 9_970,
            },
            KnownChatMember {
                user_id: "2".to_owned(),
                first_name: "B".to_owned(),
                username: String::new(),
                last_seen: 9_400,
            },
            KnownChatMember {
                user_id: "3".to_owned(),
                first_name: "C".to_owned(),
                username: String::new(),
                last_seen: 2_800,
            },
            KnownChatMember {
                user_id: "4".to_owned(),
                first_name: "D".to_owned(),
                username: String::new(),
                last_seen: -76_400,
            },
        ];
        assert_eq!(
            render_chat_members(&members, 10_000, Locale::Es),
            "Miembros conocidos:\n- A (@a) — visto hace unos segundos\n- B — visto hace 10 min\n- C — visto hace 2 h\n- D — visto hace 1 d"
        );
        assert_eq!(
            render_chat_members(&[], 10_000, Locale::Es),
            "no conozco a nadie en este chat todavía"
        );
        assert_eq!(
            render_chat_members(&members, 10_000, Locale::En),
            "Known members:\n- A (@a) — seen a few seconds ago\n- B — seen 10 min ago\n- C — seen 2 h ago\n- D — seen 1 d ago"
        );
    }
}

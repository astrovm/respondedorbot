//! Decoding and deterministic AI rendering for known chat members.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownChatMember {
    pub user_id: String,
    pub first_name: String,
    pub username: String,
    pub last_seen: i64,
}

#[derive(Debug, Default, Deserialize)]
struct StoredMemberPayload {
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
        .map(|(user_id, payload)| {
            let parsed = serde_json::from_str::<StoredMemberPayload>(payload).unwrap_or_default();
            KnownChatMember {
                user_id: user_id.clone(),
                first_name: parsed.first_name,
                username: parsed.username,
                last_seen: parsed.last_seen,
            }
        })
        .collect()
}

#[must_use]
pub fn render_chat_members(members: &[KnownChatMember], now_unix: i64) -> String {
    if members.is_empty() {
        return "no conozco a nadie en este chat todavia".to_owned();
    }
    let lines = members
        .iter()
        .map(|member| {
            let age = now_unix.saturating_sub(member.last_seen);
            let ago = if age < 60 {
                "hace unos segundos".to_owned()
            } else if age < 3_600 {
                format!("hace {} min", age / 60)
            } else if age < 86_400 {
                format!("hace {}h", age / 3_600)
            } else {
                format!("hace {}d", age / 86_400)
            };
            let name = if member.username.is_empty() {
                member.first_name.clone()
            } else {
                format!("{} (@{})", member.first_name, member.username)
            };
            format!("- {name} — visto {ago}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("Miembros conocidos:\n{lines}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_compatible_payloads_and_defaults_malformed_json() {
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
            [
                KnownChatMember {
                    user_id: "7".to_owned(),
                    first_name: "Ana".to_owned(),
                    username: "ana".to_owned(),
                    last_seen: 100,
                },
                KnownChatMember {
                    user_id: "8".to_owned(),
                    first_name: String::new(),
                    username: String::new(),
                    last_seen: 0,
                },
            ]
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
            render_chat_members(&members, 10_000),
            "Miembros conocidos:\n- A (@a) — visto hace unos segundos\n- B — visto hace 10 min\n- C — visto hace 2h\n- D — visto hace 1d"
        );
        assert_eq!(
            render_chat_members(&[], 10_000),
            "no conozco a nadie en este chat todavia"
        );
    }
}

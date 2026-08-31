//! Side-effect-free planning for incremental Telegram response edits.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAction {
    None,
    Send,
    Edit,
}

impl StreamAction {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Send => "send",
            Self::Edit => "edit",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedPlan {
    pub buffer: String,
    pub action: StreamAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizePlan {
    pub text: String,
    pub action: StreamAction,
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn should_edit(
    done: bool,
    has_message_id: bool,
    now_seconds: f64,
    last_edit_seconds: f64,
    buffer_chars: usize,
    sent_chars: usize,
    min_edit_interval_seconds: f64,
    min_chars_between_edits: usize,
) -> bool {
    if done || !has_message_id {
        return false;
    }
    now_seconds - last_edit_seconds >= min_edit_interval_seconds
        && buffer_chars.saturating_sub(sent_chars) >= min_chars_between_edits
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn plan_feed(
    done: bool,
    has_message_id: bool,
    send_attempted: bool,
    buffer: &str,
    sent_text: &str,
    token: &str,
    now_seconds: f64,
    last_edit_seconds: f64,
    min_edit_interval_seconds: f64,
    min_chars_between_edits: usize,
) -> FeedPlan {
    if done {
        return FeedPlan {
            buffer: buffer.to_owned(),
            action: StreamAction::None,
        };
    }
    let buffer = format!("{buffer}{token}");
    let action = if !has_message_id && !send_attempted && !buffer.trim().is_empty() {
        StreamAction::Send
    } else if should_edit(
        false,
        has_message_id,
        now_seconds,
        last_edit_seconds,
        buffer.chars().count(),
        sent_text.chars().count(),
        min_edit_interval_seconds,
        min_chars_between_edits,
    ) {
        StreamAction::Edit
    } else {
        StreamAction::None
    };
    FeedPlan { buffer, action }
}

#[must_use]
pub fn plan_finalize(
    buffer: &str,
    sent_text: &str,
    has_message_id: bool,
    final_text: Option<&str>,
) -> FinalizePlan {
    let text = final_text.unwrap_or(buffer).to_owned();
    let action = if !has_message_id {
        StreamAction::Send
    } else if text != sent_text {
        StreamAction::Edit
    } else {
        StreamAction::None
    };
    FinalizePlan { text, action }
}

#[cfg(test)]
mod tests {
    use super::{FeedPlan, FinalizePlan, StreamAction, plan_feed, plan_finalize, should_edit};

    #[test]
    fn edit_threshold_requires_message_time_and_new_characters() {
        assert!(should_edit(false, true, 2.0, 1.0, 20, 5, 1.0, 15));
        assert!(!should_edit(true, true, 2.0, 1.0, 20, 5, 1.0, 15));
        assert!(!should_edit(false, false, 2.0, 1.0, 20, 5, 1.0, 15));
        assert!(!should_edit(false, true, 1.9, 1.0, 20, 5, 1.0, 15));
        assert!(!should_edit(false, true, 2.0, 1.0, 19, 5, 1.0, 15));
    }

    #[test]
    fn feed_plans_initial_send_periodic_edit_and_noop() {
        assert_eq!(
            plan_feed(false, false, false, "", "", "hi", 0.0, 0.0, 0.3, 15),
            FeedPlan {
                buffer: "hi".to_owned(),
                action: StreamAction::Send,
            }
        );
        assert_eq!(
            plan_feed(false, false, false, "", "", " ", 0.0, 0.0, 0.3, 15),
            FeedPlan {
                buffer: " ".to_owned(),
                action: StreamAction::None,
            }
        );
        assert_eq!(
            plan_feed(false, false, false, " ", "", "hi", 0.0, 0.0, 0.3, 15),
            FeedPlan {
                buffer: " hi".to_owned(),
                action: StreamAction::Send,
            }
        );
        assert_eq!(
            plan_feed(
                false, true, true, "hello", "hello", " there", 1.1, 1.0, 0.3, 5
            ),
            FeedPlan {
                buffer: "hello there".to_owned(),
                action: StreamAction::None,
            }
        );
        assert_eq!(
            plan_feed(
                false, true, true, "hello", "hello", " there", 1.3, 1.0, 0.3, 5
            ),
            FeedPlan {
                buffer: "hello there".to_owned(),
                action: StreamAction::Edit,
            }
        );
        assert_eq!(
            plan_feed(true, false, false, "done", "", " ignored", 0.0, 0.0, 0.0, 0,),
            FeedPlan {
                buffer: "done".to_owned(),
                action: StreamAction::None,
            }
        );
    }

    #[test]
    fn finalize_plans_send_edit_and_noop() {
        assert_eq!(
            plan_finalize("buffer", "", false, None),
            FinalizePlan {
                text: "buffer".to_owned(),
                action: StreamAction::Send,
            }
        );
        assert_eq!(
            plan_finalize("buffer", "old", true, Some("final")),
            FinalizePlan {
                text: "final".to_owned(),
                action: StreamAction::Edit,
            }
        );
        assert_eq!(
            plan_finalize("same", "same", true, None),
            FinalizePlan {
                text: "same".to_owned(),
                action: StreamAction::None,
            }
        );
    }
}

use bot_core::chat_config::ChatConfig;
use bot_core::config_callbacks::{ConfigCallbackOutcome, plan_config_callback};
use bot_core::config_command::{plan_config_command, render_config};
use bot_core::locale::Locale;
use bot_core::telegram_actions::TelegramAction;
use bot_core::telegram_input::{ChatId, MessageId};

#[test]
fn public_config_renderer_covers_bilingual_private_and_group_states() {
    let cases = [
        (
            ChatConfig::default(),
            Locale::Es,
            true,
            "config del gordo",
            7,
        ),
        (
            ChatConfig {
                language: "es".to_owned(),
                link_mode: "delete".to_owned(),
                ai_command_followups: false,
                ignore_link_fix_followups: false,
                timezone_offset: 0,
                ai_random_replies: false,
                creditless_user_hourly_limit: 0,
            },
            Locale::Es,
            false,
            "solo disponible en grupos",
            5,
        ),
        (
            ChatConfig {
                language: "en".to_owned(),
                link_mode: "off".to_owned(),
                timezone_offset: i64::MAX,
                ai_random_replies: false,
                creditless_user_hourly_limit: -1,
                ..ChatConfig::default()
            },
            Locale::En,
            true,
            "Do not modify links",
            7,
        ),
        (
            ChatConfig {
                timezone_offset: i64::MIN,
                link_mode: "delete".to_owned(),
                ..ChatConfig::default()
            },
            Locale::En,
            false,
            "Delete the original and repost the fixed link",
            5,
        ),
    ];
    for (config, locale, is_group, expected, row_count) in cases {
        let (text, keyboard) = render_config(&config, locale, is_group);
        assert!(text.contains(expected));
        assert_eq!(keyboard.inline_keyboard.len(), row_count);
    }

    assert!(matches!(
        plan_config_command(
            ChatId(1),
            MessageId(2),
            "/settings",
            "@mybot",
            Locale::En,
            &ChatConfig::default(),
            false,
        ),
        Some(TelegramAction::SendMessage(_))
    ));
    assert!(
        plan_config_command(
            ChatId(1),
            MessageId(2),
            "/other",
            "@mybot",
            Locale::Es,
            &ChatConfig::default(),
            true,
        )
        .is_none()
    );
}

#[test]
fn public_config_callback_planner_covers_every_transition_family() {
    let current = ChatConfig::default();
    for data in [
        "cfg:language:en",
        "cfg:link:delete",
        "cfg:random:toggle",
        "cfg:followups:toggle",
        "cfg:linkfixfollowups:toggle",
        "cfg:timezone:2",
        "cfg:timezone:999999999999999999999999999999999",
        "cfg:timezone:invalid",
        "cfg:creditless:none",
        "cfg:creditless:decrease",
        "cfg:creditless:increase",
        "cfg:creditless:unlimited",
        "cfg:creditless:-2",
        "cfg:unknown:value",
    ] {
        let (outcome, _updated) = plan_config_callback(data, &current);
        assert!(matches!(outcome, ConfigCallbackOutcome::Render { .. }));
    }

    for data in [
        "cfg:timezone:current",
        "cfg:creditless:current",
        "cfg:broken",
    ] {
        let (outcome, _updated) = plan_config_callback(data, &current);
        assert!(matches!(outcome, ConfigCallbackOutcome::Guard));
    }
    let (outcome, _updated) = plan_config_callback("other", &current);
    assert!(matches!(outcome, ConfigCallbackOutcome::NotHandled));

    let maximum = ChatConfig {
        creditless_user_hourly_limit: i64::MAX,
        ..current
    };
    let (outcome, _updated) = plan_config_callback("cfg:creditless:increase", &maximum);
    assert!(matches!(outcome, ConfigCallbackOutcome::LegacyRequired));
}

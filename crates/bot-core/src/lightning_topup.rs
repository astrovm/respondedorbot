//! Lightning top-up menus, pricing, and invoice messages.
//!
//! Lightning packs give the same credits as the Stars packs. They cost what
//! Telegram pays out per Star plus OpenNode's fee, so a pack nets the same
//! either way and buyers skip Telegram's cut.

use crate::credit_units::{CreditUnits, display_credit_units};
use crate::locale::Locale;
use crate::menu_ui::{back, button, localized};
use crate::provider_pricing::STAR_PAYOUT_USD_MICROS;
use crate::telegram_actions::{
    CopyTextButton, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, SendMessage,
};
use crate::telegram_input::ChatId;
use crate::telegram_payments::{
    BillingPackTerms, billing_packs, default_billing_pack, whole_credits,
};

pub const LIGHTNING_MENU_CALLBACK: &str = "topup:ln";
pub const STARS_MENU_CALLBACK: &str = "topup:stars";
const LIGHTNING_PACK_PREFIX: &str = "topup:ln:";
/// How long a Lightning invoice can be paid.
pub const LIGHTNING_INVOICE_TTL_MINUTES: u32 = 30;
/// OpenNode's cut of each payment, in hundredths of a percent.
const OPENNODE_FEE_BASIS_POINTS: i128 = 100;
/// Telegram copy buttons hold at most this many characters.
const MAX_COPY_TEXT_LENGTH: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LightningCallback {
    /// Show the Lightning packs.
    Menu,
    /// Go back to the Stars packs.
    Stars,
    /// Create an invoice for one pack.
    Pack(BillingPackTerms),
    /// A Lightning pack id that does not exist.
    InvalidPack,
}

/// Price of a pack paid with Lightning, rounded up to whole cents, so what
/// is left after OpenNode's fee matches the Stars payout.
#[must_use]
pub fn lightning_usd_cents(pack: &BillingPackTerms) -> i64 {
    let payout = i128::from(pack.xtr_amount) * STAR_PAYOUT_USD_MICROS;
    // Micro-dollars to cents is / 10_000, and dividing by the share OpenNode
    // leaves us is * 10_000 / (10_000 - fee), so the two cancel.
    let kept = 10_000 - OPENNODE_FEE_BASIS_POINTS;
    i64::try_from((payout + kept - 1) / kept).unwrap_or(i64::MAX)
}

#[must_use]
pub fn format_usd(cents: i64) -> String {
    format!("US${}.{:02}", cents / 100, cents % 100)
}

#[must_use]
pub fn parse_lightning_callback(data: &str) -> Option<LightningCallback> {
    if data == LIGHTNING_MENU_CALLBACK {
        return Some(LightningCallback::Menu);
    }
    if data == STARS_MENU_CALLBACK {
        return Some(LightningCallback::Stars);
    }
    let pack_id = data.strip_prefix(LIGHTNING_PACK_PREFIX)?;
    Some(
        default_billing_pack(pack_id)
            .map_or(LightningCallback::InvalidPack, LightningCallback::Pack),
    )
}

#[must_use]
pub fn lightning_menu(locale: Locale) -> (String, InlineKeyboardMarkup) {
    let rows = billing_packs()
        .map(|pack| {
            let credits = whole_credits(pack.credits_awarded);
            let price = format_usd(lightning_usd_cents(&pack));
            vec![button(
                match locale {
                    Locale::Es => format!("{credits} créditos por {price} ⚡"),
                    Locale::En => format!("{credits} credits for {price} ⚡"),
                },
                format!("{LIGHTNING_PACK_PREFIX}{}", pack.id),
            )]
        })
        .chain(std::iter::once(vec![back(locale, STARS_MENU_CALLBACK)]))
        .collect();
    (
        localized(
            locale,
            "Cargar con Lightning ⚡\n\nSin la comisión de Telegram, así que te sale más barato. Elegí un pack.",
            "Pay with Lightning ⚡\n\nNo Telegram fee, so it's cheaper. Choose a pack.",
        )
        .to_owned(),
        InlineKeyboardMarkup {
            inline_keyboard: rows,
        },
    )
}

/// What the payment provider returned for a new charge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightningInvoice {
    pub charge_id: String,
    pub payreq: String,
    pub checkout_url: Option<String>,
    pub sats: Option<i64>,
}

/// Invoice text with the BOLT11 request in a code entity, which Telegram
/// copies on tap.
#[must_use]
pub fn lightning_invoice_message(
    chat_id: ChatId,
    pack: &BillingPackTerms,
    invoice: &LightningInvoice,
    locale: Locale,
) -> SendMessage {
    let credits = whole_credits(pack.credits_awarded);
    let price = format_usd(lightning_usd_cents(pack));
    let price = match invoice.sats {
        Some(sats) => format!("{price} ({sats} sats)"),
        None => price,
    };
    let minutes = LIGHTNING_INVOICE_TTL_MINUTES;
    let text = match locale {
        Locale::Es => format!(
            "Factura Lightning ⚡\n\n{credits} créditos por {price}\nVence en {minutes} minutos. Cuando la pagues, te acredito solo.\n\n<code>{}</code>",
            invoice.payreq
        ),
        Locale::En => format!(
            "Lightning invoice ⚡\n\n{credits} credits for {price}\nExpires in {minutes} minutes. I'll add the credits as soon as it's paid.\n\n<code>{}</code>",
            invoice.payreq
        ),
    };
    let mut rows = Vec::new();
    if invoice.payreq.len() <= MAX_COPY_TEXT_LENGTH {
        rows.push(vec![InlineKeyboardButton {
            text: localized(locale, "Copiar factura", "Copy invoice").to_owned(),
            url: None,
            callback_data: None,
            copy_text: Some(CopyTextButton {
                text: invoice.payreq.clone(),
            }),
        }]);
    }
    if let Some(url) = &invoice.checkout_url {
        rows.push(vec![InlineKeyboardButton {
            text: localized(locale, "Pagar en la web", "Pay on the web").to_owned(),
            url: Some(url.clone()),
            callback_data: None,
            copy_text: None,
        }]);
    }
    let mut message = SendMessage::new(chat_id, &text);
    message.parse_mode = Some(ParseMode::Html);
    message.disable_web_page_preview = true;
    message.reply_markup = (!rows.is_empty()).then_some(InlineKeyboardMarkup {
        inline_keyboard: rows,
    });
    message
}

#[must_use]
pub const fn lightning_invoice_failed(locale: Locale) -> &'static str {
    localized(
        locale,
        "No pude armar la factura Lightning. Probá de nuevo en un rato",
        "I couldn't create the Lightning invoice. Try again in a bit",
    )
}

#[must_use]
pub const fn lightning_invoice_ready(locale: Locale) -> &'static str {
    localized(locale, "Listo, te dejé la factura", "Invoice ready")
}

#[must_use]
pub fn lightning_paid_reply(credits_awarded: i64, user_balance: i64, locale: Locale) -> String {
    let credits = display_credit_units(CreditUnits::new(credits_awarded));
    let balance = display_credit_units(CreditUnits::new(user_balance));
    match locale {
        Locale::Es => format!(
            "Pago Lightning recibido ⚡\n+{credits} créditos\nSaldo personal: {balance} créditos"
        ),
        Locale::En => format!(
            "Lightning payment received ⚡\n+{credits} credits\nPersonal balance: {balance} credits"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LightningCallback, LightningInvoice, format_usd, lightning_invoice_failed,
        lightning_invoice_message, lightning_invoice_ready, lightning_menu, lightning_paid_reply,
        lightning_usd_cents, parse_lightning_callback,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::{CopyTextButton, ParseMode};
    use crate::telegram_input::ChatId;
    use crate::telegram_payments::{billing_packs, default_billing_pack};

    #[test]
    fn packs_cost_the_telegram_payout_plus_the_opennode_fee_rounded_up_to_cents() {
        let prices = billing_packs()
            .map(|pack| (pack.id.clone(), lightning_usd_cents(&pack)))
            .collect::<Vec<_>>();
        assert_eq!(
            prices,
            [
                ("p50".to_owned(), 33),
                ("p100".to_owned(), 66),
                ("p250".to_owned(), 165),
                ("p500".to_owned(), 329),
                ("p1000".to_owned(), 657),
                ("p2500".to_owned(), 1_642),
            ]
        );
        for pack in billing_packs() {
            // What OpenNode leaves us, in micro-dollars, covers the Stars payout.
            let kept = i128::from(lightning_usd_cents(&pack)) * 10_000 * 99 / 100;
            let payout =
                i128::from(pack.xtr_amount) * crate::provider_pricing::STAR_PAYOUT_USD_MICROS;
            assert!(kept >= payout, "{}", pack.id);
            assert!(kept - payout < 10_000, "{}", pack.id);
        }
        assert_eq!(format_usd(33), "US$0.33");
        assert_eq!(format_usd(1_625), "US$16.25");
        assert_eq!(format_usd(500), "US$5.00");
    }

    #[test]
    fn parses_menu_back_and_pack_callbacks() {
        assert_eq!(
            parse_lightning_callback("topup:ln"),
            Some(LightningCallback::Menu)
        );
        assert_eq!(
            parse_lightning_callback("topup:stars"),
            Some(LightningCallback::Stars)
        );
        assert_eq!(
            parse_lightning_callback("topup:ln:p100"),
            default_billing_pack("p100").map(LightningCallback::Pack)
        );
        assert_eq!(
            parse_lightning_callback("topup:ln:nope"),
            Some(LightningCallback::InvalidPack)
        );
        for data in ["topup:p50", "topup:lnp50", "ln:p50", ""] {
            assert_eq!(parse_lightning_callback(data), None, "{data}");
        }
    }

    #[test]
    fn menu_lists_every_pack_and_goes_back_to_stars() {
        let (text, keyboard) = lightning_menu(Locale::Es);
        assert!(text.starts_with("Cargar con Lightning ⚡"));
        let rows = &keyboard.inline_keyboard;
        assert_eq!(rows.len(), 7);
        assert_eq!(rows[0][0].text, "50 créditos por US$0.33 ⚡");
        assert_eq!(rows[0][0].callback_data.as_deref(), Some("topup:ln:p50"));
        assert_eq!(rows[6][0].text, "‹ Volver");
        assert_eq!(rows[6][0].callback_data.as_deref(), Some("topup:stars"));
        let (text, keyboard) = lightning_menu(Locale::En);
        assert_eq!(
            text,
            "Pay with Lightning ⚡\n\nNo Telegram fee, so it's cheaper. Choose a pack."
        );
        assert_eq!(
            keyboard.inline_keyboard[5][0].text,
            "2,500 credits for US$16.42 ⚡"
        );
    }

    #[test]
    fn stars_menu_offers_lightning_only_when_available() {
        use crate::telegram_payments::topup_menu;
        let (text, keyboard) = topup_menu(Locale::Es, true);
        assert!(text.ends_with("\n\nCon Lightning ⚡ te sale más barato."));
        let rows = &keyboard.inline_keyboard;
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[6][0].text, "⚡ Pagar con Lightning");
        assert_eq!(rows[6][0].callback_data.as_deref(), Some("topup:ln"));
        assert_eq!(rows[7][0].callback_data.as_deref(), Some("topup:close"));
        let (text, keyboard) = topup_menu(Locale::En, true);
        assert!(text.ends_with("\n\nLightning ⚡ is cheaper."));
        assert_eq!(keyboard.inline_keyboard[6][0].text, "⚡ Pay with Lightning");
        let (text, keyboard) = topup_menu(Locale::En, false);
        assert!(!text.contains("Lightning"));
        assert_eq!(keyboard.inline_keyboard.len(), 7);
    }

    fn invoice(payreq_length: usize, checkout: bool, sats: Option<i64>) -> LightningInvoice {
        LightningInvoice {
            charge_id: "charge-1".to_owned(),
            payreq: format!("lnbc{}", "x".repeat(payreq_length - 4)),
            checkout_url: checkout.then(|| "https://checkout.example.test/charge-1".to_owned()),
            sats,
        }
    }

    #[test]
    fn invoice_message_shows_price_code_and_buttons() {
        let pack = default_billing_pack("p50");
        assert!(pack.is_some());
        let Some(pack) = pack else { return };
        let short = invoice(20, true, Some(512));
        let message = lightning_invoice_message(ChatId(88), &pack, &short, Locale::Es);
        assert_eq!(message.chat_id, ChatId(88));
        assert_eq!(message.parse_mode, Some(ParseMode::Html));
        assert!(message.disable_web_page_preview);
        assert_eq!(
            message.text,
            format!(
                "Factura Lightning ⚡\n\n50 créditos por US$0.33 (512 sats)\nVence en 30 minutos. Cuando la pagues, te acredito solo.\n\n<code>{}</code>",
                short.payreq
            )
        );
        let rows = message
            .reply_markup
            .map(|markup| markup.inline_keyboard)
            .unwrap_or_default();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0].text, "Copiar factura");
        assert_eq!(
            rows[0][0].copy_text,
            Some(CopyTextButton {
                text: short.payreq.clone()
            })
        );
        assert_eq!(rows[1][0].text, "Pagar en la web");
        assert_eq!(
            rows[1][0].url.as_deref(),
            Some("https://checkout.example.test/charge-1")
        );

        // Telegram rejects copy buttons over 256 characters.
        let long = invoice(257, false, None);
        let message = lightning_invoice_message(ChatId(88), &pack, &long, Locale::En);
        assert!(
            message.text.starts_with(
                "Lightning invoice ⚡\n\n50 credits for US$0.33\nExpires in 30 minutes."
            )
        );
        assert_eq!(message.reply_markup, None);
        let message =
            lightning_invoice_message(ChatId(88), &pack, &invoice(256, false, None), Locale::En);
        let rows = message
            .reply_markup
            .map(|markup| markup.inline_keyboard)
            .unwrap_or_default();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0].text, "Copy invoice");
    }

    #[test]
    fn localized_status_replies() {
        assert_eq!(
            lightning_invoice_failed(Locale::Es),
            "No pude armar la factura Lightning. Probá de nuevo en un rato"
        );
        assert_eq!(
            lightning_invoice_failed(Locale::En),
            "I couldn't create the Lightning invoice. Try again in a bit"
        );
        assert_eq!(
            lightning_invoice_ready(Locale::Es),
            "Listo, te dejé la factura"
        );
        assert_eq!(lightning_invoice_ready(Locale::En), "Invoice ready");
        assert_eq!(
            lightning_paid_reply(5_000, 12_345, Locale::Es),
            "Pago Lightning recibido ⚡\n+50.00 créditos\nSaldo personal: 123.45 créditos"
        );
        assert_eq!(
            lightning_paid_reply(5_000, 12_345, Locale::En),
            "Lightning payment received ⚡\n+50.00 credits\nPersonal balance: 123.45 credits"
        );
    }
}

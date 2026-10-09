//! PNG QR codes for payment requests.

use std::io::Cursor;

use image::{ImageFormat, Luma};
use qrcode::QrCode;

/// Smallest side of the image, so phones scan it from a chat bubble.
const MIN_SIDE_PIXELS: u32 = 512;

/// A PNG QR code for a BOLT11 invoice, as a `lightning:` link so wallets
/// open it directly. Uppercase uses the denser alphanumeric mode.
#[must_use]
pub fn lightning_invoice_png(payreq: &str) -> Option<Vec<u8>> {
    let code = QrCode::new(format!("lightning:{payreq}").to_uppercase()).ok()?;
    let image = code
        .render::<Luma<u8>>()
        .min_dimensions(MIN_SIDE_PIXELS, MIN_SIDE_PIXELS)
        .build();
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, ImageFormat::Png).ok()?;
    Some(png.into_inner())
}

#[cfg(test)]
mod tests {
    use super::lightning_invoice_png;

    #[test]
    fn invoices_become_large_png_codes_and_oversized_text_is_refused() {
        let png = lightning_invoice_png("lnbc10n1synthetic").unwrap_or_default();
        assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));
        let decoded = image::load_from_memory(&png).map(|image| image.width());
        assert!(decoded.is_ok_and(|width| width >= 512));
        // Past what a QR code can hold.
        assert_eq!(lightning_invoice_png(&"a".repeat(8_000)), None);
    }
}

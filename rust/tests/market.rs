use respondedorbot::market::{build_rulo_lines, compute_devo_message, powerlaw_message, rainbow_message, RuloPath};

#[test]
fn test_compute_devo_message() {
    let msg = compute_devo_message(0.05, 0.0, 1000.0, 900.0, 1200.0);
    assert!(msg.contains("Profit: 4.17%"));
    assert!(msg.contains("Fee: 5.00%"));
}

#[test]
fn test_compute_devo_with_compra() {
    let msg = compute_devo_message(0.05, 100.0, 1000.0, 900.0, 1200.0);
    assert!(msg.contains("Ganarias"));
}

#[test]
fn test_build_rulo_lines() {
    let lines = build_rulo_lines(
        1000.0,
        1000.0,
        1100.0,
        1200.0,
        Some(RuloPath {
            usd_to_usdt_exchange: "binance".to_string(),
            usd_to_usdt_rate: 1.0,
            usdt_to_ars_exchange: "belo".to_string(),
            usdt_to_ars_rate: 1300.0,
        }),
    );
    let joined = lines.join("\n");
    assert!(joined.contains("USDT"));
    assert!(joined.contains("Ganancia"));
}

#[test]
fn test_powerlaw_message() {
    let msg = powerlaw_message(120.0, 100.0);
    assert!(msg.contains("caro"));
}

#[test]
fn test_rainbow_message() {
    let msg = rainbow_message(80.0, 100.0);
    assert!(msg.contains("regalado"));
}

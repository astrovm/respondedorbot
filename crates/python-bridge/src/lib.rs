//! Temporary Python bridge for incrementally adopting `bot-core`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bot_core::command_parsing::parse_command as parse_command_core;
use bot_core::credit_units::{
    CreditUnits, format_credit_units as format_credit_units_core,
    parse_credit_units as parse_credit_units_core,
    rescale_credit_units as rescale_credit_units_core,
    whole_credits_to_units as whole_credits_to_units_core,
};

/// Return the compatibility protocol version shared with Python.
#[pyfunction]
fn migration_protocol_version() -> u16 {
    bot_core::migration_protocol_version()
}

/// Convert whole credits to stored hundredth-credit units.
#[pyfunction]
fn whole_credits_to_units(credits: i64) -> PyResult<i64> {
    whole_credits_to_units_core(credits)
        .map(CreditUnits::value)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Rescale legacy units to stored hundredth-credit units.
#[pyfunction]
fn rescale_credit_units(units: i64, source_scale: i64) -> PyResult<i64> {
    rescale_credit_units_core(units, Some(source_scale))
        .map(CreditUnits::value)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Parse a human credit amount into stored hundredth-credit units.
#[pyfunction]
fn parse_credit_units(value: &str) -> Option<i64> {
    parse_credit_units_core(value).map(CreditUnits::value)
}

/// Format stored hundredth-credit units with two decimal places.
#[pyfunction]
fn format_credit_units(units: i64) -> String {
    format_credit_units_core(CreditUnits::new(units))
}

/// Normalize one Telegram command token and its remaining text.
#[pyfunction]
fn parse_command(message_text: &str, bot_name: &str) -> (String, String) {
    let parsed = parse_command_core(message_text, bot_name);
    (parsed.command, parsed.message_text)
}

/// Register the temporary `respondedorbot_rs` Python module.
#[pymodule]
fn respondedorbot_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(migration_protocol_version, module)?)?;
    module.add_function(wrap_pyfunction!(whole_credits_to_units, module)?)?;
    module.add_function(wrap_pyfunction!(rescale_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(parse_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(format_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(parse_command, module)?)?;
    Ok(())
}

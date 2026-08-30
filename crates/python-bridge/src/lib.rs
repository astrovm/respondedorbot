//! Temporary Python bridge for incrementally adopting `bot-core`.

use pyo3::prelude::*;

/// Return the compatibility protocol version shared with Python.
#[pyfunction]
fn migration_protocol_version() -> u16 {
    bot_core::migration_protocol_version()
}

/// Register the temporary `respondedorbot_rs` Python module.
#[pymodule]
fn respondedorbot_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(migration_protocol_version, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_the_core_protocol_version() {
        assert_eq!(super::migration_protocol_version(), 1);
    }
}

//! Native process entrypoint.

use std::process::ExitCode;

fn main() -> ExitCode {
    botd::cli::run(std::env::args())
}

//! Native `random_choice` AI tool backed by injected randomness.

use std::fmt::Display;

use bot_core::locale::Locale;
use bot_core::random_selection::{RandomSelection, parse_random_selection};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::dispatcher::RandomSource;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub struct RandomChoiceTool<Random> {
    random: Random,
    locale: Locale,
}

impl<Random> RandomChoiceTool<Random> {
    #[must_use]
    pub const fn new(random: Random, locale: Locale) -> Self {
        Self { random, locale }
    }
}

impl<Random> ExternalToolExecutor for RandomChoiceTool<Random>
where
    Random: RandomSource,
    Random::Error: Display,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::RandomChoice { request } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "random_choice",
            ));
        };
        let selection = match parse_random_selection(&request) {
            Ok(selection) => selection,
            Err(_) => RandomSelection::Invalid,
        };
        let result = match selection {
            RandomSelection::Invalid => invalid(self.locale),
            RandomSelection::Choices { values } => match self.random.choice_index(values.len()) {
                Ok(index) => values
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| invalid(self.locale)),
                Err(error) => {
                    return random_failure(self.locale, &error);
                }
            },
            RandomSelection::InclusiveRange { start, end } => {
                match self.random.inclusive_integer(&start, &end) {
                    Ok(value) => value.to_string(),
                    Err(error) => return random_failure(self.locale, &error),
                }
            }
        };
        ToolExecutionResult::output(result)
    }
}

fn random_failure(locale: Locale, error: &impl Display) -> ToolExecutionResult {
    ToolExecutionResult::with_diagnostics(
        tool_output::failed(locale, "random_choice"),
        vec![format!("random_choice source failed: {error}")],
    )
}

fn invalid(locale: Locale) -> String {
    match locale {
        Locale::Es => {
            "mandate algo como 'pizza, carne, sushi' o '1-10' boludo, no me hagas laburar al pedo"
                .to_owned()
        }
        Locale::En => "send options like 'pizza, steak, sushi' or a range like '1-10'".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::*;

    struct Random {
        choice: Result<usize, &'static str>,
        integer: Result<BigInt, &'static str>,
    }

    impl RandomSource for Random {
        type Error = &'static str;

        fn choice_index(&mut self, _upper_exclusive: usize) -> Result<usize, Self::Error> {
            self.choice
        }

        fn inclusive_integer(
            &mut self,
            _start: &BigInt,
            _end: &BigInt,
        ) -> Result<BigInt, Self::Error> {
            self.integer.clone()
        }
    }

    fn request(value: &str) -> ExternalToolRequest {
        ExternalToolRequest::RandomChoice {
            request: value.to_owned(),
        }
    }

    #[test]
    fn selects_choices_and_arbitrary_precision_inclusive_integers() {
        let mut tool = RandomChoiceTool::new(
            Random {
                choice: Ok(1),
                integer: Ok(BigInt::from(100_u8) * BigInt::from(10_u8).pow(18) + 1),
            },
            Locale::En,
        );
        assert_eq!(
            tool.execute(request("alpha, beta, gamma"), "call").output,
            "beta"
        );
        assert_eq!(
            tool.execute(
                request("100000000000000000000-100000000000000000002"),
                "call"
            )
            .output,
            "100000000000000000001"
        );
    }

    #[test]
    fn invalid_out_of_range_compatibility_digits_and_random_failures_are_safe() {
        let mut tool = RandomChoiceTool::new(
            Random {
                choice: Ok(99),
                integer: Err("synthetic random failure"),
            },
            Locale::Es,
        );
        assert_eq!(
            tool.execute(request("invalid"), "call").output,
            invalid(Locale::Es)
        );
        assert_eq!(
            tool.execute(request("🧉"), "call").output,
            invalid(Locale::Es)
        );
        let result = tool.execute(request("１-３"), "call");
        assert_eq!(result.output, "falló la herramienta 'random_choice'");
        assert!(result.diagnostics[0].contains("synthetic random failure"));
        assert_eq!(
            tool.execute(request("a,b"), "call").output,
            invalid(Locale::Es)
        );
        let result = tool.execute(request("1-3"), "call");
        assert_eq!(result.output, "falló la herramienta 'random_choice'");
        assert!(result.diagnostics[0].contains("synthetic random failure"));
    }

    #[test]
    fn incompatible_request_and_choice_source_failure_are_explicit() {
        let mut tool = RandomChoiceTool::new(
            Random {
                choice: Err("synthetic choice failure"),
                integer: Ok(BigInt::from(1)),
            },
            Locale::En,
        );
        assert_eq!(
            tool.execute(ExternalToolRequest::TaskList, "call").output,
            "tool 'random_choice' received an incompatible request"
        );
        let result = tool.execute(request("a,b"), "call");
        assert_eq!(result.output, "tool 'random_choice' failed");
        assert!(result.diagnostics[0].contains("synthetic choice failure"));
    }
}

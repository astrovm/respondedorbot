//! Validation of provider tool arguments into typed external requests.

use std::collections::BTreeMap;

use bot_core::locale::Locale;
use serde_json::Value;

use crate::chat_tool_loop::ToolExecutionResult;
use crate::native_tools::{NativeTool, NativeToolPorts};
use crate::tool_output;

#[derive(Debug, Clone, PartialEq)]
pub enum ExternalToolRequest {
    CryptoPrices {
        query: String,
    },
    WebFetch {
        url: String,
    },
    WebSearch {
        query: String,
    },
    TaskSet {
        text: String,
        delay_seconds: Option<i64>,
        interval_seconds: Option<i64>,
        trigger_config: Option<Value>,
    },
    TaskList,
    TaskCancel {
        task_id: String,
    },
    GetChatMembers,
    StockPrices {
        query: String,
    },
    RandomChoice {
        request: String,
    },
    DollarRates {
        timeframe: String,
    },
    Weather {
        location: String,
    },
    HackerNews {
        limit: u8,
    },
    BotCapabilities,
}

impl ExternalToolRequest {
    #[must_use]
    pub const fn tool(&self) -> NativeTool {
        match self {
            Self::CryptoPrices { .. } => NativeTool::CryptoPrices,
            Self::WebFetch { .. } => NativeTool::WebFetch,
            Self::WebSearch { .. } => NativeTool::WebSearch,
            Self::TaskSet { .. } => NativeTool::TaskSet,
            Self::TaskList => NativeTool::TaskList,
            Self::TaskCancel { .. } => NativeTool::TaskCancel,
            Self::GetChatMembers => NativeTool::GetChatMembers,
            Self::StockPrices { .. } => NativeTool::StockPrices,
            Self::RandomChoice { .. } => NativeTool::RandomChoice,
            Self::DollarRates { .. } => NativeTool::DollarRates,
            Self::Weather { .. } => NativeTool::Weather,
            Self::HackerNews { .. } => NativeTool::HackerNews,
            Self::BotCapabilities => NativeTool::BotCapabilities,
        }
    }
}

pub trait ExternalToolServices {
    fn is_available(&self, tool: NativeTool) -> bool;

    fn execute(&mut self, request: ExternalToolRequest, tool_call_id: &str) -> ToolExecutionResult;
}

pub trait ExternalToolExecutor {
    fn execute(&mut self, request: ExternalToolRequest, tool_call_id: &str) -> ToolExecutionResult;
}

pub struct ExternalToolbox {
    executors: BTreeMap<NativeTool, Box<dyn ExternalToolExecutor>>,
    locale: Locale,
}

impl ExternalToolbox {
    #[must_use]
    pub const fn new(locale: Locale) -> Self {
        Self {
            executors: BTreeMap::new(),
            locale,
        }
    }

    #[must_use]
    pub fn with_executor(
        mut self,
        tool: NativeTool,
        executor: Box<dyn ExternalToolExecutor>,
    ) -> Self {
        self.executors.insert(tool, executor);
        self
    }
}

impl ExternalToolServices for ExternalToolbox {
    fn is_available(&self, tool: NativeTool) -> bool {
        self.executors.contains_key(&tool)
    }

    fn execute(&mut self, request: ExternalToolRequest, tool_call_id: &str) -> ToolExecutionResult {
        let tool = request.tool();
        self.executors.get_mut(&tool).map_or_else(
            || ToolExecutionResult::output(tool_output::unavailable(self.locale, tool.name())),
            |executor| executor.execute(request, tool_call_id),
        )
    }
}

pub struct ValidatedNativeToolPorts<Services> {
    services: Services,
    locale: Locale,
}

impl<Services> ValidatedNativeToolPorts<Services> {
    #[must_use]
    pub const fn new(services: Services, locale: Locale) -> Self {
        Self { services, locale }
    }

    #[must_use]
    pub const fn services(&self) -> &Services {
        &self.services
    }
}

impl<Services: ExternalToolServices> NativeToolPorts for ValidatedNativeToolPorts<Services> {
    fn is_available(&self, tool: NativeTool) -> bool {
        self.services.is_available(tool)
    }

    fn execute_external(
        &mut self,
        tool: NativeTool,
        arguments: &Value,
        tool_call_id: &str,
    ) -> ToolExecutionResult {
        match validate_request(tool, arguments, self.locale) {
            Ok(request) => self.services.execute(request, tool_call_id),
            Err(output) => ToolExecutionResult::output(output),
        }
    }
}

pub(crate) fn validate_request(
    tool: NativeTool,
    arguments: &Value,
    locale: Locale,
) -> Result<ExternalToolRequest, String> {
    match tool {
        NativeTool::CryptoPrices => {
            let assets = bounded_string_list(arguments.get("assets"), 20);
            if assets.is_empty() {
                return Err(localized(
                    locale,
                    "indicá al menos una crypto",
                    "provide at least one cryptocurrency",
                ));
            }
            let convert = text(arguments, "convert");
            let timeframe = text(arguments, "timeframe");
            Ok(ExternalToolRequest::CryptoPrices {
                query: format!(
                    "{} in {} {}",
                    assets.join(","),
                    if convert.is_empty() {
                        "USD".to_owned()
                    } else {
                        convert.to_uppercase()
                    },
                    if timeframe.is_empty() {
                        "24h".to_owned()
                    } else {
                        timeframe.to_lowercase()
                    }
                ),
            })
        }
        NativeTool::Calculate => Err("calculate is handled by bot-core".to_owned()),
        NativeTool::WebFetch => required_text(arguments, "url").map_or_else(
            || {
                Err(localized(
                    locale,
                    "no se proporcionó una URL",
                    "no URL was provided",
                ))
            },
            |url| Ok(ExternalToolRequest::WebFetch { url }),
        ),
        NativeTool::WebSearch => {
            let query = collapse_and_limit(&text(arguments, "query"), 500);
            if query.is_empty() {
                Err(localized(
                    locale,
                    "Error de búsqueda: falta la consulta.",
                    "Search error: missing query.",
                ))
            } else {
                Ok(ExternalToolRequest::WebSearch { query })
            }
        }
        NativeTool::TaskSet => required_text(arguments, "text").map_or_else(
            || {
                Err(localized(
                    locale,
                    "no se que tarea crear, pasame el texto",
                    "send the task text",
                ))
            },
            |text| {
                Ok(ExternalToolRequest::TaskSet {
                    text,
                    delay_seconds: integer(arguments, "delay_seconds"),
                    interval_seconds: integer(arguments, "interval_seconds"),
                    trigger_config: arguments
                        .get("trigger_config")
                        .filter(|value| value.is_object())
                        .cloned(),
                })
            },
        ),
        NativeTool::TaskList => Ok(ExternalToolRequest::TaskList),
        NativeTool::TaskCancel => required_text(arguments, "task_id").map_or_else(
            || {
                Err(localized(
                    locale,
                    "necesito el ID de la tarea, usá /tareas para verlas",
                    "provide the task ID; use /tasks to list them",
                ))
            },
            |task_id| Ok(ExternalToolRequest::TaskCancel { task_id }),
        ),
        NativeTool::GetChatMembers => Ok(ExternalToolRequest::GetChatMembers),
        NativeTool::StockPrices => {
            let queries = bounded_string_list(arguments.get("queries"), 20);
            if queries.is_empty() {
                Err(localized(
                    locale,
                    "indicá al menos un símbolo o empresa",
                    "provide at least one symbol or company",
                ))
            } else {
                Ok(ExternalToolRequest::StockPrices {
                    query: queries.join(","),
                })
            }
        }
        NativeTool::RandomChoice => required_text(arguments, "request").map_or_else(
            || {
                Err(localized(
                    locale,
                    "indicá opciones o un rango numérico",
                    "provide options or a numeric range",
                ))
            },
            |request| Ok(ExternalToolRequest::RandomChoice { request }),
        ),
        NativeTool::DollarRates => Ok(ExternalToolRequest::DollarRates {
            timeframe: text(arguments, "timeframe"),
        }),
        NativeTool::Weather => required_text(arguments, "location").map_or_else(
            || {
                Err(localized(
                    locale,
                    "indicá una ciudad o ubicación",
                    "provide a city or location",
                ))
            },
            |location| Ok(ExternalToolRequest::Weather { location }),
        ),
        NativeTool::HackerNews => {
            let limit = integer(arguments, "limit").unwrap_or(5).clamp(1, 10);
            Ok(ExternalToolRequest::HackerNews {
                limit: u8::try_from(limit).unwrap_or(5),
            })
        }
        NativeTool::BotCapabilities => Ok(ExternalToolRequest::BotCapabilities),
    }
}

fn localized(locale: Locale, spanish: &str, english: &str) -> String {
    match locale {
        Locale::Es => spanish.to_owned(),
        Locale::En => english.to_owned(),
    }
}

fn text(arguments: &Value, key: &str) -> String {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn required_text(arguments: &Value, key: &str) -> Option<String> {
    let value = text(arguments, key);
    (!value.is_empty()).then_some(value)
}

fn integer(arguments: &Value, key: &str) -> Option<i64> {
    arguments.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })
}

fn bounded_string_list(value: Option<&Value>, limit: usize) -> Vec<String> {
    match value {
        Some(Value::String(value)) => nonempty(value).into_iter().collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .filter_map(nonempty)
            .take(limit)
            .collect(),
        _ => Vec::new(),
    }
}

fn nonempty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn collapse_and_limit(value: &str, limit: usize) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::native_tools::{NativeToolRegistry, StandardNativeToolBackend};

    #[derive(Default)]
    struct Services {
        calls: Vec<(ExternalToolRequest, String)>,
    }

    impl ExternalToolServices for Services {
        fn is_available(&self, tool: NativeTool) -> bool {
            !matches!(tool, NativeTool::WebFetch)
        }

        fn execute(
            &mut self,
            request: ExternalToolRequest,
            tool_call_id: &str,
        ) -> ToolExecutionResult {
            self.calls.push((request, tool_call_id.to_owned()));
            ToolExecutionResult::output("synthetic result")
        }
    }

    fn ports(locale: Locale) -> ValidatedNativeToolPorts<Services> {
        ValidatedNativeToolPorts::new(Services::default(), locale)
    }

    #[test]
    fn validates_and_normalizes_market_weather_random_and_news_requests() {
        let cases = [
            (
                NativeTool::CryptoPrices,
                json!({"assets": [" BTC ", "", "ETH"], "convert": "ars", "timeframe": "7D"}),
                ExternalToolRequest::CryptoPrices {
                    query: "BTC,ETH in ARS 7d".to_owned(),
                },
            ),
            (
                NativeTool::StockPrices,
                json!({"queries": " AAPL "}),
                ExternalToolRequest::StockPrices {
                    query: "AAPL".to_owned(),
                },
            ),
            (
                NativeTool::RandomChoice,
                json!({"request": " one, two "}),
                ExternalToolRequest::RandomChoice {
                    request: "one, two".to_owned(),
                },
            ),
            (
                NativeTool::DollarRates,
                json!({"timeframe": " 6h "}),
                ExternalToolRequest::DollarRates {
                    timeframe: "6h".to_owned(),
                },
            ),
            (
                NativeTool::Weather,
                json!({"location": " Synthetic City "}),
                ExternalToolRequest::Weather {
                    location: "Synthetic City".to_owned(),
                },
            ),
            (
                NativeTool::HackerNews,
                json!({"limit": 100}),
                ExternalToolRequest::HackerNews { limit: 10 },
            ),
        ];
        let mut ports = ports(Locale::En);
        for (tool, arguments, expected) in cases {
            let result = ports.execute_external(tool, &arguments, "call-1");
            assert_eq!(result.output, "synthetic result");
            assert_eq!(
                ports.services().calls.last().map(|call| &call.0),
                Some(&expected)
            );
        }
    }

    #[test]
    fn validates_web_and_task_requests_without_leaking_raw_provider_shapes() {
        let mut ports = ports(Locale::En);
        let long_query = format!("  {}\n next  ", "x".repeat(510));
        ports.execute_external(
            NativeTool::WebSearch,
            &json!({"query": long_query}),
            "search-1",
        );
        ports.execute_external(
            NativeTool::TaskSet,
            &json!({
                "text": " remind me ",
                "delay_seconds": "60",
                "interval_seconds": 120,
                "trigger_config": {"type": "interval", "days": 1}
            }),
            "task-1",
        );
        ports.execute_external(
            NativeTool::TaskCancel,
            &json!({"task_id": " task-7 "}),
            "task-2",
        );
        assert!(matches!(
            &ports.services().calls[0].0,
            ExternalToolRequest::WebSearch { query } if query.chars().count() == 500
        ));
        assert_eq!(
            ports.services().calls[1].0,
            ExternalToolRequest::TaskSet {
                text: "remind me".to_owned(),
                delay_seconds: Some(60),
                interval_seconds: Some(120),
                trigger_config: Some(json!({"type": "interval", "days": 1})),
            }
        );
        assert_eq!(
            ports.services().calls[2].0,
            ExternalToolRequest::TaskCancel {
                task_id: "task-7".to_owned()
            }
        );
    }

    #[test]
    fn missing_arguments_return_exact_localized_results_without_side_effects() {
        for (tool, arguments, spanish, english) in [
            (
                NativeTool::CryptoPrices,
                json!({"assets": []}),
                "indicá al menos una crypto",
                "provide at least one cryptocurrency",
            ),
            (
                NativeTool::StockPrices,
                json!({}),
                "indicá al menos un símbolo o empresa",
                "provide at least one symbol or company",
            ),
            (
                NativeTool::Weather,
                json!({"location": " "}),
                "indicá una ciudad o ubicación",
                "provide a city or location",
            ),
            (
                NativeTool::RandomChoice,
                json!({}),
                "indicá opciones o un rango numérico",
                "provide options or a numeric range",
            ),
            (
                NativeTool::WebFetch,
                json!({}),
                "no se proporcionó una URL",
                "no URL was provided",
            ),
            (
                NativeTool::WebSearch,
                json!({"query": " \n "}),
                "Error de búsqueda: falta la consulta.",
                "Search error: missing query.",
            ),
            (
                NativeTool::TaskSet,
                json!({}),
                "no se que tarea crear, pasame el texto",
                "send the task text",
            ),
            (
                NativeTool::TaskCancel,
                json!({}),
                "necesito el ID de la tarea, usá /tareas para verlas",
                "provide the task ID; use /tasks to list them",
            ),
        ] {
            let mut spanish_ports = ports(Locale::Es);
            let mut english_ports = ports(Locale::En);
            assert_eq!(
                spanish_ports
                    .execute_external(tool, &arguments, "call")
                    .output,
                spanish
            );
            assert_eq!(
                english_ports
                    .execute_external(tool, &arguments, "call")
                    .output,
                english
            );
            assert!(spanish_ports.services().calls.is_empty());
            assert!(english_ports.services().calls.is_empty());
        }
    }

    #[test]
    fn registry_filters_unavailable_services_and_keeps_pure_calculation() {
        let ports = ports(Locale::En);
        let backend = StandardNativeToolBackend::new(ports, Locale::En);
        let registry = NativeToolRegistry::new(backend, Locale::En);
        assert!(!NativeToolPorts::is_available(
            registry.backend().ports(),
            NativeTool::WebFetch
        ));
        assert!(NativeToolPorts::is_available(
            registry.backend().ports(),
            NativeTool::Weather
        ));
    }

    struct Executor;

    impl ExternalToolExecutor for Executor {
        fn execute(
            &mut self,
            request: ExternalToolRequest,
            tool_call_id: &str,
        ) -> ToolExecutionResult {
            ToolExecutionResult::output(format!("{}:{tool_call_id}", request.tool().name()))
        }
    }

    #[test]
    fn toolbox_has_one_executor_per_typed_tool() {
        let mut toolbox = ExternalToolbox::new(Locale::En)
            .with_executor(NativeTool::BotCapabilities, Box::new(Executor));
        assert!(toolbox.is_available(NativeTool::BotCapabilities));
        assert!(!toolbox.is_available(NativeTool::Weather));
        assert_eq!(
            toolbox
                .execute(ExternalToolRequest::BotCapabilities, "call-1")
                .output,
            "bot_capabilities:call-1"
        );
        assert_eq!(
            toolbox
                .execute(
                    ExternalToolRequest::Weather {
                        location: "Synthetic City".to_owned()
                    },
                    "call-2"
                )
                .output,
            "tool 'weather' is unavailable"
        );
    }
}

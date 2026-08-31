//! Declarative native AI tool catalog and side-effect boundary.

use serde_json::{Value, json};

use bot_core::ai_calculator::calculate_expression;
use bot_core::ai_capabilities::render_ai_capabilities;
use bot_core::locale::Locale;

use crate::chat_tool_loop::{NativeToolRuntime, ToolExecutionResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NativeTool {
    CryptoPrices,
    Calculate,
    WebFetch,
    WebSearch,
    TaskSet,
    TaskList,
    TaskCancel,
    GetChatMembers,
    StockPrices,
    RandomChoice,
    DollarRates,
    Weather,
    HackerNews,
    BotCapabilities,
}

impl NativeTool {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CryptoPrices => "crypto_prices",
            Self::Calculate => "calculate",
            Self::WebFetch => "web_fetch",
            Self::WebSearch => "web_search",
            Self::TaskSet => "task_set",
            Self::TaskList => "task_list",
            Self::TaskCancel => "task_cancel",
            Self::GetChatMembers => "get_chat_members",
            Self::StockPrices => "stock_prices",
            Self::RandomChoice => "random_choice",
            Self::DollarRates => "dollar_rates",
            Self::Weather => "weather",
            Self::HackerNews => "hacker_news",
            Self::BotCapabilities => "bot_capabilities",
        }
    }

    #[must_use]
    pub const fn task_allowed(self) -> bool {
        !matches!(self, Self::TaskSet)
    }

    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        ALL_TOOLS.iter().copied().find(|tool| tool.name() == name)
    }
}

const ALL_TOOLS: [NativeTool; 14] = [
    NativeTool::CryptoPrices,
    NativeTool::Calculate,
    NativeTool::WebFetch,
    NativeTool::WebSearch,
    NativeTool::TaskSet,
    NativeTool::TaskList,
    NativeTool::TaskCancel,
    NativeTool::GetChatMembers,
    NativeTool::StockPrices,
    NativeTool::RandomChoice,
    NativeTool::DollarRates,
    NativeTool::Weather,
    NativeTool::HackerNews,
    NativeTool::BotCapabilities,
];

pub trait NativeToolBackend {
    fn is_available(&self, tool: NativeTool) -> bool;

    fn execute(
        &mut self,
        tool: NativeTool,
        arguments: &Value,
        tool_call_id: &str,
    ) -> ToolExecutionResult;
}

pub trait NativeToolPorts {
    fn is_available(&self, tool: NativeTool) -> bool;

    fn execute_external(
        &mut self,
        tool: NativeTool,
        arguments: &Value,
        tool_call_id: &str,
    ) -> ToolExecutionResult;
}

pub struct StandardNativeToolBackend<Ports> {
    ports: Ports,
    locale: Locale,
}

impl<Ports> StandardNativeToolBackend<Ports> {
    #[must_use]
    pub const fn new(ports: Ports, locale: Locale) -> Self {
        Self { ports, locale }
    }

    #[must_use]
    pub const fn ports(&self) -> &Ports {
        &self.ports
    }
}

impl<Ports: NativeToolPorts> NativeToolBackend for StandardNativeToolBackend<Ports> {
    fn is_available(&self, tool: NativeTool) -> bool {
        matches!(tool, NativeTool::Calculate | NativeTool::BotCapabilities)
            || self.ports.is_available(tool)
    }

    fn execute(
        &mut self,
        tool: NativeTool,
        arguments: &Value,
        tool_call_id: &str,
    ) -> ToolExecutionResult {
        if tool == NativeTool::Calculate {
            let expression = arguments
                .get("expression")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return ToolExecutionResult::output(calculate_expression(expression, self.locale));
        }
        if tool == NativeTool::BotCapabilities {
            return ToolExecutionResult::output(render_ai_capabilities(self.locale));
        }
        self.ports.execute_external(tool, arguments, tool_call_id)
    }
}

pub struct NativeToolRegistry<Backend> {
    backend: Backend,
}

impl<Backend> NativeToolRegistry<Backend> {
    #[must_use]
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    #[must_use]
    pub const fn backend(&self) -> &Backend {
        &self.backend
    }
}

impl<Backend: NativeToolBackend> NativeToolRuntime for NativeToolRegistry<Backend> {
    fn schemas(&self, task_mode: bool) -> Vec<Value> {
        ALL_TOOLS
            .iter()
            .copied()
            .filter(|tool| self.backend.is_available(*tool))
            .filter(|tool| !task_mode || tool.task_allowed())
            .map(tool_schema)
            .collect()
    }

    fn contains(&self, name: &str, task_mode: bool) -> bool {
        NativeTool::from_name(name).is_some_and(|tool| {
            self.backend.is_available(tool) && (!task_mode || tool.task_allowed())
        })
    }

    fn execute(
        &mut self,
        name: &str,
        arguments: &Value,
        tool_call_id: &str,
    ) -> ToolExecutionResult {
        NativeTool::from_name(name).map_or_else(
            || ToolExecutionResult::output(format!("Unknown tool: {name}")),
            |tool| self.backend.execute(tool, arguments, tool_call_id),
        )
    }
}

fn tool_schema(tool: NativeTool) -> Value {
    let (description, parameters) = match tool {
        NativeTool::CryptoPrices => (
            "Get cryptocurrency prices from CoinMarketCap by symbol or slug, with a quote currency and change timeframe.",
            json!({
                "type": "object",
                "properties": {
                    "assets": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20, "description": "CoinMarketCap symbols or slugs, such as BTC or bitcoin-cash."},
                    "convert": {"type": "string", "description": "Quote currency symbol, such as USD, EUR, ARS, or BTC.", "default": "USD"},
                    "timeframe": {"type": "string", "enum": ["1h", "24h", "7d", "30d"], "default": "24h"}
                },
                "required": ["assets"]
            }),
        ),
        NativeTool::Calculate => (
            "Evaluate a mathematical expression safely. Supports +, -, *, /, %, **. Example: '2 ** 10' or '100 / 35000 * 100'.",
            json!({
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate, e.g. '150000 / 35000 * 100'"}},
                "required": ["expression"]
            }),
        ),
        NativeTool::WebFetch => (
            "Fetch and extract text content from a URL. Returns the page title and visible text content.",
            json!({
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch and read"}},
                "required": ["url"]
            }),
        ),
        NativeTool::WebSearch => (
            "Search the public web with Firecrawl. Use it for current facts or when the user asks you to search. The result contains source URLs and descriptions.",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A concise web search query", "maxLength": 500}},
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
        NativeTool::TaskSet => (
            "Create a scheduled task. Put only the future instruction in text and preserve its subject, perspective, and pronouns. Put time or frequency only in delay_seconds, interval_seconds, or trigger_config. For example, 'tomorrow remind me to pay' uses text='remind me to pay' plus a delay; 'every day at 20:30 tell me the score' uses text='tell me the score' plus a cron trigger. Choose a reasonable hour if the user omits one.",
            task_set_parameters(),
        ),
        NativeTool::TaskList => (
            "List all tasks (one-shot and recurring) for the current chat.",
            empty_parameters(),
        ),
        NativeTool::TaskCancel => (
            "Cancel a task by its ID. Use task_list or /tareas to get the ID.",
            json!({
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "The task ID to cancel"}},
                "required": ["task_id"]
            }),
        ),
        NativeTool::GetChatMembers => (
            "Get list of known chat members. Returns users who have sent messages in this group.",
            empty_parameters(),
        ),
        NativeTool::StockPrices => (
            "Get prices for stocks, ETFs, indexes, or futures from Yahoo Finance. Use exact Yahoo symbols when known; company names are also accepted.",
            json!({
                "type": "object",
                "properties": {"queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20, "description": "Yahoo symbols or company names."}},
                "required": ["queries"]
            }),
        ),
        NativeTool::RandomChoice => (
            "Choose one item at random or generate a random integer. This has the same behavior as the /random command.",
            json!({
                "type": "object",
                "properties": {"request": {"type": "string", "description": "Comma-separated options, such as 'option-alpha, option-beta', or an inclusive integer range, such as '1-10'."}},
                "required": ["request"]
            }),
        ),
        NativeTool::DollarRates => (
            "Get current Argentine dollar exchange rates and their change over an optional historical timeframe.",
            json!({
                "type": "object",
                "properties": {"timeframe": {"type": "string", "enum": ["1h", "6h", "12h", "24h", "48h"], "description": "Optional comparison timeframe. Defaults to 24h."}},
                "required": []
            }),
        ),
        NativeTool::Weather => (
            "Get the current weather for any city or location.",
            json!({
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City or location, including region or country when ambiguous."}},
                "required": ["location"]
            }),
        ),
        NativeTool::HackerNews => (
            "Get current top technology stories from Hacker News.",
            json!({
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Number of stories to return. Defaults to 5."}},
                "required": []
            }),
        ),
        NativeTool::BotCapabilities => (
            "Get the authoritative list of bot features and commands. Use when a user asks what the bot can do or which command to use.",
            empty_parameters(),
        ),
    };
    json!({
        "type": "function",
        "function": {
            "name": tool.name(),
            "description": description,
            "parameters": parameters
        }
    })
}

fn empty_parameters() -> Value {
    json!({"type": "object", "properties": {}, "required": []})
}

fn task_set_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Content-only future instruction the bot will execute later. Preserve perspective, subject, and pronouns, but exclude scheduling/time expressions that belong in delay_seconds, interval_seconds, or trigger_config."},
            "delay_seconds": {"type": "integer", "description": "Delay in seconds for one-shot task. 60=1min, 3600=1h, 86400=1d. Max 315360000 (10y)."},
            "interval_seconds": {"type": "integer", "description": "Interval in seconds for recurring task. 300=5min, 3600=1h, 86400=1d, 604800=1w."},
            "trigger_config": {
                "type": "object",
                "description": "Advanced trigger config with type=interval/cron. interval: {type:'interval', days:N}. cron: {type:'cron', hour:0-23, minute:0-59, day_of_week:'mon,wed' or 'lun,mie', day:1-31}",
                "properties": {
                    "type": {"type": "string", "enum": ["interval", "cron"]},
                    "days": {"type": "integer", "description": "For interval type: number of days between runs"},
                    "hour": {"type": "integer", "description": "For cron type: hour (0-23)"},
                    "minute": {"type": "integer", "description": "For cron type: minute (0-59)"},
                    "day_of_week": {"type": "string", "description": "For cron type: days of week in English or Spanish abbreviations (mon,wed,fri or lun,mie,vie)"},
                    "day": {"type": "integer", "description": "For cron type: day of month (1-31)"}
                }
            }
        },
        "required": ["text"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Backend {
        unavailable: Option<NativeTool>,
        calls: Vec<(NativeTool, Value, String)>,
    }

    impl NativeToolBackend for Backend {
        fn is_available(&self, tool: NativeTool) -> bool {
            self.unavailable != Some(tool)
        }

        fn execute(
            &mut self,
            tool: NativeTool,
            arguments: &Value,
            tool_call_id: &str,
        ) -> ToolExecutionResult {
            self.calls
                .push((tool, arguments.clone(), tool_call_id.to_owned()));
            ToolExecutionResult::output(tool.name())
        }
    }

    fn registry(unavailable: Option<NativeTool>) -> NativeToolRegistry<Backend> {
        NativeToolRegistry::new(Backend {
            unavailable,
            calls: Vec::new(),
        })
    }

    #[test]
    fn catalog_preserves_all_python_names_schemas_and_task_restriction() {
        let registry = registry(None);
        let normal = registry.schemas(false);
        let names = normal
            .iter()
            .filter_map(|schema| schema["function"]["name"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "crypto_prices",
                "calculate",
                "web_fetch",
                "web_search",
                "task_set",
                "task_list",
                "task_cancel",
                "get_chat_members",
                "stock_prices",
                "random_choice",
                "dollar_rates",
                "weather",
                "hacker_news",
                "bot_capabilities"
            ]
        );
        assert_eq!(normal[0]["function"]["parameters"]["required"][0], "assets");
        assert_eq!(
            normal[3]["function"]["parameters"]["additionalProperties"],
            false
        );
        assert_eq!(registry.schemas(true).len(), normal.len() - 1);
        assert!(!registry.contains("task_set", true));
        assert!(registry.contains("task_set", false));
    }

    #[test]
    fn availability_and_dispatch_have_one_typed_owner() {
        let mut registry = registry(Some(NativeTool::Weather));
        assert!(!registry.contains("weather", false));
        assert!(!registry.contains("unknown", false));
        assert_eq!(registry.schemas(false).len(), ALL_TOOLS.len() - 1);
        let result = registry.execute("calculate", &json!({"expression": "2+2"}), "call-1");
        assert_eq!(result.output, "calculate");
        assert_eq!(registry.backend().calls[0].0, NativeTool::Calculate);
        assert_eq!(registry.backend().calls[0].2, "call-1");
        assert_eq!(
            registry.execute("unknown", &json!({}), "call-2").output,
            "Unknown tool: unknown"
        );
    }

    struct Ports {
        calls: Vec<NativeTool>,
    }

    impl NativeToolPorts for Ports {
        fn is_available(&self, tool: NativeTool) -> bool {
            tool == NativeTool::Weather
        }

        fn execute_external(
            &mut self,
            tool: NativeTool,
            _arguments: &Value,
            _tool_call_id: &str,
        ) -> ToolExecutionResult {
            self.calls.push(tool);
            ToolExecutionResult::output("external")
        }
    }

    #[test]
    fn standard_backend_keeps_calculation_pure_and_other_io_behind_ports() {
        let backend = StandardNativeToolBackend::new(Ports { calls: Vec::new() }, Locale::En);
        let mut registry = NativeToolRegistry::new(backend);
        let schemas = registry.schemas(false);
        assert_eq!(schemas.len(), 3);
        assert!(registry.contains("calculate", false));
        assert!(registry.contains("weather", false));
        assert!(registry.contains("bot_capabilities", false));
        assert!(!registry.contains("web_fetch", false));

        assert_eq!(
            registry
                .execute("calculate", &json!({"expression": "2 ** 10"}), "call-1")
                .output,
            "1024"
        );
        assert!(
            registry
                .execute("bot_capabilities", &json!({}), "call-3")
                .output
                .starts_with("BOT CAPABILITIES:")
        );
        assert!(registry.backend().ports().calls.is_empty());
        assert_eq!(
            registry
                .execute("weather", &json!({"location": "Synthetic City"}), "call-2")
                .output,
            "external"
        );
        assert_eq!(registry.backend().ports().calls, [NativeTool::Weather]);
    }
}

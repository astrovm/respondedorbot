//! Runs read-only tools concurrently by giving each call its own copy.

use std::sync::Arc;

use bot_core::locale::Locale;

use crate::chat_tool_loop::{ConcurrentToolCall, ToolExecutionResult};
use crate::native_tools::NativeTool;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

/// Wraps a tool that only reads data. Sequential calls use the copy built
/// up front; each concurrent call builds a fresh one on its own thread, so
/// calls never share caches or clocks. Transports and Redis pools are shared
/// underneath, which keeps building cheap.
pub struct ReadOnlyTool<Tool, Build> {
    tool: Tool,
    build: Arc<Build>,
    name: &'static str,
    locale: Locale,
}

/// Build the tool once now, so setup errors surface like any other tool's.
pub fn read_only_tool<Tool, Build>(
    tool: NativeTool,
    locale: Locale,
    build: Build,
) -> Result<Box<dyn ExternalToolExecutor>, String>
where
    Tool: ExternalToolExecutor + 'static,
    Build: Fn() -> Result<Tool, String> + Send + Sync + 'static,
{
    Ok(Box::new(ReadOnlyTool {
        tool: build()?,
        build: Arc::new(build),
        name: tool.name(),
        locale,
    }))
}

impl<Tool, Build> ExternalToolExecutor for ReadOnlyTool<Tool, Build>
where
    Tool: ExternalToolExecutor,
    Build: Fn() -> Result<Tool, String> + Send + Sync + 'static,
{
    fn execute(&mut self, request: ExternalToolRequest, tool_call_id: &str) -> ToolExecutionResult {
        self.tool.execute(request, tool_call_id)
    }

    fn concurrent(
        &mut self,
        request: ExternalToolRequest,
        tool_call_id: &str,
    ) -> Option<ConcurrentToolCall> {
        let build = Arc::clone(&self.build);
        let tool_call_id = tool_call_id.to_owned();
        let (name, locale) = (self.name, self.locale);
        Some(Box::new(move || match build() {
            Ok(mut tool) => tool.execute(request, &tool_call_id),
            Err(error) => ToolExecutionResult::with_diagnostics(
                tool_output::failed(locale, name),
                vec![format!("{name} tool setup failed: {error}")],
            ),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct Counter(usize);

    impl ExternalToolExecutor for Counter {
        fn execute(
            &mut self,
            request: ExternalToolRequest,
            tool_call_id: &str,
        ) -> ToolExecutionResult {
            self.0 += 1;
            ToolExecutionResult::output(format!(
                "{} {tool_call_id} call {}",
                request.tool().name(),
                self.0
            ))
        }
    }

    fn weather() -> ExternalToolRequest {
        ExternalToolRequest::Weather {
            location: "Synthetic City".to_owned(),
        }
    }

    #[test]
    fn sequential_calls_reuse_one_copy_and_concurrent_calls_get_their_own()
    -> crate::test_env::TestResult {
        let builds = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&builds);
        let mut tool = read_only_tool(NativeTool::Weather, Locale::En, move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(Counter(0))
        })?;
        assert_eq!(tool.execute(weather(), "a").output, "weather a call 1");
        assert_eq!(tool.execute(weather(), "b").output, "weather b call 2");
        let run = tool
            .concurrent(weather(), "c")
            .ok_or("read-only tools run concurrently")?;
        let handle = std::thread::spawn(run);
        let output = handle.join().map_err(|_| "worker panicked")?.output;
        assert_eq!(output, "weather c call 1");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn setup_failures_surface_up_front_and_per_concurrent_call() -> crate::test_env::TestResult {
        let failing = read_only_tool(NativeTool::Weather, Locale::En, || {
            Err::<Counter, _>("synthetic setup failure".to_owned())
        });
        assert_eq!(failing.err().as_deref(), Some("synthetic setup failure"));

        let builds = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&builds);
        let mut tool = read_only_tool(NativeTool::Polymarket, Locale::Es, move || {
            if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(Counter(0))
            } else {
                Err("synthetic setup failure".to_owned())
            }
        })?;
        let run = tool
            .concurrent(weather(), "c")
            .ok_or("read-only tools run concurrently")?;
        let result = run();
        assert_eq!(result.output, "falló la herramienta 'polymarket'");
        assert_eq!(
            result.diagnostics,
            ["polymarket tool setup failed: synthetic setup failure"]
        );
        Ok(())
    }
}

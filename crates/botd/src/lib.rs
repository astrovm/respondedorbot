//! Application composition and lifecycle for the native bot process.

pub mod chat_provider;
pub mod chat_tool_loop;
pub mod composition;
pub mod config;
pub mod dispatcher;
pub mod firecrawl_tool;
pub mod native_ai;
pub mod native_tools;
pub mod random_tool;
pub mod runtime;
pub mod scheduler;
pub mod task_executor;
pub mod task_service;
pub mod tool_requests;

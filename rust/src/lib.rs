//! Cuba-Memorys — Knowledge Graph MCP Server.
//!
//! Exposes modules for integration testing.
//! The binary entry point is in `main.rs`.

pub mod calibrate_cli;
pub mod cli;
pub mod cognitive;
pub mod constants;
pub mod core;
pub mod dashboard;
pub mod db;
pub mod doctor;
pub mod embeddings;
pub mod eval;
pub mod export;
pub mod graph;
pub mod handlers;
pub mod link_cli;
pub mod observability;
pub mod project;
pub mod protocol;
pub mod recall_cli;
pub mod reembed_cli;
pub mod search;
pub mod session;
pub mod setup;
pub mod setup_agent;
pub mod skills_cli;
pub mod sync;
pub mod tasks;

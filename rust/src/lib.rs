//! Cuba-Memorys — Knowledge Graph MCP Server.
//!
//! Exposes modules for integration testing.
//! The binary entry point is in `main.rs`.

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
pub mod observability;
pub mod project;
pub mod protocol;
pub mod search;
pub mod session;
pub mod setup;
pub mod setup_agent;
pub mod sync;
pub mod tasks;

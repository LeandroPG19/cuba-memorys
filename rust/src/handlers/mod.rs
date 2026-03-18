//! Handler dispatch — routes tool names to handler functions.
//!
//! Each handler is in its own module file to keep CC ≤ 7.
//! AI-2: Instrumented with tracing spans for RED metrics observability.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub mod alarma;
pub mod alma;
pub mod cronica;
pub mod decreto;
pub mod eco;
pub mod expediente;
pub mod faro;
pub mod forget;
pub mod jornada;
pub mod puente;
pub mod remedio;
pub mod vigia;
pub mod zafra;

/// Dispatch a tool call to the appropriate handler.
///
/// Returns MCP content response (text or structured).
///
/// AI-2: Instrumented with tracing span for observability (RED metrics).
/// - Rate: span count per tool_name
/// - Errors: error events within span
/// - Duration: elapsed_ms field in span close
#[tracing::instrument(skip(pool, args), fields(tool = %tool_name))]
pub async fn dispatch(pool: &PgPool, tool_name: &str, args: Value) -> Result<Value> {
    let start = std::time::Instant::now();

    let result = match tool_name {
        "cuba_alma" => alma::handle(pool, args).await?,
        "cuba_cronica" => cronica::handle(pool, args).await?,
        "cuba_faro" => faro::handle(pool, args).await?,
        "cuba_forget" => forget::handle(pool, args).await?,
        "cuba_puente" => puente::handle(pool, args).await?,
        "cuba_eco" => eco::handle(pool, args).await?,
        "cuba_alarma" => alarma::handle(pool, args).await?,
        "cuba_remedio" => remedio::handle(pool, args).await?,
        "cuba_expediente" => expediente::handle(pool, args).await?,
        "cuba_jornada" => jornada::handle(pool, args).await?,
        "cuba_decreto" => decreto::handle(pool, args).await?,
        "cuba_vigia" => vigia::handle(pool, args).await?,
        "cuba_zafra" => zafra::handle(pool, args).await?,
        _ => {
            tracing::warn!(tool = %tool_name, "unknown tool");
            anyhow::bail!("Unknown tool: {tool_name}")
        }
    };

    let elapsed_ms = start.elapsed().as_millis();
    tracing::info!(tool = %tool_name, elapsed_ms = %elapsed_ms, "handler completed");

    // Wrap in MCP content format
    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&result)?
        }]
    }))
}

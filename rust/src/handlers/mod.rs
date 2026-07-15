use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub mod alarma;
pub mod alma;
pub mod archivo;
pub mod calibrar;
pub mod centinela;
pub mod contradiccion;
pub mod cronica;
pub mod decreto;
#[cfg(feature = "docs")]
pub mod docs;
pub mod eco;
pub mod expediente;
pub mod faro;
pub mod forget;
pub mod hipotesis;
pub mod ingesta;
pub mod jornada;
pub mod juez;
pub mod pizarra;
pub mod pre_compact;
pub mod proyecto;
pub mod puente;
pub mod receta;
pub mod reflexion;
pub mod remedio;
pub mod sync;
pub mod tools;
pub mod vigia;
pub mod zafra;

#[tracing::instrument(skip(pool, args), fields(tool = %tool_name))]
pub async fn dispatch(pool: &PgPool, tool_name: &str, args: Value) -> Result<Value> {
    let start = std::time::Instant::now();

    let dispatch_result: Result<Value> = async {
        match tool_name {
            "cuba_alma" => alma::handle(pool, args).await,
            "cuba_cronica" => cronica::handle(pool, args).await,
            "cuba_faro" => faro::handle(pool, args).await,
            "cuba_receta" => receta::handle(pool, args).await,
            "cuba_tools" => tools::handle_tools(pool, args).await,
            "cuba_call" => tools::handle_call(pool, args).await,
            "cuba_forget" => forget::handle(pool, args).await,
            "cuba_hipotesis" => hipotesis::handle(pool, args).await,
            "cuba_puente" => puente::handle(pool, args).await,
            "cuba_reflexion" => reflexion::handle(pool, args).await,
            "cuba_eco" => eco::handle(pool, args).await,
            "cuba_alarma" => alarma::handle(pool, args).await,
            "cuba_remedio" => remedio::handle(pool, args).await,
            "cuba_expediente" => expediente::handle(pool, args).await,
            "cuba_jornada" => jornada::handle(pool, args).await,
            "cuba_decreto" => decreto::handle(pool, args).await,
            "cuba_vigia" => vigia::handle(pool, args).await,
            "cuba_zafra" => zafra::handle(pool, args).await,
            "cuba_centinela" => centinela::handle(pool, args).await,
            "cuba_contradiccion" => contradiccion::handle(pool, args).await,
            "cuba_calibrar" => calibrar::handle(pool, args).await,
            "cuba_ingesta" => ingesta::handle(pool, args).await,
            "cuba_proyecto" => proyecto::handle(pool, args).await,
            "cuba_pre_compact" => pre_compact::handle(pool, args).await,
            "cuba_sync" => sync::handle(pool, args).await,
            "cuba_juez" => juez::handle(pool, args).await,
            "cuba_pizarra" => pizarra::handle(pool, args).await,
            "cuba_archivo" => archivo::handle(pool, args).await,
            #[cfg(feature = "docs")]
            "cuba_docs" => docs::handle(&args).await,
            _ => {
                tracing::warn!(tool = %tool_name, "unknown tool");
                anyhow::bail!("Unknown tool: {tool_name}")
            }
        }
    }
    .await;

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_millis();
    let outcome = if dispatch_result.is_ok() {
        "ok"
    } else {
        "error"
    };
    crate::observability::record_handler(tool_name, outcome, elapsed.as_secs_f64());
    tracing::info!(tool = %tool_name, elapsed_ms = %elapsed_ms, outcome = %outcome, "handler completed");

    let result = dispatch_result?;

    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&result)?
        }]
    }))
}

pub fn is_known_tool(name: &str) -> bool {
    #[cfg(feature = "docs")]
    if name == "cuba_docs" {
        return docs::enabled();
    }
    matches!(
        name,
        "cuba_alma"
            | "cuba_cronica"
            | "cuba_faro"
            | "cuba_forget"
            | "cuba_hipotesis"
            | "cuba_puente"
            | "cuba_reflexion"
            | "cuba_eco"
            | "cuba_alarma"
            | "cuba_remedio"
            | "cuba_expediente"
            | "cuba_jornada"
            | "cuba_decreto"
            | "cuba_vigia"
            | "cuba_zafra"
            | "cuba_contradiccion"
            | "cuba_centinela"
            | "cuba_calibrar"
            | "cuba_ingesta"
            | "cuba_proyecto"
            | "cuba_pre_compact"
            | "cuba_sync"
            | "cuba_archivo"
            | "cuba_pizarra"
            | "cuba_juez"
            | "cuba_tools"
            | "cuba_call"
            | "cuba_receta"
    )
}

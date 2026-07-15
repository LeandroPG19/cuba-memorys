use anyhow::Result;
use ort::session::builder::SessionBuilder;

pub fn configure(builder: SessionBuilder) -> Result<SessionBuilder> {
    #[allow(unused_mut)]
    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();

    #[cfg(feature = "cuda")]
    providers.push(ort::ep::CUDA::default().build());

    #[cfg(feature = "directml")]
    providers.push(ort::ep::DirectML::default().build());

    if providers.is_empty() {
        return Ok(builder);
    }

    builder
        .with_execution_providers(providers)
        .map_err(|e| anyhow::anyhow!("registrando execution providers GPU: {e}"))
}

pub fn active_provider() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        return "cuda (cae a directml/cpu si no hay NVIDIA)";
    }
    #[cfg(all(not(feature = "cuda"), feature = "directml"))]
    {
        return "directml (cae a cpu si no hay GPU)";
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "directml")))]
    {
        "cpu (compilado sin soporte GPU)"
    }
}

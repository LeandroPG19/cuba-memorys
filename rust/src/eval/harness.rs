use crate::eval::datasets::EvaluationSample;

pub struct BenchmarkHarness {
    dataset: Vec<EvaluationSample>,
}

impl BenchmarkHarness {
    pub fn new(dataset: Vec<EvaluationSample>) -> Self {
        Self { dataset }
    }

    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!(
            samples = self.dataset.len(),
            "eval benchmark harness (LOCOMO placeholder)"
        );
        Ok(())
    }
}

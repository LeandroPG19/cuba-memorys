use std::io;

#[derive(Debug, Clone)]
pub struct EvaluationSample {
    pub query: String,
    pub expected_answer: String,
    pub context: Option<String>,
}

pub fn load_locomo_dataset(_path: &str) -> Result<Vec<EvaluationSample>, io::Error> {
    Ok(vec![])
}

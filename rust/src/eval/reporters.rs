pub fn generate_json_report(results: &serde_json::Value) -> String {
    serde_json::to_string_pretty(results).unwrap_or_default()
}

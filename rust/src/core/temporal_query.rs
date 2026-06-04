use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSpan {
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub reference_date: DateTime<Utc>,
}

pub fn parse_relative_time(expr: &str, reference: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let expr = expr.to_lowercase();

    if expr.contains("ayer") || expr.contains("yesterday") {
        return Some(reference - Duration::days(1));
    }
    if expr.contains("hoy") || expr.contains("today") {
        return Some(reference);
    }
    if expr.contains("mañana") || expr.contains("tomorrow") {
        return Some(reference + Duration::days(1));
    }

    if let Some((amount, unit)) = parse_last_n_units(&expr) {
        let duration = match unit.as_str() {
            "day" | "days" | "dia" | "dias" => Duration::days(amount),
            "week" | "weeks" => Duration::weeks(amount),
            "month" | "months" | "mes" | "meses" => Duration::days(amount * 30),
            "year" | "years" | "ano" | "anos" => Duration::days(amount * 365),
            _ => return None,
        };
        return Some(reference - duration);
    }

    None
}

fn parse_last_n_units(expr: &str) -> Option<(i64, String)> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    for window in parts.windows(3) {
        if (window[0] == "last" || window[0] == "ultimo" || window[0] == "ultimos")
            && let Ok(amount) = window[1].parse::<i64>()
        {
            return Some((amount, window[2].trim_end_matches('s').to_string()));
        }
    }
    None
}

pub fn calculate_temporal_score(valid_from: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
    let age = now.signed_duration_since(valid_from).num_seconds().max(0) as f32;
    let decay_rate = 0.00001_f32;
    (-decay_rate * age).exp()
}

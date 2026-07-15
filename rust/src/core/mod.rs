pub mod bitemporal;
pub mod entity_linking;
pub mod temporal_query;

pub fn bitemporal_enabled() -> bool {
    bitemporal_enabled_from_env(std::env::var("CUBA_BITEMPORAL").ok())
}

fn bitemporal_enabled_from_env(raw: Option<String>) -> bool {
    match raw.map(|v| v.to_ascii_lowercase()) {
        None => true,
        Some(v) if matches!(v.as_str(), "0" | "false" | "no" | "off") => false,
        Some(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitemporal_defaults_on() {
        assert!(bitemporal_enabled_from_env(None));
        assert!(!bitemporal_enabled_from_env(Some("0".into())));
        assert!(bitemporal_enabled_from_env(Some("1".into())));
    }
}

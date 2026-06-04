pub mod bitemporal;
pub mod entity_linking;
pub mod temporal_query;

pub fn bitemporal_enabled() -> bool {
    std::env::var("CUBA_BITEMPORAL")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
}

//! Search module — hybrid search with RRF, caching, and confidence.
//!
//! V0.9 additions:
//! - `mmr` — Maximal Marginal Relevance diversification (Carbonell-Goldstein 1998)
//! - `ood` — Out-of-distribution detection via Mahalanobis (Lee NeurIPS 2018)
//! - `budget` — exact token counting via tiktoken cl100k_base (replaces 4 chars/tok heuristic)

pub mod calibrate;
pub mod bm25;
pub mod budget;
pub mod cache;
pub mod confidence;
pub mod mmr;
pub mod ood;
pub mod ood_cache;
pub mod rerank;
pub mod rrf;

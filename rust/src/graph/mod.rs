//! Graph module — PageRank, community detection, betweenness centrality.
//!
//! V0.9 additions:
//! - `closeness` — Bavelas 1950 closeness + Boldi-Vigna 2014 harmonic (robust
//!   for disconnected graphs).
//! - `kcore` — Batagelj-Zaversnik 2003 k-core decomposition for "structural
//!   backbone" detection (used by `cuba_forget` to refuse deleting load-bearing
//!   nodes).

pub mod activation;
pub mod centrality;
pub mod closeness;
pub mod community;
pub mod energy;
pub mod kcore;
pub mod pagerank;

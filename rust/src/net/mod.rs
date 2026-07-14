//! Everything cuba-memorys does over the network — which, without the `docs` feature,
//! is nothing at all.
//!
//! A memory server has no business making outbound requests, and until v0.13 this one
//! made none. `cuba_docs` changes that, so the capability is fenced off here, behind a
//! Cargo feature, behind an SSRF guard that assumes the URL is hostile.

pub mod fetch;
pub mod guard;

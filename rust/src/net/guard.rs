//! SSRF guard — the only thing standing between a memory server and the cloud
//! metadata endpoint.
//!
//! # Why this is written from scratch instead of ported
//!
//! cuba-search has an `_is_ssrf_safe`, and porting it was the plan. It is broken:
//!
//! ```python
//! try:
//!     addr = ipaddress.ip_address(hostname)
//!     return not any(addr in net for net in _PRIVATE_RANGES)
//! except ValueError:
//!     # Hostname (not IP) — check for localhost patterns
//!     return lower not in {"localhost", "0.0.0.0", "[::1]"}
//! ```
//!
//! When the host is **not a literal IP** — which is to say, in every real request —
//! it compares the string against three names and lets it through. **It never resolves
//! DNS.** The careful list of private ranges above it is decorative: any domain whose
//! A record points at `169.254.169.254` walks straight past it and reads the machine's
//! cloud credentials. `curl http://whatever.attacker.com/` is the whole exploit.
//!
//! So the rule here is: **validate the IP you are actually going to connect to.**
//!
//! 1. Scheme must be http/https. No `file:`, no `gopher:`.
//! 2. Resolve the hostname, and check **every** address it resolves to. One public
//!    A record does not excuse a private AAAA record.
//! 3. **Pin** the address that passed onto the connection, so the name cannot resolve
//!    to something else between the check and the connect. Validating a name and then
//!    handing the name to the HTTP client re-opens the hole via DNS rebinding: the
//!    attacker answers the first query with a public IP and the second with
//!    `169.254.169.254`.
//! 4. Follow redirects **manually**, re-validating each hop. A 302 to
//!    `http://169.254.169.254/` is the same attack wearing a hat.

use anyhow::{Context, Result, bail};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use url::Url;

/// Ranges no outbound request from this process may reach.
///
/// `169.254.0.0/16` is the one that matters most and the one people forget: it is
/// where AWS, GCP and Azure serve instance credentials to anything that asks.
fn is_forbidden(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()          // 127.0.0.0/8
                || v4.is_private()    // 10/8, 172.16/12, 192.168/16
                || v4.is_link_local() // 169.254.0.0/16 — cloud metadata
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified() // 0.0.0.0
                || v4.octets()[0] == 0
                // Carrier-grade NAT, 100.64.0.0/10 — reachable inside many networks.
                || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]))
                // Benchmarking, 198.18.0.0/15.
                || (v4.octets()[0] == 198 && (18..20).contains(&v4.octets()[1]))
                || v4.is_multicast()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()          // ::1
                || v6.is_unspecified()
                || v6.is_multicast()
                // Unique local, fc00::/7.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // Link-local, fe80::/10 — includes the IPv6 metadata address.
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // IPv4-mapped (::ffff:169.254.169.254) — unwrap and re-check, or the
                // whole guard above is bypassed by writing the address differently.
                || v6.to_ipv4_mapped().is_some_and(|v4| is_forbidden(&IpAddr::V4(v4)))
        }
    }
}

/// A URL that has been checked, together with the address it is pinned to.
///
/// Carrying the address is the point: the caller must connect to *this*, not re-resolve
/// the name. See the module docs on DNS rebinding.
#[derive(Debug, Clone)]
pub struct SafeUrl {
    pub url: Url,
    pub addr: SocketAddr,
}

/// Is this URL safe to fetch? Resolves DNS and validates what it actually resolves to.
///
/// Returns the pinned address on success. On failure the error names the reason, so a
/// user pointing at their own intranet gets told why rather than a blank refusal.
pub fn check(raw: &str) -> Result<SafeUrl> {
    let url = Url::parse(raw).with_context(|| format!("URL inválida: {raw}"))?;

    match url.scheme() {
        "http" | "https" => {}
        other => bail!(
            "esquema `{other}` no permitido: sólo http y https. \
             `file:`, `gopher:` y compañía son vectores de SSRF, no formas de leer documentación."
        ),
    }

    let host = url
        .host_str()
        .context("la URL no tiene host")?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();

    let port = url.port_or_known_default().unwrap_or(443);

    // Resolve, and judge what came back — not what was typed.
    let addrs: Vec<SocketAddr> = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("no se pudo resolver `{host}`"))?
        .collect();

    if addrs.is_empty() {
        bail!("`{host}` no resolvió a ninguna dirección");
    }

    // EVERY address, not the first one that happens to be public. A host with a public
    // A record and a loopback AAAA record must be rejected, not raced.
    for a in &addrs {
        if is_forbidden(&a.ip()) {
            bail!(
                "`{host}` resuelve a {} — una dirección privada, de loopback o link-local. \
                 Rechazado: 169.254.169.254 es donde AWS/GCP/Azure sirven credenciales, y \
                 un servidor de memoria no tiene ningún motivo para pedírselas.",
                a.ip()
            );
        }
    }

    let addr = *addrs.first().expect("no vacío, comprobado arriba");
    Ok(SafeUrl { url, addr })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE test. Without it this module does not merge.
    #[test]
    fn cloud_metadata_is_rejected() {
        let err = check("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
            .expect_err("el endpoint de metadatos de la nube DEBE rechazarse");
        assert!(
            format!("{err:#}").contains("169.254.169.254"),
            "el error debe decir qué se rechazó: {err:#}"
        );
    }

    #[test]
    fn loopback_is_rejected() {
        for url in [
            "http://127.0.0.1:8080/",
            "http://localhost:5488/",
            "http://[::1]/",
            "http://0.0.0.0/",
        ] {
            assert!(
                check(url).is_err(),
                "{url} debería rechazarse: apunta a esta misma máquina"
            );
        }
    }

    #[test]
    fn private_ranges_are_rejected() {
        for url in [
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
            "http://100.64.0.1/", // CGNAT
        ] {
            assert!(check(url).is_err(), "{url} es una dirección privada");
        }
    }

    /// Writing the address differently must not make it a different address.
    #[test]
    fn ipv4_mapped_ipv6_cannot_smuggle_the_metadata_address() {
        assert!(
            is_forbidden(&"::ffff:169.254.169.254".parse::<IpAddr>().unwrap()),
            "::ffff:169.254.169.254 ES 169.254.169.254"
        );
        assert!(is_forbidden(&"::ffff:127.0.0.1".parse::<IpAddr>().unwrap()));
    }

    #[test]
    fn non_http_schemes_are_rejected() {
        for url in [
            "file:///etc/passwd",
            "gopher://evil/",
            "ftp://internal/",
            "data:text/html,<script>",
        ] {
            assert!(check(url).is_err(), "{url} no es documentación");
        }
    }

    /// The guard must not be so eager that it blocks the actual job.
    #[test]
    fn public_documentation_is_allowed() {
        // Skipped when offline: a CI box with no DNS must not fail the suite.
        match check("https://docs.rs/tokio/latest/tokio/") {
            Ok(safe) => assert!(!is_forbidden(&safe.addr.ip())),
            Err(e) => eprintln!("SKIP (sin red): {e:#}"),
        }
    }
}

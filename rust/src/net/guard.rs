use anyhow::{Context, Result, bail};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use url::Url;

fn is_forbidden(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || v4.octets()[0] == 0
                || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]))
                || (v4.octets()[0] == 198 && (18..20).contains(&v4.octets()[1]))
                || v4.is_multicast()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| is_forbidden(&IpAddr::V4(v4)))
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeUrl {
    pub url: Url,
    pub addr: SocketAddr,
}

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

    let addrs: Vec<SocketAddr> = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("no se pudo resolver `{host}`"))?
        .collect();

    if addrs.is_empty() {
        bail!("`{host}` no resolvió a ninguna dirección");
    }

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
            "http://100.64.0.1/",
        ] {
            assert!(check(url).is_err(), "{url} es una dirección privada");
        }
    }

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

    #[test]
    fn public_documentation_is_allowed() {
        match check("https://docs.rs/tokio/latest/tokio/") {
            Ok(safe) => assert!(!is_forbidden(&safe.addr.ip())),
            Err(e) => eprintln!("SKIP (sin red): {e:#}"),
        }
    }
}

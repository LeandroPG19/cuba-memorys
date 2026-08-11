use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

const MIN_TOKEN_CHARS: usize = 24;
const QUICK_TUNNEL_HOST: &str = "trycloudflare.com";
const URL_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

#[derive(Debug, PartialEq, Eq)]
pub enum TokenVerdict {
    Missing,
    TooShort(usize),
    Ok,
}

pub fn inspect_token(token: Option<&str>) -> TokenVerdict {
    match token.map(str::trim).filter(|t| !t.is_empty()) {
        None => TokenVerdict::Missing,
        Some(t) if t.chars().count() < MIN_TOKEN_CHARS => TokenVerdict::TooShort(t.chars().count()),
        Some(_) => TokenVerdict::Ok,
    }
}

pub fn suggest_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

pub fn extract_tunnel_url(line: &str) -> Option<String> {
    let start = line.find("https://")?;
    let rest = &line[start..];
    let end = rest
        .find(|c: char| c.is_whitespace() || c == '|' || c == '"')
        .unwrap_or(rest.len());
    let url = rest[..end].trim_end_matches(['.', ',']);
    url.contains(QUICK_TUNNEL_HOST).then(|| url.to_string())
}

pub fn client_config(public_url: &str, token: &str) -> String {
    let body = serde_json::json!({
        "mcpServers": {
            "cuba-memorys": {
                "type": "http",
                "url": format!("{public_url}/mcp"),
                "headers": {
                    "Authorization": format!("Bearer {token}"),
                    "Mcp-Client-Id": "claude-web"
                }
            }
        }
    });
    serde_json::to_string_pretty(&body).unwrap_or_default()
}

async fn daemon_is_up(addr: &str) -> bool {
    let url = format!("http://{addr}/health");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build();
    match client {
        Ok(c) => matches!(c.get(&url).send().await, Ok(r) if r.status().is_success()),
        Err(_) => false,
    }
}

fn refuse_without_token(verdict: &TokenVerdict) -> Option<String> {
    let detail = match verdict {
        TokenVerdict::Ok => return None,
        TokenVerdict::Missing => "CUBA_HTTP_TOKEN no está definido".to_string(),
        TokenVerdict::TooShort(n) => {
            format!("CUBA_HTTP_TOKEN tiene {n} caracteres; hacen falta {MIN_TOKEN_CHARS}")
        }
    };
    Some(format!(
        "{detail}.\n\n\
         Un túnel publica el daemon en internet, y el daemon sirve el grafo entero \
         sin autenticación por defecto. Sin un token, cualquiera con la URL lee y \
         escribe tu memoria.\n\n\
         Generá uno y arrancá el daemon con él:\n\n  \
         export CUBA_HTTP_TOKEN={}\n  \
         cuba-memorys serve\n",
        suggest_token()
    ))
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut addr = crate::http::bind_addr();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--addr" => addr = it.next().cloned().context("--addr needs an address")?,
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys tunnel [--addr HOST:PORT]\n\n\
                     Publishes the running daemon through a Cloudflare quick tunnel and \
                     prints the client configuration to paste into Claude on the web.\n\n\
                     Requires CUBA_HTTP_TOKEN, because the daemon serves the whole graph \
                     with no authentication of its own."
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown tunnel flag: {other} (try --help)"),
        }
    }

    let token = std::env::var("CUBA_HTTP_TOKEN").ok();
    if let Some(refusal) = refuse_without_token(&inspect_token(token.as_deref())) {
        anyhow::bail!(refusal);
    }
    let token = token.unwrap_or_default();

    if !daemon_is_up(&addr).await {
        anyhow::bail!(
            "no hay daemon respondiendo en http://{addr}/health.\n\
             Arrancalo primero con el mismo token:\n\n  \
             CUBA_HTTP_TOKEN=$CUBA_HTTP_TOKEN cuba-memorys serve {addr}\n"
        );
    }

    let mut child = tokio::process::Command::new("cloudflared")
        .args(["tunnel", "--url", &format!("http://{addr}")])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context(
            "no pude ejecutar cloudflared — instalalo desde \
             https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
        )?;

    let stderr = child.stderr.take().context("cloudflared sin stderr")?;
    let mut lines = BufReader::new(stderr).lines();

    let deadline = tokio::time::Instant::now() + URL_WAIT;
    let public_url = loop {
        let next = tokio::time::timeout_at(deadline, lines.next_line()).await;
        match next {
            Err(_) => anyhow::bail!("cloudflared no publicó ninguna URL en {URL_WAIT:?}"),
            Ok(Ok(Some(line))) => {
                if let Some(url) = extract_tunnel_url(&line) {
                    break url;
                }
                tracing::debug!(line, "cloudflared");
            }
            Ok(Ok(None)) => anyhow::bail!("cloudflared terminó sin publicar una URL"),
            Ok(Err(e)) => return Err(anyhow::Error::from(e).context("leyendo cloudflared")),
        }
    };

    println!("\n  túnel arriba: {public_url}");
    println!("  MCP endpoint: {public_url}/mcp\n");
    println!("  Configuración para el cliente:\n");
    println!("{}\n", client_config(&public_url, &token));
    println!("  La URL vive mientras este proceso siga corriendo. Ctrl-C la cierra.\n");

    tokio::select! {
        status = child.wait() => {
            let status = status.context("esperando a cloudflared")?;
            anyhow::bail!("cloudflared terminó: {status}");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("  cerrando el túnel");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tunnel_without_a_token_is_refused() {
        assert_eq!(inspect_token(None), TokenVerdict::Missing);
        assert_eq!(inspect_token(Some("")), TokenVerdict::Missing);
        assert_eq!(inspect_token(Some("   ")), TokenVerdict::Missing);

        let refusal = refuse_without_token(&TokenVerdict::Missing).expect("must refuse");
        assert!(
            refusal.contains("sin autenticación"),
            "the refusal has to say why, or it reads as a bug: {refusal}"
        );
        assert!(refusal.contains("export CUBA_HTTP_TOKEN="));
    }

    #[test]
    fn a_short_token_is_refused_with_its_length() {
        assert_eq!(inspect_token(Some("hunter2")), TokenVerdict::TooShort(7));
        assert_eq!(
            inspect_token(Some(&"a".repeat(23))),
            TokenVerdict::TooShort(23)
        );
        assert_eq!(inspect_token(Some(&"a".repeat(24))), TokenVerdict::Ok);

        let refusal = refuse_without_token(&TokenVerdict::TooShort(7)).expect("must refuse");
        assert!(refusal.contains("7 caracteres"), "{refusal}");
    }

    #[test]
    fn a_usable_token_is_not_refused() {
        assert!(refuse_without_token(&TokenVerdict::Ok).is_none());
    }

    #[test]
    fn a_suggested_token_passes_its_own_check() {
        let a = suggest_token();
        let b = suggest_token();

        assert_eq!(inspect_token(Some(&a)), TokenVerdict::Ok);
        assert_ne!(a, b, "two tunnels on one machine must not share a token");
        assert!(
            a.chars().all(|c| c.is_ascii_alphanumeric()),
            "it goes into a shell export and an HTTP header value, so anything a shell \
             or a header parser treats specially would break the copy-paste: {a}"
        );
        assert!(a.len() >= MIN_TOKEN_CHARS);
    }

    #[test]
    fn the_url_is_read_out_of_the_line_cloudflared_actually_prints() {
        let banner = "2026-08-11T14:00:00Z INF |  https://tidy-pear-mango-42.trycloudflare.com  |";
        assert_eq!(
            extract_tunnel_url(banner).as_deref(),
            Some("https://tidy-pear-mango-42.trycloudflare.com")
        );

        let plain =
            "INF Your quick Tunnel has been created! Visit it at https://a-b-c.trycloudflare.com";
        assert_eq!(
            extract_tunnel_url(plain).as_deref(),
            Some("https://a-b-c.trycloudflare.com")
        );
    }

    #[test]
    fn a_line_that_is_not_the_tunnel_url_is_ignored() {
        assert_eq!(extract_tunnel_url("INF starting tunnel"), None);
        assert_eq!(
            extract_tunnel_url("see https://developers.cloudflare.com/downloads/"),
            None,
            "the install hint carries an https URL too, and taking it would print a \
             documentation link as if it were the tunnel"
        );
        assert_eq!(extract_tunnel_url("https://example.com"), None);
    }

    #[test]
    fn the_printed_config_carries_the_token_and_the_mcp_path() {
        let cfg = client_config("https://a-b-c.trycloudflare.com", "s3cret-token-value-here");
        let parsed: serde_json::Value = serde_json::from_str(&cfg).expect("valid JSON");

        let server = &parsed["mcpServers"]["cuba-memorys"];
        assert_eq!(server["url"], "https://a-b-c.trycloudflare.com/mcp");
        assert_eq!(server["type"], "http");
        assert_eq!(
            server["headers"]["Authorization"], "Bearer s3cret-token-value-here",
            "without the bearer every call comes back 401 and the tunnel looks broken"
        );
        assert!(
            server["headers"]["Mcp-Client-Id"].is_string(),
            "each client needs its own id or one session leaks into the next"
        );
    }
}

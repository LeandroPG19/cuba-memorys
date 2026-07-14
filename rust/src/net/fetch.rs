//! Fetching a URL the paranoid way.
//!
//! Every guarantee in [`crate::net::guard`] is worthless if the HTTP client is then
//! handed the *hostname* and left to resolve it again. This module closes the two gaps
//! that reopens:
//!
//! - **DNS rebinding.** The address that passed validation is pinned onto the
//!   connection. The name cannot resolve to `169.254.169.254` on the second lookup,
//!   because there is no second lookup.
//! - **Redirects.** `reqwest` will happily follow a `302` to anywhere. Redirects are
//!   disabled and followed by hand, re-validating every hop — a redirect to the
//!   metadata endpoint is the same attack with an extra step.

use anyhow::{Context, Result, bail};
use std::time::Duration;

use super::guard;

/// Longest we will wait. A documentation page that cannot answer in ten seconds is not
/// worth blocking a memory lookup on.
const TIMEOUT: Duration = Duration::from_secs(10);

/// Most bytes we will read from a response body.
///
/// Enforced while streaming, not after: `Content-Length` is a claim by the server, and
/// a hostile one can send a gigabyte while promising a kilobyte. The only number that
/// cannot lie is the count of bytes actually received.
const MAX_BYTES: usize = 2 * 1024 * 1024;

/// How many redirects to follow before concluding we are being led somewhere.
const MAX_HOPS: usize = 5;

/// Fetch a URL, validating the destination at every hop.
pub async fn get(raw: &str) -> Result<String> {
    let mut target = raw.to_string();

    for hop in 0..=MAX_HOPS {
        let safe = guard::check(&target)?;
        let host = safe
            .url
            .host_str()
            .context("URL sin host tras la validación")?
            .to_string();

        // Pin the validated address. `reqwest` will connect to exactly this, and will
        // not ask DNS a second question it might get a different answer to.
        let client = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, safe.addr)
            .user_agent(concat!("cuba-memorys/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("construyendo el cliente HTTP")?;

        let resp = client
            .get(safe.url.clone())
            .send()
            .await
            .with_context(|| format!("pidiendo {}", safe.url))?;

        let status = resp.status();

        if status.is_redirection() {
            if hop == MAX_HOPS {
                bail!("demasiadas redirecciones ({MAX_HOPS}) desde {raw}");
            }
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .context("redirección sin cabecera Location")?;

            // Relative redirects are legal, so resolve against the current URL — and
            // then send the result straight back through the guard. This is the hop
            // that a naive client would follow to the metadata endpoint.
            target = safe
                .url
                .join(location)
                .with_context(|| format!("Location inválida: {location}"))?
                .to_string();
            continue;
        }

        if !status.is_success() {
            bail!("{} respondió {}", safe.url, status);
        }

        return read_capped(resp).await;
    }

    bail!("demasiadas redirecciones ({MAX_HOPS}) desde {raw}")
}

/// Read the body, stopping at [`MAX_BYTES`] regardless of what the server claimed.
async fn read_capped(mut resp: reqwest::Response) -> Result<String> {
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = resp.chunk().await.context("leyendo el cuerpo")? {
        if buf.len() + chunk.len() > MAX_BYTES {
            let room = MAX_BYTES - buf.len();
            buf.extend_from_slice(&chunk[..room]);
            tracing::warn!(
                limit = MAX_BYTES,
                "respuesta truncada: el servidor mandó más de lo permitido"
            );
            break;
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// HTML to something a model can read, without the navigation, scripts and cookie
/// banners that make up most of a documentation page's bytes.
pub fn html_to_text(html: &str, width: usize) -> String {
    html2text::from_read(html.as_bytes(), width)
}

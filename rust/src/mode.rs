use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Local,
    Red,
    Completo,
}

pub fn active() -> Mode {
    match env::var("CUBA_MODE")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        .as_str()
    {
        "red" | "cloud" | "nube" => Mode::Red,
        "completo" | "full" | "complete" => Mode::Completo,
        _ => Mode::Local,
    }
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Local => "local",
            Mode::Red => "red",
            Mode::Completo => "completo",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Mode::Local => "BD local (Docker) · modelos locales · sin red saliente",
            Mode::Red => {
                "BD compartida en la nube (TLS) · procedencia por nodo · sync entre máquinas"
            }
            Mode::Completo => "todo activado · reranker (GPU si hay) · cuba_docs",
        }
    }

    pub fn is_cloud(self) -> bool {
        match self {
            Mode::Red => true,
            Mode::Completo => env::var("DATABASE_URL")
                .map(|u| !u.is_empty())
                .unwrap_or(false),
            Mode::Local => false,
        }
    }

    pub fn docs_default(self) -> bool {
        matches!(self, Mode::Completo)
    }

    pub fn rerank_default(self) -> bool {
        matches!(self, Mode::Completo)
    }
}

pub fn env_toggle(name: &str) -> Option<bool> {
    let v = env::var(name).ok()?;
    let v = v.trim().to_lowercase();
    if v.is_empty() {
        return None;
    }
    Some(matches!(
        v.as_str(),
        "1" | "true" | "on" | "yes" | "sí" | "si"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completo_turns_capabilities_on_others_leave_them_off() {
        assert!(Mode::Completo.docs_default());
        assert!(Mode::Completo.rerank_default());
        assert!(!Mode::Local.docs_default());
        assert!(!Mode::Local.rerank_default());
        assert!(!Mode::Red.rerank_default());
        assert!(Mode::Red.is_cloud());
        assert!(!Mode::Local.is_cloud());
    }
}

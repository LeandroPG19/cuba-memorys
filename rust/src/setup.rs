use std::process::Command;
use std::time::Duration;

const CONTAINER_NAME: &str = "cuba-memorys-db";
const PG_IMAGE: &str = "pgvector/pgvector:pg18";
const PG_USER: &str = "cuba";
const PG_PASSWORD: &str = "memorys2026";
const PG_DB: &str = "brain";
const PG_PORT: u16 = 5488;

pub async fn resolve_database_url() -> String {
    if let Ok(url) = std::env::var("DATABASE_URL")
        && !url.is_empty()
    {
        return url;
    }

    if crate::mode::active().is_cloud() {
        log("CUBA_MODE=red (nube) pero DATABASE_URL no está seteada.");
        log("El modo red usa una base compartida en la nube — poné la URL de tu");
        log("Postgres gestionado (Supabase/Neon/…), con TLS:");
        log("  export DATABASE_URL=\"postgresql://user:pass@host/db?sslmode=require\"");
        std::process::exit(1);
    }
    if matches!(get_container_state(), ContainerState::Running) {
        return build_url();
    }
    if matches!(get_container_state(), ContainerState::Unknown) && port_answers() {
        return build_url();
    }

    log("DATABASE_URL not set. Attempting automatic PostgreSQL setup...");

    if !is_docker_available() {
        log("");
        log("=== Cuba-Memorys Setup Required ===");
        log("");
        log("PostgreSQL with pgvector is needed but DATABASE_URL is not set");
        log("and Docker is not available for automatic setup.");
        log("");
        log("Option 1: Install Docker and restart (recommended)");
        log("  https://docs.docker.com/get-docker/");
        log("");
        log("Option 2: Set up PostgreSQL manually:");
        log("  1. Install PostgreSQL 15+ with pgvector extension");
        log("  2. Create a database: CREATE DATABASE brain;");
        log("  3. Set the environment variable:");
        log("     export DATABASE_URL=\"postgresql://user:pass@localhost:5432/brain\"");
        log("");
        std::process::exit(1);
    }

    match get_container_state() {
        ContainerState::Running => {
            log("PostgreSQL container 'cuba-memorys-db' is already running.");
            return build_url();
        }
        ContainerState::Stopped => {
            log("Starting existing PostgreSQL container 'cuba-memorys-db'...");
            docker_start();
        }
        ContainerState::Unknown => {
            if port_answers() {
                log("Docker no responde, pero PostgreSQL contesta en el puerto — se usa.");
                return build_url();
            }
            log("ERROR: el daemon de Docker no responde (docker ps falló).");
            log("En Windows esto suele ser WSL2 sin arrancar: revisá que la");
            log("'Plataforma de máquina virtual' esté activada y Docker Desktop corriendo.");
            log("Diagnóstico: docker info");
            std::process::exit(1);
        }
        ContainerState::NotFound => {
            log("");
            log("=== Cuba-Memorys Automatic Setup ===");
            log("");
            log("This will create a local PostgreSQL database for AI memory storage.");
            log("A Docker container 'cuba-memorys-db' will be created with:");
            log(&format!("  - Image:    {PG_IMAGE}"));
            log(&format!(
                "  - Port:     {PG_PORT} (mapped to container 5432)"
            ));
            log(&format!("  - Database: {PG_DB}"));
            log(&format!("  - User:     {PG_USER}"));
            log("  - Volume:   cuba_memorys_data (persistent across restarts)");
            log("");
            log("Creating and starting PostgreSQL container...");
            docker_create_and_start();
        }
    }

    log("Waiting for PostgreSQL to accept connections...");
    if wait_for_healthy(Duration::from_secs(60)).await {
        tokio::time::sleep(Duration::from_secs(2)).await;
        log("PostgreSQL is ready.");
        log(&format!("DATABASE_URL: {}", build_url()));
        log("");
    } else {
        log("ERROR: PostgreSQL did not become ready within 60 seconds.");
        log("Check Docker logs: docker logs cuba-memorys-db");
        std::process::exit(1);
    }

    build_url()
}

fn log(msg: &str) {
    eprintln!("[cuba-memorys] {msg}");
}

fn build_url() -> String {
    format!("postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{PG_PORT}/{PG_DB}")
}

fn is_docker_available() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn port_answers() -> bool {
    use std::net::TcpStream;
    use std::time::Duration;
    let addr = format!("127.0.0.1:{PG_PORT}");
    TcpStream::connect_timeout(
        &addr.parse().expect("host:port literal is valid"),
        Duration::from_millis(500),
    )
    .is_ok()
}

fn name_already_in_use() -> bool {
    matches!(
        get_container_state(),
        ContainerState::Running | ContainerState::Stopped
    )
}

enum ContainerState {
    Running,
    Stopped,
    NotFound,
    Unknown,
}

fn get_container_state() -> ContainerState {
    let output = Command::new("docker")
        .args([
            "ps",
            "-a",
            "--filter",
            &format!("name=^{CONTAINER_NAME}$"),
            "--format",
            "{{.Status}}",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => parse_container_status(&String::from_utf8_lossy(&o.stdout)),
        _ => ContainerState::Unknown,
    }
}

fn parse_container_status(stdout: &str) -> ContainerState {
    let status = stdout.trim();
    if status.is_empty() {
        ContainerState::NotFound
    } else if status.starts_with("Up") {
        ContainerState::Running
    } else {
        ContainerState::Stopped
    }
}

fn docker_start() {
    let status = Command::new("docker")
        .args(["start", CONTAINER_NAME])
        .status();

    if let Ok(s) = status
        && !s.success()
    {
        log("ERROR: Failed to start container. Run: docker start cuba-memorys-db");
        std::process::exit(1);
    }
}

fn docker_create_and_start() {
    let status = Command::new("docker")
        .args([
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-e",
            &format!("POSTGRES_USER={PG_USER}"),
            "-e",
            &format!("POSTGRES_PASSWORD={PG_PASSWORD}"),
            "-e",
            &format!("POSTGRES_DB={PG_DB}"),
            "-p",
            &format!("{PG_PORT}:5432"),
            "-v",
            "cuba_memorys_data:/var/lib/postgresql",
            "--health-cmd",
            &format!("pg_isready -U {PG_USER} -d {PG_DB}"),
            "--health-interval",
            "2s",
            "--health-timeout",
            "3s",
            "--health-retries",
            "15",
            "--restart",
            "unless-stopped",
            PG_IMAGE,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            log("Container created successfully.");
        }
        _ if name_already_in_use() => {
            log("El contenedor ya existía — se reutiliza en vez de recrearlo.");
            docker_start();
        }
        _ => {
            log("ERROR: Failed to create Docker container.");
            log("Make sure Docker is running: docker info");
            std::process::exit(1);
        }
    }
}

async fn wait_for_healthy(timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    let poll_interval = Duration::from_millis(500);

    while start.elapsed() < timeout {
        let ok = Command::new("docker")
            .args([
                "exec",
                CONTAINER_NAME,
                "pg_isready",
                "-U",
                PG_USER,
                "-d",
                PG_DB,
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if ok {
            return true;
        }

        tokio::time::sleep(poll_interval).await;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_status_is_read_correctly() {
        assert!(
            matches!(
                parse_container_status("Up 2 hours (healthy)"),
                ContainerState::Running
            ),
            "«Up …» es Running"
        );
        assert!(matches!(
            parse_container_status("Up 3 seconds"),
            ContainerState::Running
        ));
        assert!(
            matches!(
                parse_container_status("Exited (0) 5 minutes ago"),
                ContainerState::Stopped
            ),
            "«Exited …» es Stopped, no NotFound: existe, hay que arrancarlo"
        );
        assert!(matches!(
            parse_container_status("Created"),
            ContainerState::Stopped
        ));
        assert!(
            matches!(parse_container_status(""), ContainerState::NotFound),
            "sin fila, el nombre no existe"
        );
        assert!(
            matches!(parse_container_status("  \n"), ContainerState::NotFound),
            "solo espacios = ninguna fila"
        );
    }
}

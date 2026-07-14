//! Auto-setup: detect missing PostgreSQL and provision via Docker.
//!
//! When DATABASE_URL is not set:
//! 1. Check if Docker is available
//! 2. Check if cuba-memorys-db container exists
//! 3. If not, create and start it automatically
//! 4. Wait for health check
//! 5. Return the DATABASE_URL
//!
//! All messages go to stderr (stdout reserved for MCP JSON-RPC).

use std::process::Command;
use std::time::Duration;

/// Default container configuration (matches docker-compose.yml).
const CONTAINER_NAME: &str = "cuba-memorys-db";
const PG_IMAGE: &str = "pgvector/pgvector:pg18";
const PG_USER: &str = "cuba";
const PG_PASSWORD: &str = "memorys2026";
const PG_DB: &str = "brain";
const PG_PORT: u16 = 5488; // Non-standard to avoid conflicts with system PostgreSQL

/// Resolve DATABASE_URL: use env var if set, otherwise auto-provision Docker PostgreSQL.
pub async fn resolve_database_url() -> String {
    // 1. Check if user already set DATABASE_URL
    if let Ok(url) = std::env::var("DATABASE_URL")
        && !url.is_empty()
    {
        return url;
    }
    // Also check if the default container is already running (user may have forgotten to set env)
    if matches!(get_container_state(), ContainerState::Running) {
        return build_url();
    }
    // A running container the daemon simply could not report is still reachable on its
    // port. If the DB answers, use it rather than trying to provision over the top.
    if matches!(get_container_state(), ContainerState::Unknown) && port_answers() {
        return build_url();
    }

    log("DATABASE_URL not set. Attempting automatic PostgreSQL setup...");

    // 2. Check Docker availability
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

    // 3. Check if container already exists
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
            // The daemon is not answering `docker ps`. Creating a container now is the
            // one thing guaranteed to go wrong (it may already exist). If the DB port
            // answers, use it; otherwise tell the user the daemon is down rather than
            // colliding on the name.
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

    // 4. Wait for PostgreSQL to be ready
    log("Waiting for PostgreSQL to accept connections...");
    if wait_for_healthy(Duration::from_secs(60)).await {
        // Brief pause to ensure port mapping is fully established
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

/// Print a message to stderr (not stdout — MCP protocol uses stdout).
fn log(msg: &str) {
    eprintln!("[cuba-memorys] {msg}");
}

/// Build the DATABASE_URL from default credentials.
fn build_url() -> String {
    format!("postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{PG_PORT}/{PG_DB}")
}

/// Check if Docker CLI is available.
fn is_docker_available() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Does something answer on the Postgres port? A TCP connect is the ground truth when
/// `docker` cannot be trusted — a container the daemon won't report is still listening.
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

/// Is the container name already taken? Used to recover from a `docker run` that failed
/// on collision — a container exists, so start it instead of dying.
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
    /// The daemon did not answer. NOT the same as NotFound — this is the state that
    /// used to be silently misread as "create a new one", producing "name already in
    /// use" on Windows where the daemon is briefly unreachable while WSL2 comes up.
    Unknown,
}

/// Check the state of the cuba-memorys-db container.
///
/// `docker ps -a` with an anchored name filter, not `inspect --format
/// {{.State.Running}}`. The inspect form has two failure modes that both ended in a
/// bogus `docker run`: the Go template renders differently across shells, and — worse —
/// the old code collapsed EVERY non-success exit to `NotFound`, so a daemon that was
/// merely slow to answer looked like "no container", and the next step tried to create
/// one that already existed.
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
        // Command ran but errored, or could not spawn at all: the daemon is not
        // answering. Do not guess "NotFound" — that guess is what breaks.
        _ => ContainerState::Unknown,
    }
}

/// Read `docker ps --format {{.Status}}` output into a state. Pure, so the mapping that
/// once sent a running container down the "create a new one" path is unit-tested.
fn parse_container_status(stdout: &str) -> ContainerState {
    let status = stdout.trim();
    if status.is_empty() {
        ContainerState::NotFound // no row matched the name filter
    } else if status.starts_with("Up") {
        ContainerState::Running
    } else {
        ContainerState::Stopped // Exited, Created, Restarting…
    }
}

/// Start an existing stopped container.
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

/// Create and start a new PostgreSQL container with pgvector.
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
        // The safety net: if the name is already taken, the container exists and our
        // state check missed it. Do not die — start what is there. This is the last
        // line of defence against the "name already in use" the field report hit.
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

/// Wait for PostgreSQL to accept connections (poll pg_isready via docker exec).
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

    /// The mapping that broke in the field: a running container must never read as
    /// "not found", because NotFound is what triggers `docker run` and the collision.
    #[test]
    fn container_status_is_read_correctly() {
        assert!(
            matches!(parse_container_status("Up 2 hours (healthy)"), ContainerState::Running),
            "«Up …» es Running"
        );
        assert!(
            matches!(parse_container_status("Up 3 seconds"), ContainerState::Running)
        );
        assert!(
            matches!(parse_container_status("Exited (0) 5 minutes ago"), ContainerState::Stopped),
            "«Exited …» es Stopped, no NotFound: existe, hay que arrancarlo"
        );
        assert!(
            matches!(parse_container_status("Created"), ContainerState::Stopped)
        );
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

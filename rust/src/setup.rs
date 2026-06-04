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

enum ContainerState {
    Running,
    Stopped,
    NotFound,
}

/// Check the state of the cuba-memorys-db container.
fn get_container_state() -> ContainerState {
    let output = Command::new("docker")
        .args(["inspect", "--format", "{{.State.Running}}", CONTAINER_NAME])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let state = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if state == "true" {
                ContainerState::Running
            } else {
                ContainerState::Stopped
            }
        }
        _ => ContainerState::NotFound,
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

use uuid::Uuid;

fn unique_name(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string()[..8])
}

async fn pool() -> sqlx::PgPool {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    cuba_memorys::db::create_pool(&url)
        .await
        .expect("connect to test database")
}

async fn isolated_entity_with_notes(pool: &sqlx::PgPool, name: &str, notes: &[&str]) -> Uuid {
    let id: (Uuid,) = sqlx::query_as(
        "INSERT INTO brain_entities (name, entity_type) VALUES ($1, 'project') RETURNING id",
    )
    .bind(name)
    .fetch_one(pool)
    .await
    .expect("creating the entity");

    for note in notes {
        sqlx::query(
            "INSERT INTO brain_observations (entity_id, content, observation_type, source)
             VALUES ($1, $2, 'fact', 'agent')",
        )
        .bind(id.0)
        .bind(note)
        .execute(pool)
        .await
        .expect("adding an observation");
    }
    id.0
}

#[tokio::test]
#[ignore]
async fn an_isolated_entity_gets_wired_into_the_graph_from_its_own_notes() {
    if cuba_memorys::cognitive::judge::resolve_offline_llm().is_none() {
        eprintln!("skipping: no local LLM CLI on PATH");
        return;
    }
    let pool = pool().await;
    let name = unique_name("Orquestador");
    let id = isolated_entity_with_notes(
        &pool,
        &name,
        &[
            &format!("{name} está escrito en Rust y se despliega con Docker."),
            &format!("{name} guarda su estado en PostgreSQL."),
            &format!("El equipo decidió que {name} reemplace al cron antiguo."),
        ],
    )
    .await;

    let before: (i64,) =
        sqlx::query_as("SELECT count(*) FROM brain_relations WHERE from_entity=$1 OR to_entity=$1")
            .bind(id)
            .fetch_one(&pool)
            .await
            .expect("counting edges before");
    assert_eq!(before.0, 0, "the fixture must start isolated");

    let pending = cuba_memorys::handlers::ingesta::entities_awaiting_relation_scan(&pool, 50)
        .await
        .expect("listing candidates");
    assert!(
        pending.contains(&id),
        "an isolated entity holding notes must be queued for scanning"
    );

    let linked = cuba_memorys::handlers::ingesta::scan_entity_relations(&pool, id)
        .await
        .expect("scanning");
    assert!(
        linked > 0,
        "notes naming Rust, Docker and PostgreSQL must yield at least one relation"
    );

    let after: Vec<(String, String)> = sqlx::query_as(
        "SELECT eb.name, r.relation_type FROM brain_relations r
         JOIN brain_entities eb ON eb.id = r.to_entity
         WHERE r.from_entity = $1 AND r.provenance = 'inferred'",
    )
    .bind(id)
    .fetch_all(&pool)
    .await
    .expect("reading the new edges");
    assert!(
        !after.is_empty(),
        "the scan must leave edges tagged as LLM-inferred"
    );
    eprintln!("edges created for {name}: {after:?}");

    let scanned: (Option<chrono::DateTime<chrono::Utc>>,) =
        sqlx::query_as("SELECT relations_scanned_at FROM brain_entities WHERE id = $1")
            .bind(id)
            .fetch_one(&pool)
            .await
            .expect("reading the scan stamp");
    assert!(
        scanned.0.is_some(),
        "a scanned entity must be stamped so the daemon does not pay for it twice"
    );

    let requeued = cuba_memorys::handlers::ingesta::entities_awaiting_relation_scan(&pool, 50)
        .await
        .expect("listing candidates again");
    assert!(
        !requeued.contains(&id),
        "an entity that now has edges must drop out of the queue"
    );

    sqlx::query("DELETE FROM brain_entities WHERE name = $1")
        .bind(&name)
        .execute(&pool)
        .await
        .ok();
}

#[tokio::test]
#[ignore]
async fn a_scanned_entity_is_revisited_once_new_notes_arrive() {
    let pool = pool().await;
    let name = unique_name("Silencioso");
    let id = isolated_entity_with_notes(&pool, &name, &["Nota inicial sin relaciones."]).await;

    sqlx::query("UPDATE brain_entities SET relations_scanned_at = NOW() WHERE id = $1")
        .bind(id)
        .execute(&pool)
        .await
        .expect("stamping as scanned");

    let queue = cuba_memorys::handlers::ingesta::entities_awaiting_relation_scan(&pool, 200)
        .await
        .expect("listing candidates");
    assert!(
        !queue.contains(&id),
        "a freshly scanned entity must not be rescanned for free"
    );

    sqlx::query(
        "INSERT INTO brain_observations (entity_id, content, observation_type, source, created_at)
         VALUES ($1, 'Ahora sabemos que depende de Redis.', 'fact', 'agent', NOW() + INTERVAL '1 second')",
    )
    .bind(id)
    .execute(&pool)
    .await
    .expect("adding a later observation");

    let requeued = cuba_memorys::handlers::ingesta::entities_awaiting_relation_scan(&pool, 200)
        .await
        .expect("listing candidates after new notes");
    assert!(
        requeued.contains(&id),
        "new notes must put the entity back in the queue, otherwise the graph freezes"
    );

    sqlx::query("DELETE FROM brain_entities WHERE name = $1")
        .bind(&name)
        .execute(&pool)
        .await
        .ok();
}

#[test]
fn the_scan_prompt_carries_the_notes_and_the_known_entities() {
    let prompt = cuba_memorys::handlers::ingesta::build_relation_scan_prompt(
        "Orquestador",
        "project",
        &["Corre sobre Rust.".to_string()],
        &["Rust".to_string(), "Docker".to_string()],
    );
    assert!(prompt.contains("Orquestador"));
    assert!(prompt.contains("Corre sobre Rust."));
    assert!(
        prompt.contains("Rust, Docker"),
        "known entities must be offered so the LLM reuses names instead of coining new ones"
    );
    assert!(
        prompt.contains("empty list"),
        "the prompt must make 'no relations' an explicitly acceptable answer"
    );
}

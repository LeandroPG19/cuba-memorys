use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct EntityAlias {
    pub alias_id: Uuid,
    pub entity_id: Uuid,
    pub alias_text: String,
    pub language_code: String,
    pub frequency_count: i64,
}

#[derive(Debug, Clone)]
pub struct EntityResolution {
    pub entity_id: Uuid,
    pub confidence: f32,
    pub match_type: MatchType,
}

#[derive(Debug, Clone)]
pub enum MatchType {
    Exact,
    Fuzzy(f32),
}

pub async fn resolve_entity(
    pool: &PgPool,
    query: &str,
) -> Result<Option<EntityResolution>, sqlx::Error> {
    let exact = sqlx::query_as::<_, EntityAlias>(
        "SELECT alias_id, entity_id, alias_text, language_code, frequency_count
         FROM brain_entity_aliases WHERE alias_text = $1 LIMIT 1",
    )
    .bind(query)
    .fetch_optional(pool)
    .await?;

    if let Some(alias) = exact {
        return Ok(Some(EntityResolution {
            entity_id: alias.entity_id,
            confidence: 1.0,
            match_type: MatchType::Exact,
        }));
    }

    let fuzzy = sqlx::query(
        "SELECT entity_id, similarity(alias_text, $1)::float4 AS score
         FROM brain_entity_aliases
         WHERE similarity(alias_text, $1) > 0.4
         ORDER BY score DESC LIMIT 1",
    )
    .bind(query)
    .fetch_optional(pool)
    .await?;

    if let Some(row) = fuzzy {
        let entity_id: Uuid = row.get("entity_id");
        let score: f32 = row.get("score");
        return Ok(Some(EntityResolution {
            entity_id,
            confidence: score,
            match_type: MatchType::Fuzzy(score),
        }));
    }

    Ok(None)
}

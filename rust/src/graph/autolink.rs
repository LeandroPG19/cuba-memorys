use anyhow::{Context, Result};
use sqlx::{PgPool, Row};

pub const MIN_CO_SESSIONS: i64 = 2;

pub const DEFAULT_NPMI_THRESHOLD: f64 = 0.3;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub from_id: uuid::Uuid,
    pub to_id: uuid::Uuid,
    pub from_name: String,
    pub to_name: String,
    pub co_sessions: i64,
    pub npmi: f64,
}

pub fn npmi(co: f64, count_a: f64, count_b: f64, total: f64) -> Option<f64> {
    if total <= 0.0 || co <= 0.0 || count_a <= 0.0 || count_b <= 0.0 {
        return None;
    }
    let p_ab = co / total;
    let p_a = count_a / total;
    let p_b = count_b / total;
    if p_ab >= 1.0 {
        return None;
    }
    let pmi = (p_ab / (p_a * p_b)).ln();
    let out = pmi / -p_ab.ln();
    out.is_finite().then_some(out)
}

pub async fn candidates(pool: &PgPool, min_co: i64, threshold: f64) -> Result<Vec<Candidate>> {
    let rows = sqlx::query(
        "WITH entity_sessions AS (
             -- One row per (entity, session) it was written in.
             SELECT DISTINCT o.entity_id, o.session_id
             FROM brain_observations o
             WHERE o.session_id IS NOT NULL
               AND o.observation_type != 'superseded'
         ),
         totals AS (SELECT count(DISTINCT session_id)::float8 AS n FROM entity_sessions),
         counts AS (
             SELECT entity_id, count(*)::float8 AS c FROM entity_sessions GROUP BY entity_id
         ),
         pairs AS (
             -- a < b keeps each unordered pair once.
             SELECT a.entity_id AS a_id, b.entity_id AS b_id, count(*)::float8 AS ab
             FROM entity_sessions a
             JOIN entity_sessions b
               ON a.session_id = b.session_id AND a.entity_id < b.entity_id
             GROUP BY a.entity_id, b.entity_id
             HAVING count(*) >= $1
         )
         SELECT p.a_id, p.b_id, p.ab, ca.c AS c_a, cb.c AS c_b, t.n,
                ea.name AS a_name, eb.name AS b_name
         FROM pairs p
         JOIN counts ca ON ca.entity_id = p.a_id
         JOIN counts cb ON cb.entity_id = p.b_id
         JOIN brain_entities ea ON ea.id = p.a_id
         JOIN brain_entities eb ON eb.id = p.b_id
         CROSS JOIN totals t
         -- Never propose an edge that already exists, in either direction.
         WHERE NOT EXISTS (
             SELECT 1 FROM brain_relations r
             WHERE (r.from_entity = p.a_id AND r.to_entity = p.b_id)
                OR (r.from_entity = p.b_id AND r.to_entity = p.a_id)
         )",
    )
    .bind(min_co)
    .fetch_all(pool)
    .await
    .context("computing entity co-occurrence")?;

    let mut out = Vec::new();
    for r in &rows {
        let co: f64 = r.try_get("ab").unwrap_or(0.0);
        let c_a: f64 = r.try_get("c_a").unwrap_or(0.0);
        let c_b: f64 = r.try_get("c_b").unwrap_or(0.0);
        let n: f64 = r.try_get("n").unwrap_or(0.0);
        let Some(score) = npmi(co, c_a, c_b, n) else {
            continue;
        };
        if score < threshold {
            continue;
        }
        out.push(Candidate {
            from_id: r.try_get("a_id")?,
            to_id: r.try_get("b_id")?,
            from_name: r.try_get("a_name").unwrap_or_default(),
            to_name: r.try_get("b_name").unwrap_or_default(),
            co_sessions: co as i64,
            npmi: score,
        });
    }
    out.sort_by(|a, b| {
        b.npmi
            .partial_cmp(&a.npmi)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.from_id.cmp(&b.from_id))
            .then_with(|| a.to_id.cmp(&b.to_id))
    });
    Ok(out)
}

pub async fn apply(pool: &PgPool, candidates: &[Candidate]) -> Result<usize> {
    let mut inserted = 0usize;
    for c in candidates {
        let result = sqlx::query(
            "INSERT INTO brain_relations (from_entity, to_entity, relation_type, strength, bidirectional)
             VALUES ($1, $2, 'related_to', $3, true)
             ON CONFLICT DO NOTHING",
        )
        .bind(c.from_id)
        .bind(c.to_id)
        .bind(c.npmi.clamp(0.0, 1.0))
        .execute(pool)
        .await
        .context("inserting auto-link")?;
        inserted += result.rows_affected() as usize;
    }
    Ok(inserted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_entities_score_about_zero() {
        let score = npmi(25.0, 50.0, 50.0, 100.0).expect("finite");
        assert!(score.abs() < 1e-9, "esperaba ~0, dio {score}");
    }

    #[test]
    fn always_together_scores_one() {
        let score = npmi(10.0, 10.0, 10.0, 100.0).expect("finite");
        assert!((score - 1.0).abs() < 1e-9, "esperaba 1.0, dio {score}");
    }

    #[test]
    fn a_ubiquitous_entity_earns_no_edge_from_being_ubiquitous() {
        let score = npmi(10.0, 10.0, 100.0, 100.0).expect("finite");
        assert!(
            score.abs() < 1e-9,
            "una entidad ubicua no debe ganar aristas: {score}"
        );
    }

    #[test]
    fn co_occurring_less_than_chance_is_negative() {
        let score = npmi(1.0, 50.0, 50.0, 100.0).expect("finite");
        assert!(score < 0.0, "esperaba negativo, dio {score}");
    }

    #[test]
    fn degenerate_inputs_yield_none_not_infinity() {
        assert_eq!(npmi(0.0, 10.0, 10.0, 100.0), None);
        assert_eq!(npmi(10.0, 0.0, 10.0, 100.0), None);
        assert_eq!(npmi(10.0, 10.0, 10.0, 0.0), None);
        assert_eq!(npmi(100.0, 100.0, 100.0, 100.0), None);
    }
}

use anyhow::Result;
use sqlx::PgPool;
use std::collections::{HashMap, VecDeque};

pub async fn compute_bridges(pool: &PgPool, top_k: usize) -> Result<Vec<(String, f64)>> {
    let edges: Vec<(uuid::Uuid, uuid::Uuid)> =
        sqlx::query_as("SELECT from_entity, to_entity FROM brain_relations")
            .fetch_all(pool)
            .await?;

    let entities: Vec<(uuid::Uuid, String)> = sqlx::query_as("SELECT id, name FROM brain_entities")
        .fetch_all(pool)
        .await?;

    if entities.is_empty() || edges.is_empty() {
        return Ok(vec![]);
    }

    let mut id_to_idx: HashMap<uuid::Uuid, usize> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    for (i, (id, name)) in entities.iter().enumerate() {
        id_to_idx.insert(*id, i);
        names.push(name.clone());
    }
    let n = names.len();

    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (from, to) in &edges {
        if let (Some(&fi), Some(&ti)) = (id_to_idx.get(from), id_to_idx.get(to)) {
            adj[fi].push(ti);
            adj[ti].push(fi);
        }
    }

    let mut betweenness: Vec<f64> = vec![0.0; n];

    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut pred: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    let mut sigma: Vec<f64> = vec![0.0; n];
    let mut dist: Vec<i64> = vec![-1; n];
    let mut delta: Vec<f64> = vec![0.0; n];

    for s in 0..n {
        stack.clear();
        for p in pred.iter_mut() {
            p.clear();
        }
        sigma.fill(0.0);
        dist.fill(-1);
        delta.fill(0.0);

        sigma[s] = 1.0;
        dist[s] = 0;

        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    let norm = if n > 2 {
        ((n - 1) * (n - 2)) as f64 / 2.0
    } else {
        1.0
    };
    for b in betweenness.iter_mut() {
        *b /= norm;
    }

    let mut ranked: Vec<(String, f64)> = names
        .into_iter()
        .zip(betweenness)
        .filter(|(_, b)| *b > 0.0)
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_k);

    Ok(ranked)
}

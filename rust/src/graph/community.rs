//! Community detection: Louvain local-move with a connectivity refinement pass.
//!
//! Honest scope: this is single-level Louvain (Blondel et al. 2008) greedy
//! modularity maximization, followed by a BFS pass that splits any internally
//! disconnected community — the one well-connectedness guarantee Leiden (Traag
//! et al., Nature 2019) is known for. It is NOT full Leiden: there is no
//! aggregation/coarsening phase, so it does not build a hierarchy or reach
//! Leiden's refined-partition guarantees. The outer loop re-runs local-move on
//! the same graph, not on a contracted one.
//!
//! Phases:
//! 1. Local move: greedily move nodes to maximize modularity gain.
//! 2. Refinement: split communities that are internally disconnected.

use anyhow::Result;
use sqlx::PgPool;
use std::collections::HashMap;

const MAX_OUTER_ITERATIONS: usize = 10;
const MAX_LOCAL_ITERATIONS: usize = 50;

/// Detect communities and return them as Vec<(community_id, members)>.
///
/// Uses single-level Louvain + connectivity refinement (see module docs),
/// not full Leiden. Guarantees internally connected communities.
pub async fn detect(pool: &PgPool) -> Result<Vec<(usize, Vec<String>)>> {
    // Fetch graph
    let edges: Vec<(uuid::Uuid, uuid::Uuid, f64)> =
        sqlx::query_as("SELECT from_entity, to_entity, strength FROM brain_relations")
            .fetch_all(pool)
            .await?;

    let entities: Vec<(uuid::Uuid, String)> = sqlx::query_as("SELECT id, name FROM brain_entities")
        .fetch_all(pool)
        .await?;

    if entities.is_empty() {
        return Ok(vec![]);
    }

    // Map entity_id → index
    let mut id_to_idx: HashMap<uuid::Uuid, usize> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    for (i, (id, name)) in entities.iter().enumerate() {
        id_to_idx.insert(*id, i);
        names.push(name.clone());
    }

    let n = names.len();

    // Build adjacency (undirected weighted)
    let mut neighbors: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    let mut total_weight = 0.0;
    for (from, to, strength) in &edges {
        if let (Some(&fi), Some(&ti)) = (id_to_idx.get(from), id_to_idx.get(to)) {
            neighbors[fi].push((ti, *strength));
            neighbors[ti].push((fi, *strength));
            total_weight += strength;
        }
    }

    if total_weight == 0.0 {
        // No edges: every node is its own community
        let communities: Vec<(usize, Vec<String>)> = names
            .iter()
            .enumerate()
            .map(|(i, name)| (i, vec![name.clone()]))
            .collect();
        return Ok(communities);
    }

    // Node strengths (weighted degree)
    let node_strength: Vec<f64> = (0..n)
        .map(|i| neighbors[i].iter().map(|(_, w)| w).sum())
        .collect();

    // Initialize: each node in its own community
    let mut labels: Vec<usize> = (0..n).collect();

    // Louvain local-move outer loop (no aggregation — see module docs)
    for _outer in 0..MAX_OUTER_ITERATIONS {
        let changed = louvain_local_move(&neighbors, &node_strength, &mut labels, total_weight);
        if !changed {
            tracing::info!(
                iterations = _outer + 1,
                "Louvain community detection converged"
            );
            break;
        }
    }

    // Refinement: ensure communities are internally connected
    refine_communities(&neighbors, &mut labels);

    // Group by community
    let mut communities: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        communities.entry(label).or_default().push(names[i].clone());
    }

    let mut result: Vec<(usize, Vec<String>)> = communities.into_iter().collect();
    result.sort_by_key(|b| std::cmp::Reverse(b.1.len())); // Largest first
    Ok(result)
}

/// Detect communities and persist to `brain_communities` + `brain_node_metrics`.
pub async fn detect_and_persist(pool: &PgPool) -> Result<(Vec<(usize, Vec<String>)>, usize)> {
    let communities = detect(pool).await?;
    if communities.is_empty() {
        return Ok((vec![], 0));
    }

    let mut tx = pool.begin().await?;
    sqlx::query("UPDATE brain_node_metrics SET community_id = NULL")
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM brain_communities")
        .execute(&mut *tx)
        .await?;

    let mut nodes_updated = 0usize;
    for (idx, members) in &communities {
        let community_name = format!("community_{idx}");
        let row: (uuid::Uuid,) = sqlx::query_as(
            "INSERT INTO brain_communities (community_name, algorithm_version)
             VALUES ($1, 'louvain_refined_v1') RETURNING community_id",
        )
        .bind(&community_name)
        .fetch_one(&mut *tx)
        .await?;

        for name in members {
            let updated = sqlx::query(
                r#"INSERT INTO brain_node_metrics (node_id, community_id, last_calculated)
                   SELECT e.id, $1, NOW() FROM brain_entities e WHERE e.name = $2
                   ON CONFLICT (node_id) DO UPDATE SET
                     community_id = EXCLUDED.community_id,
                     last_calculated = NOW()"#,
            )
            .bind(row.0)
            .bind(name)
            .execute(&mut *tx)
            .await?;
            nodes_updated += updated.rows_affected() as usize;
        }
    }

    tx.commit().await?;
    Ok((communities, nodes_updated))
}

/// Phase 1: Local Move — greedily move nodes to maximize modularity gain.
///
/// Modularity gain formula (Newman-Girvan):
/// ΔQ = [Σ_in + 2·k_i,in] / (2m) - [(Σ_tot + k_i) / (2m)]²
///      - [Σ_in / (2m) - (Σ_tot / (2m))² - (k_i / (2m))²]
///
/// Returns true if any node changed community.
fn louvain_local_move(
    neighbors: &[Vec<(usize, f64)>],
    node_strength: &[f64],
    labels: &mut [usize],
    total_weight: f64,
) -> bool {
    let n = labels.len();
    let m2 = 2.0 * total_weight; // 2m for modularity formula
    let mut changed = false;

    for _iter in 0..MAX_LOCAL_ITERATIONS {
        let mut local_changed = false;

        // Precompute community totals once per iteration
        let mut community_totals: HashMap<usize, f64> = HashMap::new();
        for (node_idx, &label) in labels.iter().enumerate() {
            *community_totals.entry(label).or_default() += node_strength[node_idx];
        }

        for i in 0..n {
            if neighbors[i].is_empty() {
                continue;
            }

            let current_community = labels[i];
            let ki = node_strength[i];

            // Compute weight to each neighboring community
            let mut community_weights: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &neighbors[i] {
                *community_weights.entry(labels[j]).or_default() += w;
            }

            // Find best community (maximizes modularity gain ΔQ)
            // ΔQ = ki_in_D/m - ki*Σ_tot_D/(2m²) - (ki_in_C/m - ki*(Σ_tot_C - ki)/(2m²))
            let mut best_community = current_community;
            let mut best_delta_q = 0.0;

            let ki_in_current = community_weights
                .get(&current_community)
                .copied()
                .unwrap_or(0.0);
            let sigma_tot_current = community_totals
                .get(&current_community)
                .copied()
                .unwrap_or(0.0);

            for (&candidate, &ki_in_candidate) in &community_weights {
                if candidate == current_community {
                    continue;
                }

                let sigma_tot_candidate = community_totals.get(&candidate).copied().unwrap_or(0.0);

                // Standard Louvain ΔQ (Newman-Girvan), moving i from C to D:
                //   ΔQ = [k_i,in(D) - k_i,in(C)]/m
                //      + k_i·[(Σ_tot(C) - k_i) - Σ_tot(D)]/(2m²)
                // The neighbor-attraction term is /m, i.e. *2/m2 — it was /m2
                // (half), which biased every move toward the size-penalty term and
                // over-fragmented the partition. Verified against the closed form.
                let delta_q = (ki_in_candidate - ki_in_current) * 2.0 / m2
                    + ki * ((sigma_tot_current - ki) - sigma_tot_candidate) / (m2 * m2) * 2.0;

                if delta_q > best_delta_q {
                    best_delta_q = delta_q;
                    best_community = candidate;
                }
            }

            if best_community != current_community {
                // Update community totals incrementally
                *community_totals.entry(current_community).or_default() -= ki;
                *community_totals.entry(best_community).or_default() += ki;
                labels[i] = best_community;
                local_changed = true;
                changed = true;
            }
        }

        if !local_changed {
            break;
        }
    }

    changed
}

/// Phase 2: Refinement — ensure internal connectivity.
///
/// For each community, run BFS/DFS to verify all nodes are reachable.
/// If a community has disconnected components, split them into
/// separate communities. This is the key Leiden guarantee over Louvain.
fn refine_communities(neighbors: &[Vec<(usize, f64)>], labels: &mut [usize]) {
    // Group nodes by community
    let mut community_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        community_members.entry(label).or_default().push(i);
    }

    let mut next_label = labels.iter().copied().max().unwrap_or(0) + 1;

    for members in community_members.values() {
        if members.len() <= 1 {
            continue; // Single-node communities are trivially connected
        }

        // BFS within community to find connected components
        let member_set: std::collections::HashSet<usize> = members.iter().copied().collect();
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut components: Vec<Vec<usize>> = Vec::new();

        for &start in members {
            if visited.contains(&start) {
                continue;
            }

            // BFS from start within community
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited.insert(start);

            while let Some(node) = queue.pop_front() {
                component.push(node);
                for &(neighbor, _) in &neighbors[node] {
                    if member_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            components.push(component);
        }

        // If multiple components, split: keep first component with original label,
        // assign new labels to disconnected fragments
        if components.len() > 1 {
            for component in components.iter().skip(1) {
                for &node in component {
                    labels[node] = next_label;
                }
                next_label += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_louvain_local_move_two_clusters() {
        // Two fully connected cliques: {0,1,2} and {3,4,5}
        // connected by a weak bridge 2-3
        let neighbors = vec![
            vec![(1, 1.0), (2, 1.0)],           // 0
            vec![(0, 1.0), (2, 1.0)],           // 1
            vec![(0, 1.0), (1, 1.0), (3, 0.1)], // 2 (bridge)
            vec![(2, 0.1), (4, 1.0), (5, 1.0)], // 3 (bridge)
            vec![(3, 1.0), (5, 1.0)],           // 4
            vec![(3, 1.0), (4, 1.0)],           // 5
        ];

        let node_strength: Vec<f64> = neighbors
            .iter()
            .map(|n| n.iter().map(|(_, w)| w).sum())
            .collect();

        let total_weight: f64 = neighbors
            .iter()
            .flat_map(|n| n.iter().map(|(_, w)| w))
            .sum::<f64>()
            / 2.0;

        let mut labels: Vec<usize> = (0..6).collect();

        louvain_local_move(&neighbors, &node_strength, &mut labels, total_weight);

        // Nodes 0, 1, 2 should be in the same community
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);

        // Nodes 3, 4, 5 should be in the same community
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);

        // The two clusters should be different
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_refinement_splits_disconnected() {
        // Community {0,1,2,3} but 0-1 connected and 2-3 connected, no bridge
        let neighbors = vec![
            vec![(1, 1.0)], // 0
            vec![(0, 1.0)], // 1
            vec![(3, 1.0)], // 2
            vec![(2, 1.0)], // 3
        ];

        let mut labels = vec![0, 0, 0, 0]; // All in community 0
        refine_communities(&neighbors, &mut labels);

        // Should split into two components
        assert_eq!(labels[0], labels[1]); // 0-1 connected
        assert_eq!(labels[2], labels[3]); // 2-3 connected
        assert_ne!(labels[0], labels[2]); // Disconnected → split
    }

    #[test]
    fn test_single_node_community() {
        let neighbors: Vec<Vec<(usize, f64)>> = vec![vec![]];
        let mut labels = vec![0];
        refine_communities(&neighbors, &mut labels);
        assert_eq!(labels[0], 0);
    }
}

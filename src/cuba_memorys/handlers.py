"""Handler functions for all 12 cuba-memorys MCP tools.

CC reduction strategy: each action-dispatch handler delegates to
sub-functions, keeping each function's cyclomatic complexity ≤7.
Technique: McCabe (1976) extract-and-dispatch.

v1.4.0 activations:
  - FSRS stability updates on recall (Ye 2023)
  - Spreading activation on entity access (Collins & Loftus 1975)
  - Information density gating (Shannon 1948)
  - Embedding-enhanced dedup (cosine sim)
"""
import json
import logging
import math
from typing import Any

import asyncpg

from cuba_memorys import __version__, db, search, embeddings
from cuba_memorys.search import rrf_fuse
from cuba_memorys.hebbian import (
    fsrs_retrievability,
    fsrs_update_stability,
    information_density,
    oja_negative,
    oja_positive,
    synapse_weight_boost,
    calculate_thermal_diffusion,
)
from cuba_memorys.tfidf import tfidf_index
from cuba_memorys.constants import (
    CONTRADICTION_THRESHOLD,
    DEDUP_THRESHOLD,
    EXPORT_MAX_ENTITIES,
    EXPORT_MAX_ERRORS,
    EXPORT_MAX_OBSERVATIONS,
    EXPORT_MAX_RELATIONS,
    EXPORT_MAX_SESSIONS,
    GRAPH_RELATIONS_SQL,
    IMPORTANCE_ACCESS_BOOST,
    IMPORTANCE_NEIGHBOR_BOOST,
    IMPORTANCE_TRAVERSE_BOOST,
    MAX_CONTENT_LENGTH,
    MAX_ERROR_MESSAGE_LENGTH,
    MAX_NAME_LENGTH,
    OBS_OVERLOAD_THRESHOLD,
    VALID_ENTITY_TYPES,
)

logger = logging.getLogger("cuba-memorys.handlers")


# ── Helpers ─────────────────────────────────────────────────────────

async def _build_brain_graph(directed: bool = False) -> tuple[Any, bool]:
    """Fetch relations and build a networkx graph.

    Returns:
        (graph, has_data): The graph and whether any relations exist.
    """
    import networkx as nx  # type: ignore[import-untyped]
    rels = await db.fetch(GRAPH_RELATIONS_SQL)
    if not rels:
        return None, False
    g = nx.DiGraph() if directed else nx.Graph()
    for r in rels:
        g.add_edge(r["src"], r["dst"], weight=float(r["strength"]))
    return g, True


async def _check_dedup(
    entity_id: Any, content: str,
) -> tuple[bool, list[str], str | None]:
    """DB-side duplicate/contradiction check with embedding enhancement.

    Uses pg_trgm similarity + optional embedding cosine similarity (E3)
    for paraphrase detection.

    Returns:
        (is_duplicate, warnings, existing_content_preview)
    """
    warnings: list[str] = []
    dupes = await db.fetch(
        "SELECT id, content, similarity(content, $2) AS sim "
        "FROM brain_observations WHERE entity_id = $1 "
        "AND similarity(content, $2) > $3 "
        "ORDER BY sim DESC LIMIT 5",
        entity_id, content, CONTRADICTION_THRESHOLD,
    )
    for obs in dupes:
        sim = float(obs["sim"])
        if sim > DEDUP_THRESHOLD:
            return True, warnings, obs["content"][:80]
        # E3: Embedding-enhanced dedup — catches paraphrased duplicates
        if sim > 0.5 and embeddings.is_available():
            emb_sim = _compute_embedding_score(content, obs["content"])
            if emb_sim is not None and emb_sim > 0.92:
                return True, warnings, obs["content"][:80]
        if sim > CONTRADICTION_THRESHOLD and search.has_negation(
            content, obs["content"],
        ):
            await db.execute(
                "UPDATE brain_observations SET observation_type = 'superseded', "
                "importance = importance * 0.1, valid_until = NOW() "
                "WHERE id = $1",
                obs["id"],
            )
            warnings.append(
                f"SUPERSEDED: '{obs['content'][:60]}...' (sim={sim:.2f})"
            )
    return False, warnings, None


async def _persist_embedding(content: str, obs_id: Any) -> None:
    """Store embedding vector for an observation if pgvector is active."""
    if db.has_pgvector() and embeddings.is_available():
        vecs = await embeddings.embed_async([content])  # V7: async, no event-loop block
        if len(vecs) > 0:
            await db.execute(
                "UPDATE brain_observations SET embedding = $1 WHERE id = $2",
                vecs[0].tolist(), obs_id,
            )


def _elapsed_days(last_accessed: Any) -> float:
    """Compute elapsed days since last_accessed datetime.

    Returns 0.01 minimum to avoid division-by-zero in FSRS formulas.
    """
    if last_accessed is None:
        return 1.0
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    if hasattr(last_accessed, 'tzinfo') and last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=datetime.timezone.utc)
    delta = (now - last_accessed).total_seconds() / 86400.0
    return max(0.01, delta)


async def _fsrs_recall_boost(obs_rows: list[Any]) -> None:
    """Update FSRS stability for recalled observations (Ye 2023).

    Called when observations are accessed via entity_get or search.
    This activates the spaced repetition model that was previously
    dead code (B2 fix / E1 enhancement).
    """
    for row in obs_rows[:5]:  # Limit to top-5 to avoid excessive writes
        obs_id = row.get("id")
        stability = float(row.get("stability") or 1.0)
        difficulty = float(row.get("difficulty") or 5.0)
        elapsed = _elapsed_days(row.get("last_accessed"))
        retrievability = fsrs_retrievability(elapsed, stability)
        new_stability = fsrs_update_stability(
            stability, difficulty, retrievability,
        )
        if abs(new_stability - stability) > 0.001:
            await db.execute(
                "UPDATE brain_observations SET "
                "stability = $1, last_accessed = NOW(), "
                "access_count = access_count + 1 "
                "WHERE id = $2",
                new_stability, obs_id,
            )


async def _recall_boost_results(results: list[dict[str, Any]]) -> None:
    """Boost FSRS stability for observations found via search (E8).

    V14: Extended from top-3 to ALL observation results using batch UPDATE.
    Paper: Karpicke & Roediger (2008) — testing effect.
    """
    obs_ids = [
        str(r["id"]) for r in results
        if r.get("_type") == "observation" and r.get("id")
    ]
    if not obs_ids:
        return
    await db.execute(
        "UPDATE brain_observations SET "
        "access_count = access_count + 1, "
        "last_accessed = NOW() "
        "WHERE id = ANY($1::uuid[])",
        obs_ids,
    )


async def _check_overload(entity_id: Any) -> str | None:
    """Return overload warning if observation count exceeds threshold."""
    obs_count = await db.fetchval(
        "SELECT COUNT(*) FROM brain_observations WHERE entity_id = $1",
        entity_id,
    )
    if obs_count > OBS_OVERLOAD_THRESHOLD:
        return (
            f"entity_overloaded: {obs_count} observations. "
            "Consider brain_consolidate(action='summarize')."
        )
    return None


# ── 1. Entity (cuba_alma) — CC target: ≤7 ──────────────────────────

async def handle_brain_entity(args: dict[str, Any]) -> str:
    """CRUD for knowledge graph entities."""
    action = args["action"]
    name = args.get("name", "")[:MAX_NAME_LENGTH]

    if action == "create":
        return await _entity_create(name, args)
    if action == "get":
        return await _entity_get(name)
    if action == "update":
        return await _entity_update(name, args)
    if action == "delete":
        return await _entity_delete(name)
    return db.serialize({"error": f"Unknown action: {action}"})


async def _entity_create(name: str, args: dict[str, Any]) -> str:
    entity_type = args.get("entity_type", "concept")
    if entity_type not in VALID_ENTITY_TYPES:
        return db.serialize({
            "error": f"Invalid entity_type '{entity_type}'. "
            f"Valid: {sorted(VALID_ENTITY_TYPES)}",
        })
    row = await db.fetchrow(
        "INSERT INTO brain_entities (name, entity_type) "
        "VALUES ($1, $2) "
        "ON CONFLICT (name) DO UPDATE SET updated_at = NOW() "
        "RETURNING id, name, entity_type, importance",
        name, entity_type,
    )
    return db.serialize({"action": "created", "entity": row})


async def _entity_get(name: str) -> str:
    row = await db.fetchrow(
        "SELECT id, name, entity_type, importance, access_count, "
        "created_at, updated_at FROM brain_entities WHERE name = $1",
        name,
    )
    if not row:
        return db.serialize({"error": f"Entity '{name}' not found"})

    entity_id = row["id"]

    await db.execute(
        "UPDATE brain_entities SET access_count = access_count + 1, "
        "importance = LEAST(1.0, importance + $2), "
        "updated_at = NOW() WHERE id = $1",
        entity_id, IMPORTANCE_ACCESS_BOOST,
    )
    # E2 (V2): Spreading activation — Thermal diffusion (Heat Equation)
    # Fetches a subgraph up to degree 3 from the active entity to prevent
    # full DB scans. Uses Laplacian diffusion to spread the boost topographically.
    edges_raw = await db.fetch(
        "WITH RECURSIVE subgraph AS ("
        "  SELECT from_entity, to_entity, strength, 1 as depth FROM brain_relations "
        "  WHERE from_entity = $1 OR to_entity = $1 "
        "  UNION "
        "  SELECT r.from_entity, r.to_entity, r.strength, s.depth + 1 "
        "  FROM brain_relations r "
        "  JOIN subgraph s ON r.from_entity = s.to_entity OR r.to_entity = s.from_entity "
        "  WHERE s.depth < 3"
        ") "
        "SELECT DISTINCT from_entity, to_entity, strength FROM subgraph",
        entity_id,
    )
    if edges_raw:
        graph_edges = [
            (str(row["from_entity"]), str(row["to_entity"]), float(row["strength"]))
            for row in edges_raw
        ]

        # Run matrix operations in a separate thread to prevent blocking the async loop
        import asyncio
        loop = asyncio.get_running_loop()
        import functools
        diffusion_boosts = await loop.run_in_executor(
            None,
            functools.partial(
                calculate_thermal_diffusion, graph_edges, str(entity_id)
            )
        )

        # Batch update based on diffusion results
        if diffusion_boosts:
            # Quitamos el nodo activo para no sobreescribir su IMPORTANCE_ACCESS_BOOST de arriba
            diffusion_boosts.pop(str(entity_id), None)

            updates = [
                (boost_val, n_id)
                for n_id, boost_val in diffusion_boosts.items()
                if boost_val > 0.0001
            ]
            if updates:
                await db.execute_many(
                    "UPDATE brain_entities SET "
                    "importance = LEAST(1.0, importance + $1) "
                    "WHERE id = $2",
                    updates,
                )

    obs = await db.fetch(
        "SELECT id, content, observation_type, importance, source, "
        "stability, difficulty, last_accessed, "
        "created_at FROM brain_observations "
        "WHERE entity_id = $1 ORDER BY importance DESC",
        entity_id,
    )
    # E1/B2: FSRS stability update on recall (Ye 2023)
    await _fsrs_recall_boost(obs)
    rels = await db.fetch(
        "SELECT e.name AS target, r.relation_type, r.strength, "
        "r.bidirectional "
        "FROM brain_relations r "
        "JOIN brain_entities e ON r.to_entity = e.id "
        "WHERE r.from_entity = $1 "
        "UNION ALL "
        "SELECT e.name AS target, r.relation_type, r.strength, "
        "r.bidirectional "
        "FROM brain_relations r "
        "JOIN brain_entities e ON r.from_entity = e.id "
        "WHERE r.to_entity = $1 AND r.bidirectional = TRUE",
        entity_id,
    )
    return db.serialize({"entity": row, "observations": obs, "relations": rels})


async def _entity_update(name: str, args: dict[str, Any]) -> str:
    new_name = args.get("new_name", name)
    try:
        await db.execute(
            "UPDATE brain_entities SET name = $1, updated_at = NOW() "
            "WHERE name = $2",
            new_name, name,
        )
    except asyncpg.UniqueViolationError:
        return db.serialize({
            "error": f"Entity '{new_name}' already exists",
        })
    return db.serialize({"action": "updated", "old_name": name, "new_name": new_name})


async def _entity_delete(name: str) -> str:
    await db.execute("DELETE FROM brain_entities WHERE name = $1", name)
    return db.serialize({"action": "deleted", "name": name})


# ── 2. Observe (cuba_cronica) — CC target: ≤8 ──────────────────────

async def handle_brain_observe(args: dict[str, Any]) -> str:
    """Attach observations to entities with dedup + contradiction detection."""
    action = args["action"]
    entity_name = args["entity_name"]

    entity = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", entity_name,
    )
    if not entity:
        entity = await db.fetchrow(
            "INSERT INTO brain_entities (name, entity_type) VALUES ($1, $2) "
            "ON CONFLICT (name) DO UPDATE SET updated_at = NOW() RETURNING id",
            entity_name, args.get("entity_type", "concept"),
        )
    if not entity:
        return db.serialize({"error": f"Failed to create entity '{entity_name}'"})

    entity_id = entity["id"]

    if action == "add":
        return await _observe_add(entity_id, args)
    if action == "batch_add":
        return await _observe_batch_add(entity_id, args)
    if action == "delete":
        return await _observe_delete(entity_id, entity_name, args)
    if action == "list":
        return await _observe_list(entity_id, entity_name)
    return db.serialize({"error": f"Unknown action: {action}"})


async def _observe_add(entity_id: Any, args: dict[str, Any]) -> str:
    content = args.get("content", "")[:MAX_CONTENT_LENGTH]
    obs_type = args.get("observation_type", "fact")
    source = args.get("source", "agent")

    if not content:
        return db.serialize({"error": "content is required"})

    is_dup, warnings, existing = await _check_dedup(entity_id, content)
    if is_dup:
        return db.serialize({
            "action": "skipped", "reason": "near_duplicate",
            "existing": existing, "similarity": round(DEDUP_THRESHOLD, 3),
        })

    # E5: Information density gating (Shannon 1948)
    # Low-entropy content gets reduced starting importance
    density = information_density(content)
    initial_importance = 0.3 if density < 0.3 else 0.5

    row = await db.fetchrow(
        "INSERT INTO brain_observations "
        "(entity_id, content, observation_type, source, importance) "
        "VALUES ($1, $2, $3, $4, $5) RETURNING id, content, observation_type",
        entity_id, content, obs_type, source, initial_importance,
    )
    if row:
        await _persist_embedding(content, row["id"])
    search.cache_clear()

    result: dict[str, Any] = {"action": "added", "observation": row}
    if density < 0.3:
        result["density_warning"] = (
            f"Low information density ({density:.2f}). "
            "Starting importance reduced to 0.3."
        )
    if warnings:
        result["contradictions"] = warnings
    overload = await _check_overload(entity_id)
    if overload:
        result["warning"] = overload
    return db.serialize(result)


async def _observe_batch_add(entity_id: Any, args: dict[str, Any]) -> str:
    observations = args.get("observations", [])
    if not observations:
        return db.serialize({"error": "observations array required"})

    added = []
    skipped = []
    all_warnings: list[str] = []
    for obs_data in observations:
        content = obs_data.get("content", "")[:MAX_CONTENT_LENGTH]
        obs_type = obs_data.get("type", "fact")
        source = obs_data.get("source", "agent")

        if not content:
            skipped.append("(empty)")
            continue

        is_dup, warnings, _ = await _check_dedup(entity_id, content)
        if is_dup:
            skipped.append(content[:60])
            continue
        all_warnings.extend(warnings)

        # V5: Information density gating — parity with _observe_add
        density = information_density(content)
        initial_importance = 0.3 if density < 0.3 else 0.5

        row = await db.fetchrow(
            "INSERT INTO brain_observations "
            "(entity_id, content, observation_type, source, importance) "
            "VALUES ($1, $2, $3, $4, $5) RETURNING id, content",
            entity_id, content, obs_type, source, initial_importance,
        )
        if row:
            await _persist_embedding(content, row["id"])
        added.append(row)

    search.cache_clear()
    result: dict[str, Any] = {
        "action": "batch_added", "count": len(added),
        "observations": added,
    }
    if skipped:
        result["skipped_duplicates"] = len(skipped)
    if all_warnings:
        result["contradictions"] = all_warnings
    overload = await _check_overload(entity_id)
    if overload:
        result["warning"] = overload
    return db.serialize(result)


async def _observe_delete(
    entity_id: Any, entity_name: str, args: dict[str, Any],
) -> str:
    content = args.get("content", "")
    await db.execute(
        "DELETE FROM brain_observations "
        "WHERE entity_id = $1 AND content = $2",
        entity_id, content,
    )
    search.cache_clear()
    return db.serialize({"action": "deleted", "entity": entity_name})


async def _observe_list(entity_id: Any, entity_name: str) -> str:
    observations = await db.fetch(
        "SELECT id, content, observation_type, importance, source, "
        "access_count, last_accessed, version, created_at "
        "FROM brain_observations WHERE entity_id = $1 "
        "AND observation_type != 'superseded' "
        "AND (valid_until IS NULL OR valid_until > NOW()) "
        "ORDER BY importance DESC",
        entity_id,
    )
    return db.serialize({"entity": entity_name, "observations": observations})


# ── 3. Relate (cuba_puente) — CC: 10→dispatch ──────────────────────

async def handle_brain_relate(args: dict[str, Any]) -> str:
    """Create/delete/traverse relations between entities."""
    action = args["action"]
    if action == "create":
        return await _relate_create(args)
    if action == "delete":
        return await _relate_delete(args)
    if action in ("traverse", "infer"):
        return await _relate_traverse(action, args)
    return db.serialize({"error": f"Unknown action: {action}"})


async def _relate_create(args: dict[str, Any]) -> str:
    from_name = args["from_entity"]
    to_name = args["to_entity"]
    rel_type = args.get("relation_type", "related_to")
    bidir = args.get("bidirectional", False)

    from_e = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", from_name,
    )
    to_e = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", to_name,
    )
    if not from_e or not to_e:
        return db.serialize({"error": "One or both entities not found"})

    await db.execute(
        "INSERT INTO brain_relations "
        "(from_entity, to_entity, relation_type, bidirectional) "
        "VALUES ($1, $2, $3, $4) "
        "ON CONFLICT (from_entity, to_entity, relation_type) DO NOTHING",
        from_e["id"], to_e["id"], rel_type, bidir,
    )
    return db.serialize({
        "action": "created",
        "from": from_name, "to": to_name,
        "type": rel_type, "bidirectional": bidir,
    })


async def _relate_delete(args: dict[str, Any]) -> str:
    from_name = args["from_entity"]
    to_name = args["to_entity"]
    rel_type = args.get("relation_type", "")

    from_e = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", from_name,
    )
    to_e = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", to_name,
    )
    if not from_e or not to_e:
        return db.serialize({"error": "One or both entities not found"})

    if rel_type:
        await db.execute(
            "DELETE FROM brain_relations "
            "WHERE ((from_entity = $1 AND to_entity = $2) "
            "   OR (from_entity = $2 AND to_entity = $1)) "
            "AND relation_type = $3",
            from_e["id"], to_e["id"], rel_type,
        )
    else:
        await db.execute(
            "DELETE FROM brain_relations "
            "WHERE (from_entity = $1 AND to_entity = $2) "
            "   OR (from_entity = $2 AND to_entity = $1)",
            from_e["id"], to_e["id"],
        )
    return db.serialize({"action": "deleted", "from": from_name, "to": to_name})


async def _relate_traverse(action: str, args: dict[str, Any]) -> str:
    start_name = args.get("start_entity", "")
    max_depth = min(args.get("max_depth", 3), 5)

    start_e = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", start_name,
    )
    if not start_e:
        return db.serialize({"error": f"Entity '{start_name}' not found"})

    rows = await db.fetch(
        "WITH RECURSIVE inference AS ("
        "  SELECT r.to_entity, r.strength, 1 AS depth, "
        "    ARRAY[r.from_entity] AS path "
        "  FROM brain_relations r WHERE r.from_entity = $1 "
        "  UNION ALL "
        "  SELECT r.to_entity, "
        "    i.strength * r.strength * 0.9, "
        "    i.depth + 1, "
        "    i.path || r.from_entity "
        "  FROM brain_relations r "
        "  JOIN inference i ON r.from_entity = i.to_entity "
        "  WHERE i.depth < $2 "
        "    AND r.from_entity != ALL(i.path) "
        "    AND i.strength * r.strength > 0.1"
        ") "
        "SELECT e.name, e.entity_type, inf.strength, inf.depth "
        "FROM inference inf "
        "JOIN brain_entities e ON inf.to_entity = e.id "
        "ORDER BY inf.strength DESC "
        "LIMIT 20",
        start_e["id"], max_depth,
    )

    await db.execute(
        "UPDATE brain_relations SET "
        "strength = LEAST(1.0, strength + $2), "
        "last_traversed = NOW() "
        "WHERE from_entity = $1",
        start_e["id"], IMPORTANCE_TRAVERSE_BOOST,
    )

    return db.serialize({
        "action": action, "start": start_name,
        "max_depth": max_depth, "results": rows,
    })


# ── 4. Search (cuba_faro) — CC 46→dispatch+sub ─────────────────────

async def handle_brain_search(args: dict[str, Any]) -> str:
    """Hybrid search with RRF fusion and session-aware boosting."""
    query = args["query"]
    mode = args.get("mode", "hybrid")
    scope = args.get("scope", "all")
    limit = min(args.get("limit", 10), 50)

    if mode == "verify":
        return await _search_verify(query)
    return await _search_hybrid(query, scope, limit)


async def _search_verify(query: str) -> str:
    """Verify a claim against stored evidence (anti-hallucination)."""
    rows = await db.fetch(search.VERIFY_SQL, query)
    if not rows:
        return db.serialize({
            "mode": "verify", "query": query,
            "confidence_score": 0.0, "confidence_level": "unknown",
            "evidence": [],
            "recommendation": "No supporting evidence found. High hallucination risk.",
        })

    evidence = []
    best_score = 0.0
    best_level = "unknown"
    for r in rows:
        tfidf_sim = tfidf_index.similarity(query, r["content"])
        emb_score = _compute_embedding_score(query, r["content"])

        score, level = search.compute_confidence(
            trgm_score=float(r.get("trgm_similarity", 0)),
            tfidf_score=tfidf_sim,
            importance=float(r.get("importance", 0.5)),
            freshness_days=float(r.get("days_since_access", 0)),
            embedding_score=emb_score,
            access_count=int(r.get("access_count", 0)),  # V4
        )
        if score > best_score:
            best_score = score
            best_level = level
        evidence.append({
            "content": r["content"][:200],
            "entity": r.get("entity_name"),
            "confidence": score,
            "level": level,
            "trgm": round(float(r.get("trgm_similarity", 0)), 3),
            "tfidf": round(tfidf_sim, 3),
            "access_count": r.get("access_count", 0),
        })

    # V3: Source triangulation — penalize single-source confirmation
    unique_entities = len({e.get("entity") for e in evidence if e.get("entity")})
    diversity_factor = min(1.0, unique_entities / max(1, len(evidence)))
    best_score = round(best_score * (0.7 + 0.3 * diversity_factor), 4)
    if best_score < 0.3:
        best_level = "unknown"
    elif best_score < 0.5:
        best_level = "weak"
    elif best_score < 0.8:
        best_level = "partial"

    return db.serialize({
        "mode": "verify", "query": query,
        "confidence_score": best_score,
        "confidence_level": best_level,
        "source_diversity": round(diversity_factor, 2),
        "evidence": evidence,
    })


def _compute_embedding_score(query: str, content: str) -> float | None:
    """Compute cosine similarity between query and content embeddings.

    V10: Uses embed_cached() to avoid redundant ONNX inference.
    """
    if not embeddings.is_available():
        return None
    vec_q = embeddings.embed_cached(query)
    vec_c = embeddings.embed_cached(content)
    if vec_q.size > 0 and vec_c.size > 0:
        return embeddings.cosine_sim(vec_q, vec_c)
    return None


async def _collect_search_signals(
    query: str, scope: str, limit: int,
) -> list[list[dict[str, Any]]]:
    """Collect ranked result lists from each search signal."""
    signals: list[list[dict[str, Any]]] = []

    if scope in ("all", "entities"):
        rows = await db.fetch(search.SEARCH_ENTITIES_SQL, query, limit)
        for r in rows:
            r["_type"] = "entity"
        signals.append(rows)

    if scope in ("all", "observations"):
        rows = await db.fetch(search.SEARCH_OBSERVATIONS_SQL, query, limit)
        _annotate_observations(query, rows)
        signals.append(rows)

    if scope in ("all", "errors"):
        rows = await db.fetch(search.SEARCH_ERRORS_SQL, query, limit)
        for r in rows:
            r["_type"] = "error"
        signals.append(rows)

    if db.has_pgvector() and embeddings.is_available() and scope in ("all", "observations"):
        signals.append(await _collect_vector_signal(query, limit))

    return signals


def _annotate_observations(
    query: str, rows: list[dict[str, Any]],
) -> None:
    """Annotate observation results with grounding metadata."""
    for r in rows:
        r["_type"] = "observation"
        tfidf_sim = tfidf_index.similarity(query, r.get("content", ""))
        r["grounding"] = {
            "trgm_similarity": round(float(r.get("trgm_similarity") or 0), 3),
            "tfidf_similarity": round(float(tfidf_sim or 0), 3),
            "source": r.get("source", "agent"),
            "age_days": round(float(r.get("days_since_access") or 0), 1),
            "access_count": r.get("access_count", 0),
        }


async def _collect_vector_signal(
    query: str, limit: int,
) -> list[dict[str, Any]]:
    """Collect pgvector semantic search signal."""
    query_vec = embeddings.embed([query])
    if len(query_vec) == 0:
        return []
    vec_list = query_vec[0].tolist()
    vec_rows = await db.fetch(search.SEARCH_VECTOR_SQL, limit, vec_list)
    for r in vec_rows:
        r["_type"] = "observation"
        r["score"] = float(r.get("vector_score", 0))
    return vec_rows


async def _search_hybrid(query: str, scope: str, limit: int) -> str:
    """Multi-signal search with RRF fusion (Cormack 2009, k=60).

    V9: KG-neighbor query expansion when recall is low.
    CC reduced from 16 to 4 via extract-and-dispatch.
    """
    cached = search.cache_get(query, "hybrid", scope, limit)
    if cached is not None:
        return db.serialize({"cached": True, "results": cached})

    signals = await _collect_search_signals(query, scope, limit)
    results = rrf_fuse(signals)[:limit]

    # V9: KG-neighbor query expansion — PRF via graph (arXiv 2502.08557)
    if len(results) < limit:
        results = await _expand_via_neighbors(query, results, scope, limit)

    _boost_session_results(results, await _get_session_keywords())
    await _enrich_graphrag(results)
    _truncate_results(results)
    # E8: Active recall — boost stability of recalled observations
    await _recall_boost_results(results)

    search.cache_set(query, "hybrid", scope, limit, results)
    return db.serialize({"results": results})


async def _expand_via_neighbors(
    query: str,
    results: list[dict[str, Any]],
    scope: str,
    limit: int,
) -> list[dict[str, Any]]:
    """V9 (V2): Expand search via KG Random Walk with Restart (RWR).

    Replaces BFS with RWR (stochastic context extraction) based on the
    underlying connection graph and relation strengths.

    Args:
        query: Original search query.
        results: Initial search results.
        scope: Search scope.
        limit: Max results requested.

    Returns:
        Merged results with expanded neighbors.
    """
    entity_ids = _collect_entity_ids(results[:5])
    if not entity_ids:
        return results
    try:
        # Retrieve local subgraph (degree <= 2) to build a focused RWR graph without full DB scan
        edges_raw = await db.fetch(
            "WITH RECURSIVE subgraph AS ("
            "  SELECT from_entity, to_entity, strength, 1 as depth FROM brain_relations "
            "  WHERE from_entity = ANY($1::uuid[]) OR to_entity = ANY($1::uuid[]) "
            "  UNION "
            "  SELECT r.from_entity, r.to_entity, r.strength, s.depth + 1 "
            "  FROM brain_relations r "
            "  JOIN subgraph s ON r.from_entity = s.to_entity OR r.to_entity = s.from_entity "
            "  WHERE s.depth < 2"
            ") "
            "SELECT DISTINCT from_entity, to_entity, strength FROM subgraph",
            [str(eid) for eid in entity_ids],
        )
        if not edges_raw:
            return results

        try:
            import networkx as nx # type: ignore[import-untyped]
        except ImportError:
            # Fallback a V9 antigua en caso de no tener networkx instalado
            neighbors = await db.fetch(
                "SELECT DISTINCT e.name FROM brain_entities e "
                "JOIN brain_relations r ON "
                "  (e.id = r.to_entity AND r.from_entity = ANY($1::uuid[])) "
                "  OR (e.id = r.from_entity AND r.to_entity = ANY($1::uuid[])) "
                "LIMIT 5",
                [str(eid) for eid in entity_ids],
            )
            if not neighbors:
                return results
            expanded_terms = " ".join(n["name"] for n in neighbors)
            expanded_query = f"{query} {expanded_terms}"
            extra_signals = await _collect_search_signals(
                expanded_query, scope, limit, exclude_ids=[str(r.get("id")) for r in results]
            )
            return results + extra_signals

        G = nx.Graph()
        for row in edges_raw:
            G.add_edge(str(row["from_entity"]), str(row["to_entity"]), weight=float(row["strength"]))

        start_nodes = [str(eid) for eid in entity_ids if str(eid) in G]
        if not start_nodes:
            return results

        # Ejecutar RWR en subproceso para no bloquear el Event Loop
        import asyncio
        loop = asyncio.get_running_loop()
        import functools

        def run_rwr(graph_obj, start_node_list):
            return nx.pagerank(
                graph_obj,
                alpha=0.85,
                personalization={node: 1.0 for node in start_node_list},
                weight='weight'
            )

        rwr_scores = await loop.run_in_executor(None, functools.partial(run_rwr, G, start_nodes))

        # Filtrar nodos iniciales y ordenar por puntuación
        expanded_nodes = []
        for node, score in sorted(rwr_scores.items(), key=lambda x: x[1], reverse=True):
            if node not in start_nodes:
                expanded_nodes.append(node)
            if len(expanded_nodes) >= 5: # Top 5 vecinos contextuales
                break

        if not expanded_nodes:
            return results

        # Obtener los nombres de las entidades con puntuaciones más altas
        neighbors = await db.fetch(
            "SELECT name FROM brain_entities WHERE id = ANY($1::uuid[])",
            expanded_nodes
        )
        if not neighbors:
            return results

        expanded_terms = " ".join(n["name"] for n in neighbors)
        expanded_query = f"{query} {expanded_terms}"
        extra_signals = await _collect_search_signals(
            expanded_query, scope, limit,
        )
        extra_results = rrf_fuse(extra_signals)[:limit]
        # Merge: deduplicate by ID, keep higher-scored
        seen_ids = {str(r.get("id", "")): r for r in results}
        for er in extra_results:
            eid = str(er.get("id", ""))
            if eid not in seen_ids:
                results.append(er)
                seen_ids[eid] = er
        results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return results[:limit]
    except (asyncpg.PostgresError, KeyError) as exc:
        logger.warning("V9 KG expansion error: %s", type(exc).__name__)
        return results


async def _get_session_keywords() -> list[str]:
    """Get keywords from the active session goals."""
    session = await db.fetchrow(
        "SELECT goals FROM brain_sessions WHERE ended_at IS NULL "
        "ORDER BY started_at DESC LIMIT 1",
    )
    if session and session["goals"]:
        keywords = " ".join(str(g) for g in session["goals"]).lower().split()
        return [kw for kw in keywords if len(kw) > 3]
    return []


def _boost_session_results(
    results: list[dict[str, Any]], keywords: list[str],
) -> None:
    """Boost results matching active session goals (in-place)."""
    if not keywords:
        return
    for r in results:
        content_lower = r.get("content", r.get("name", "")).lower()
        if any(kw in content_lower for kw in keywords):
            r["rrf_score"] = r.get("rrf_score", 0) * 1.15
    results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)


def _collect_entity_ids(
    results: list[dict[str, Any]],
) -> list[Any]:
    """Extract unique entity IDs from top results."""
    entity_ids: list[Any] = []
    seen: set[str] = set()
    for r in results:
        eid = r.get("entity_id") if r.get("_type") == "observation" else r.get("id")
        if eid and str(eid) not in seen:
            entity_ids.append(eid)
            seen.add(str(eid))
    return entity_ids


def _group_neighbors(
    neighbors: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group neighbor records by source entity ID."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for n in neighbors:
        src = str(n.get("source_entity_id", ""))
        by_source.setdefault(src, []).append({
            "name": n["name"],
            "type": n.get("entity_type", ""),
            "relation": n["relation_type"],
            "strength": float(n.get("strength", 0)),
        })
    return by_source


async def _enrich_graphrag(results: list[dict[str, Any]]) -> None:
    """Enrich top-3 results with degree-1 neighbors (GraphRAG).

    CC reduced from 12 to 4 via extract helper functions.
    """
    entity_ids = _collect_entity_ids(results[:3])
    if not entity_ids:
        return
    try:
        neighbors = await db.fetch(search.NEIGHBORS_SQL, entity_ids)
        grouped = _group_neighbors(neighbors)
        for r in results[:3]:
            eid = str(
                r.get("entity_id") if r.get("_type") == "observation"
                else r.get("id", ""),
            )
            if eid in grouped:
                r["graph_context"] = grouped[eid]
    except (asyncpg.PostgresError, KeyError) as exc:
        logger.warning("GraphRAG enrichment error: %s", type(exc).__name__)


def _truncate_results(results: list[dict[str, Any]]) -> None:
    """V1: Token-budget truncation proportional to RRF score.

    Allocates character budget per result based on its score share.
    Low-scoring results get metadata stripped to reduce context bloat.
    """
    token_budget = 800
    total_rrf = sum(r.get("rrf_score", 0.01) for r in results) or 0.01
    for r in results:
        if "content" not in r:
            continue
        share = r.get("rrf_score", 0.01) / total_rrf
        char_budget = int(share * token_budget * 4)  # ~4 chars/token
        r["content"] = r["content"][:max(50, char_budget)]
        # Strip verbose metadata from low-scoring results
        if share < 0.05:
            r.pop("grounding", None)
            r.pop("graph_context", None)


# ── 5. Error Report (cuba_alarma) — CC: ≤5 ─────────────────────────

async def handle_brain_error_report(args: dict[str, Any]) -> str:
    """Report an error with Hebbian pattern detection."""
    error_type = args["error_type"][:MAX_NAME_LENGTH]
    error_message = args["error_message"][:MAX_ERROR_MESSAGE_LENGTH]
    context = json.dumps(args.get("context", {}))
    project = args.get("project", "default")[:MAX_NAME_LENGTH]

    row = await db.fetchrow(
        "INSERT INTO brain_errors "
        "(error_type, error_message, context, project) "
        "VALUES ($1, $2, $3::jsonb, $4) "
        "RETURNING id, error_type, error_message, created_at",
        error_type, error_message, context, project,
    )
    if not row:
        return db.serialize({"error": "Failed to insert error record"})

    similar = await db.fetch(
        "SELECT id, error_message, solution, resolved, synapse_weight "
        "FROM brain_errors "
        "WHERE id != $1 AND similarity(error_message, $2) > 0.4 "
        "ORDER BY similarity(error_message, $2) DESC LIMIT 5",
        row["id"], error_message,
    )

    for s in similar:
        new_weight = synapse_weight_boost(s["synapse_weight"])
        await db.execute(
            "UPDATE brain_errors SET synapse_weight = $1 WHERE id = $2",
            new_weight, s["id"],
        )

    search.cache_clear()
    return db.serialize({
        "action": "reported",
        "error": row,
        "similar_count": len(similar),
        "is_pattern": len(similar) >= 2,
        "similar_errors": similar,
    })


# ── 6. Error Solve (cuba_remedio) — CC: ≤3 ─────────────────────────

async def handle_brain_error_solve(args: dict[str, Any]) -> str:
    """Mark error as resolved and cross-reference similar unresolved."""
    error_id = args["error_id"]
    solution = args["solution"]

    await db.execute(
        "UPDATE brain_errors SET solution = $1, resolved = TRUE, "
        "resolved_at = NOW() WHERE id = $2::uuid",
        solution, error_id,
    )

    error = await db.fetchrow(
        "SELECT error_message FROM brain_errors WHERE id = $1::uuid",
        error_id,
    )
    candidates: list[dict[str, Any]] = []
    if error:
        candidates = await db.fetch(
            "SELECT id, error_message, synapse_weight "
            "FROM brain_errors "
            "WHERE resolved = FALSE AND id != $1::uuid "
            "AND similarity(error_message, $2) > 0.5 "
            "ORDER BY similarity(error_message, $2) DESC LIMIT 5",
            error_id, error["error_message"],
        )

    search.cache_clear()
    return db.serialize({
        "action": "solved",
        "error_id": error_id,
        "solution": solution,
        "similar_unresolved": candidates,
    })


# ── 7. Error Query (cuba_expediente) — CC: ≤5 ──────────────────────

async def handle_brain_error_query(args: dict[str, Any]) -> str:
    """Search past errors and solutions with anti-repetition guard."""
    query_text = args["query"]
    project = args.get("project")
    resolved_only = args.get("resolved_only", False)

    proposed = args.get("proposed_action")
    if proposed:
        past_failures = await db.fetch(
            "SELECT error_message, solution FROM brain_errors "
            "WHERE resolved = TRUE "
            "AND similarity(solution, $1) > 0.5 "
            "ORDER BY similarity(solution, $1) DESC LIMIT 3",
            proposed,
        )
        if past_failures:
            return db.serialize({
                "warning": "Similar approach was tried before",
                "past_attempts": past_failures,
            })

    base_sql = (
        "SELECT id, error_type, error_message, solution, resolved, "
        "synapse_weight, project, created_at, "
        "(0.40 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) + "
        " 0.30 * similarity(error_message, $1) + "
        " 0.30 * (synapse_weight / GREATEST("
        "   (SELECT MAX(synapse_weight) FROM brain_errors), 1.0))"
        ") AS score "
        "FROM brain_errors "
        "WHERE (search_vector @@ plainto_tsquery('simple', $1) "
        "   OR similarity(error_message, $1) > 0.3) "
    )
    params: list[Any] = [query_text]
    if project:
        params.append(project)
        base_sql += f"AND project = ${len(params)} "
    if resolved_only:
        base_sql += "AND resolved = TRUE "
    base_sql += "ORDER BY score DESC LIMIT 10"
    rows = await db.fetch(base_sql, *params)

    return db.serialize({"results": rows})


# ── 8. Session (cuba_jornada) — CC: ≤7 ─────────────────────────────

async def handle_brain_session(args: dict[str, Any]) -> str:
    """Track working sessions with goals and outcomes."""
    action = args["action"]

    if action == "start":
        name = args.get("name", "Unnamed Session")
        goals = json.dumps(args.get("goals", []))
        row = await db.fetchrow(
            "INSERT INTO brain_sessions (session_name, goals) "
            "VALUES ($1, $2::jsonb) RETURNING id, session_name, started_at",
            name, goals,
        )
        return db.serialize({"action": "started", "session": row})

    if action == "end":
        summary = args.get("summary", "")
        outcome = args.get("outcome", "success")
        row = await db.fetchrow(
            "UPDATE brain_sessions SET summary = $1, outcome = $2, "
            "ended_at = NOW() "
            "WHERE id = ("
            "  SELECT id FROM brain_sessions "
            "  WHERE ended_at IS NULL "
            "  ORDER BY started_at DESC LIMIT 1"
            ") "
            "RETURNING id, session_name, outcome, started_at, ended_at",
            summary, outcome,
        )
        if not row:
            return db.serialize({"error": "No active session to end"})
        return db.serialize({"action": "ended", "session": row})

    if action == "current":
        row = await db.fetchrow(
            "SELECT id, session_name, goals, started_at "
            "FROM brain_sessions WHERE ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
        )
        if not row:
            return db.serialize({"status": "no_active_session"})
        return db.serialize({"session": row})

    if action == "list":
        rows = await db.fetch(
            "SELECT id, session_name, outcome, started_at, ended_at "
            "FROM brain_sessions ORDER BY started_at DESC LIMIT 20",
        )
        return db.serialize({"sessions": rows})

    return db.serialize({"error": f"Unknown action: {action}"})


# ── 9. Decision (cuba_decreto) — CC: ≤10 ───────────────────────────

async def handle_brain_decision(args: dict[str, Any]) -> str:
    """Record and query architecture/design decisions."""
    action = args["action"]

    if action == "record":
        return await _decision_record(args)
    if action == "query":
        return await _decision_query(args)
    if action == "list":
        return await _decision_list()
    return db.serialize({"error": f"Unknown action: {action}"})


async def _decision_record(args: dict[str, Any]) -> str:
    entity = await db.fetchrow(
        "INSERT INTO brain_entities (name, entity_type) "
        "VALUES ('_decisions', 'config') "
        "ON CONFLICT (name) DO UPDATE SET updated_at = NOW() "
        "RETURNING id",
    )
    if not entity:
        return db.serialize({"error": "Failed to create _decisions entity"})

    decision = {
        "title": args.get("title", ""),
        "context": args.get("context", ""),
        "alternatives": args.get("alternatives", []),
        "chosen": args.get("chosen", ""),
        "rationale": args.get("rationale", ""),
    }
    row = await db.fetchrow(
        "INSERT INTO brain_observations "
        "(entity_id, content, observation_type, source) "
        "VALUES ($1, $2, 'decision', 'agent') "
        "RETURNING id, created_at",
        entity["id"], json.dumps(decision),
    )
    if not row:
        return db.serialize({"error": "Failed to insert decision"})

    search.cache_clear()
    return db.serialize({"action": "recorded", "decision": decision, "id": row["id"]})


def _parse_decisions(rows: list[Any]) -> list[dict[str, Any]]:
    """Parse decision JSON from observation content."""
    decisions = []
    for r in rows:
        try:
            d = json.loads(r["content"])
            d["id"] = str(r["id"])
            if "importance" in r:
                d["importance"] = r["importance"]
            decisions.append(d)
        except json.JSONDecodeError:
            decisions.append({"raw": r["content"], "id": str(r["id"])})
    return decisions


async def _decision_query(args: dict[str, Any]) -> str:
    query_text = args.get("query", "")
    rows = await db.fetch(
        "SELECT id, content, importance, created_at "
        "FROM brain_observations "
        "WHERE observation_type = 'decision' "
        "AND (search_vector @@ plainto_tsquery('simple', $1) "
        "     OR similarity(content, $1) > 0.3) "
        "ORDER BY importance DESC LIMIT 10",
        query_text,
    )
    return db.serialize({"decisions": _parse_decisions(rows)})


async def _decision_list() -> str:
    rows = await db.fetch(
        "SELECT id, content, importance, created_at "
        "FROM brain_observations "
        "WHERE observation_type = 'decision' "
        "ORDER BY created_at DESC LIMIT 20",
    )
    return db.serialize({"decisions": _parse_decisions(rows)})


# ── 10. Consolidate (cuba_zafra) — CC 19→dispatch ──────────────────

async def handle_brain_consolidate(args: dict[str, Any]) -> str:
    """Memory maintenance: decay, prune, merge, summarize, pagerank, etc."""
    action = args["action"]
    dispatch = {
        "decay": _consolidate_decay,
        "prune": _consolidate_prune,
        "merge": _consolidate_merge,
        "summarize": _consolidate_summarize,
        "stats": _consolidate_stats,
        "pagerank": _consolidate_pagerank,
        "find_duplicates": _consolidate_find_duplicates,
        "export": _consolidate_export,
        "backfill": _consolidate_backfill,
    }
    handler = dispatch.get(action)
    if handler is None:
        return db.serialize({"error": f"Unknown action: {action}"})
    return await handler(args)


async def _consolidate_decay(_args: dict[str, Any]) -> str:
    # B1 fix: Add last_accessed = NOW() for idempotency.
    # Without this, repeated REM runs compound decay exponentially.
    # FSRS formula: R(t,S) = (1 + t/(9*S))^(-1), Ye (2023)
    result = await db.execute(
        "UPDATE brain_observations SET "
        "importance = GREATEST(0.01, "
        "  importance * POWER("
        "    1.0 + EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0 "
        "    / (9.0 * GREATEST(stability, 0.1)), -1"
        "  )), "
        "last_accessed = NOW() "
        "WHERE last_accessed < NOW() - INTERVAL '1 day'",
    )
    search.cache_clear()
    return db.serialize({"action": "fsrs_decay_applied", "status": result})


async def _consolidate_prune(args: dict[str, Any]) -> str:
    threshold = args.get("threshold", 0.1)
    result = await db.execute(
        "DELETE FROM brain_observations "
        "WHERE importance < $1 AND access_count < 2",
        threshold,
    )
    search.cache_clear()
    return db.serialize({"action": "pruned", "threshold": threshold, "status": result})


async def _consolidate_merge(args: dict[str, Any]) -> str:
    sim_threshold = args.get("similarity_threshold", 0.8)
    dupes = await db.fetch(
        "SELECT a.id AS id_a, b.id AS id_b, "
        "a.content AS content_a, b.content AS content_b, "
        "similarity(a.content, b.content) AS sim "
        "FROM brain_observations a "
        "JOIN brain_observations b ON a.entity_id = b.entity_id "
        "  AND a.id < b.id "
        "WHERE similarity(a.content, b.content) > $1 "
        "LIMIT 50",
        sim_threshold,
    )
    merged = 0
    for d in dupes:
        await db.execute(
            "UPDATE brain_observations SET "
            "importance = LEAST(1.0, importance + 0.1) "
            "WHERE id = $1",
            d["id_a"],
        )
        await db.execute(
            "DELETE FROM brain_observations WHERE id = $1",
            d["id_b"],
        )
        merged += 1
    search.cache_clear()
    return db.serialize({"action": "merged", "pairs_merged": merged})


async def _consolidate_summarize(args: dict[str, Any]) -> str:
    entity_name = args.get("entity_name", "")
    compressed = args.get("compressed_summary", "")
    entity = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", entity_name,
    )
    if not entity:
        return db.serialize({"error": f"Entity '{entity_name}' not found"})
    count = await db.fetchval(
        "SELECT COUNT(*) FROM brain_observations WHERE entity_id = $1",
        entity["id"],
    )
    await db.execute(
        "DELETE FROM brain_observations WHERE entity_id = $1", entity["id"],
    )
    await db.execute(
        "INSERT INTO brain_observations "
        "(entity_id, content, observation_type, importance, source) "
        "VALUES ($1, $2, 'fact', 0.9, 'consolidation')",
        entity["id"], compressed,
    )
    search.cache_clear()
    return db.serialize({
        "action": "summarized", "entity": entity_name,
        "observations_replaced": count,
    })


async def _consolidate_stats(_args: dict[str, Any]) -> str:
    stats = await db.fetchrow(
        "SELECT "
        "  (SELECT COUNT(*) FROM brain_entities) AS entities, "
        "  (SELECT COUNT(*) FROM brain_observations) AS observations, "
        "  (SELECT COUNT(*) FROM brain_relations) AS relations, "
        "  (SELECT COUNT(*) FROM brain_errors) AS errors, "
        "  (SELECT COUNT(*) FROM brain_errors WHERE resolved) AS errors_resolved, "
        "  (SELECT COUNT(*) FROM brain_sessions) AS sessions, "
        "  (SELECT AVG(importance) FROM brain_observations) AS avg_importance",
    )
    return db.serialize({"stats": stats})


async def _consolidate_pagerank(_args: dict[str, Any]) -> str:
    import networkx as nx  # type: ignore[import-untyped]
    g, has_data = await _build_brain_graph(directed=True)
    if not has_data:
        return db.serialize({"action": "pagerank", "error": "No relations to rank"})
    recent = await db.fetch(
        "SELECT name FROM brain_entities ORDER BY updated_at DESC LIMIT 10",
    )
    personalization = (
        {r["name"]: 1.0 for r in recent if r["name"] in g}
        if recent else None
    )
    pr = nx.pagerank(
        g, alpha=0.85, weight="weight",
        personalization=personalization if personalization else None,
    )
    updated = 0
    for name, pr_score in pr.items():
        pr_norm = min(1.0, pr_score * len(pr))
        await db.execute(
            "UPDATE brain_entities SET "
            "importance = LEAST(1.0, 0.6 * $1 + 0.4 * importance), "
            "updated_at = NOW() WHERE name = $2",
            pr_norm, name,
        )
        updated += 1
    search.cache_clear()
    top5 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
    return db.serialize({
        "action": "personalized_pagerank", "entities_updated": updated,
        "top_5": [{"name": n, "score": round(s, 4)} for n, s in top5],
    })


async def _consolidate_find_duplicates(_args: dict[str, Any]) -> str:
    duplicates = await db.fetch(
        "SELECT a.name AS a, b.name AS b, "
        "  similarity(a.name, b.name) * 100 AS similarity "
        "FROM brain_entities a "
        "JOIN brain_entities b ON a.id < b.id "
        "WHERE similarity(a.name, b.name) > 0.8 "
        "ORDER BY similarity(a.name, b.name) DESC LIMIT 20",
    )
    return db.serialize({"action": "find_duplicates", "duplicates": duplicates})


async def _consolidate_export(_args: dict[str, Any]) -> str:
    entities = await db.fetch(
        "SELECT id, name, entity_type, importance, access_count, "
        "created_at, updated_at FROM brain_entities LIMIT $1",
        EXPORT_MAX_ENTITIES,
    )
    observations = await db.fetch(
        "SELECT id, entity_id, content, observation_type, importance, "
        "access_count, source, version, created_at "
        "FROM brain_observations LIMIT $1",
        EXPORT_MAX_OBSERVATIONS,
    )
    relations = await db.fetch(
        "SELECT * FROM brain_relations LIMIT $1",
        EXPORT_MAX_RELATIONS,
    )
    errors = await db.fetch(
        "SELECT * FROM brain_errors LIMIT $1",
        EXPORT_MAX_ERRORS,
    )
    sessions = await db.fetch(
        "SELECT * FROM brain_sessions LIMIT $1",
        EXPORT_MAX_SESSIONS,
    )
    return db.serialize({
        "version": __version__,
        "entities": entities,
        "observations": observations,
        "relations": relations,
        "errors": errors,
        "sessions": sessions,
    })


async def _consolidate_backfill(_args: dict[str, Any]) -> str:
    tfidf_count = await db.rebuild_tfidf_index()
    embedding_count = await db.rebuild_embeddings()
    search.cache_clear()
    return db.serialize({
        "action": "backfill",
        "tfidf_documents_indexed": tfidf_count,
        "embeddings_backfilled": embedding_count,
    })


# ── 11. Feedback (cuba_eco) — CC 12→dispatch ───────────────────────

async def handle_brain_feedback(args: dict[str, Any]) -> str:
    """RLHF feedback: positive/negative/correct."""
    action = args["action"]
    entity_name = args.get("entity_name")
    obs_id = args.get("observation_id")

    if entity_name:
        return await _feedback_entity(action, entity_name)
    if obs_id:
        return await _feedback_observation(action, obs_id, args.get("correction"))
    return db.serialize({"error": "entity_name or observation_id required"})


async def _feedback_entity(action: str, entity_name: str) -> str:
    row = await db.fetchrow(
        "SELECT id, importance FROM brain_entities WHERE name = $1",
        entity_name,
    )
    if not row:
        return db.serialize({"error": f"Entity '{entity_name}' not found"})

    if action == "positive":
        new_imp = oja_positive(row["importance"])
    elif action == "negative":
        new_imp = oja_negative(row["importance"])
    else:
        return db.serialize({"error": "correct requires observation_id"})

    await db.execute(
        "UPDATE brain_entities SET importance = $1, updated_at = NOW() "
        "WHERE id = $2",
        new_imp, row["id"],
    )
    search.cache_clear()
    return db.serialize({
        "action": action, "entity": entity_name,
        "old_importance": row["importance"],
        "new_importance": new_imp,
    })


async def _feedback_observation(
    action: str, obs_id: str, correction: str | None,
) -> str:
    row = await db.fetchrow(
        "SELECT id, importance, content, version, previous_versions, "
        "stability, difficulty, last_accessed "
        "FROM brain_observations WHERE id = $1::uuid",
        obs_id,
    )
    if not row:
        return db.serialize({"error": f"Observation '{obs_id}' not found"})

    if action == "positive":
        new_imp = oja_positive(row["importance"])
        # E1/B2: FSRS stability update on positive recall (Ye 2023)
        stability = float(row.get("stability") or 1.0)
        difficulty = float(row.get("difficulty") or 5.0)
        elapsed = _elapsed_days(row.get("last_accessed"))
        retrievability = fsrs_retrievability(elapsed, stability)
        new_stability = fsrs_update_stability(
            stability, difficulty, retrievability,
        )
        await db.execute(
            "UPDATE brain_observations SET importance = $1, "
            "access_count = access_count + 1, last_accessed = NOW(), "
            "stability = $3, difficulty = GREATEST(1, difficulty - 0.1) "
            "WHERE id = $2::uuid",
            new_imp, obs_id, new_stability,
        )
    elif action == "negative":
        new_imp = oja_negative(row["importance"])
        await db.execute(
            "UPDATE brain_observations SET importance = $1, "
            "difficulty = LEAST(10, difficulty + 0.2) "
            "WHERE id = $2::uuid",
            new_imp, obs_id,
        )
    elif action == "correct" and correction:
        new_imp = row["importance"]
        # B8 fix: asyncpg returns JSONB as Python objects — no json.loads
        prev = row["previous_versions"] if row["previous_versions"] else []
        prev.append({
            "content": row["content"],
            "version": row["version"],
            "corrected_at": str(await db.fetchval("SELECT NOW()")),
        })
        await db.execute(
            "UPDATE brain_observations SET content = $1, "
            "version = version + 1, "
            "previous_versions = $2::jsonb, "
            "last_accessed = NOW() WHERE id = $3::uuid",
            correction, json.dumps(prev), obs_id,
        )
    else:
        return db.serialize({"error": "correction text required"})

    search.cache_clear()
    return db.serialize({
        "action": action, "observation_id": obs_id,
        "old_importance": row["importance"],
        "new_importance": new_imp,
    })


# ── 12. Analytics (cuba_vigia) — CC 27→dispatch ────────────────────

async def handle_brain_analytics(args: dict[str, Any]) -> str:
    """Knowledge graph analytics: summary, health, drift, communities, bridges."""
    metric = args["metric"]
    dispatch = {
        "summary": _analytics_summary,
        "health": _analytics_health,
        "drift": _analytics_drift,
        "communities": _analytics_communities,
        "bridges": _analytics_bridges,
    }
    handler = dispatch.get(metric)
    if handler is None:
        return db.serialize({"error": f"Unknown metric: {metric}"})
    return await handler()


async def _analytics_summary() -> str:
    row = await db.fetchrow(
        "SELECT "
        "  (SELECT COUNT(*) FROM brain_entities) AS entities, "
        "  (SELECT COUNT(*) FROM brain_observations) AS observations, "
        "  (SELECT COUNT(*) FROM brain_relations) AS relations, "
        "  (SELECT COUNT(*) FROM brain_errors) AS total_errors, "
        "  (SELECT COUNT(*) FROM brain_errors WHERE resolved) AS resolved_errors, "
        "  (SELECT COUNT(*) FROM brain_sessions) AS sessions",
    )
    obs_count = int(row["observations"]) if row else 0
    token_est = obs_count * 20
    return db.serialize({"summary": row, "estimated_tokens": token_est})


def _compute_entropy(counts: list[Any], key: str = "cnt") -> float:
    """Shannon entropy H = -Σ p·log₂(p) for a distribution."""
    total = sum(r[key] for r in counts) if counts else 1
    entropy = 0.0
    for r in counts:
        p = r[key] / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


async def _analytics_health() -> str:
    row = await db.fetchrow(
        "SELECT "
        "  (SELECT AVG(importance) FROM brain_observations) AS avg_importance, "
        "  (SELECT COUNT(*) FROM brain_observations "
        "   WHERE last_accessed < NOW() - INTERVAL '90 days') AS stale_observations, "
        "  (SELECT COUNT(*) FROM brain_entities "
        "   WHERE access_count = 0) AS unused_entities, "
        "  (SELECT pg_size_pretty(pg_database_size(current_database()))) AS db_size",
    )
    type_counts = await db.fetch(
        "SELECT entity_type, COUNT(*) AS cnt FROM brain_entities "
        "GROUP BY entity_type",
    )
    entity_entropy = _compute_entropy(type_counts)

    obs_counts = await db.fetch(
        "SELECT observation_type, COUNT(*) AS cnt FROM brain_observations "
        "GROUP BY observation_type",
    )
    obs_entropy = _compute_entropy(obs_counts)

    err_stats = await db.fetchrow(
        "SELECT "
        "  (SELECT COUNT(*) FROM brain_errors WHERE resolved) AS resolved, "
        "  (SELECT COUNT(*) FROM brain_errors) AS total, "
        "  (SELECT AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))) "
        "   FROM brain_errors WHERE resolved AND resolved_at IS NOT NULL) AS avg_mttr_seconds",
    )

    max_entity_entropy = math.log2(max(1, len(type_counts))) if type_counts else 1.0
    max_obs_entropy = math.log2(max(1, len(obs_counts))) if obs_counts else 1.0

    return db.serialize({
        "health": row,
        "entropy": {
            "entity_types": round(entity_entropy, 3),
            "observation_types": round(obs_entropy, 3),
            "knowledge_diversity_score": round(
                ((entity_entropy / max(0.001, max_entity_entropy)) +
                 (obs_entropy / max(0.001, max_obs_entropy))) / 2.0, 3
            ),
            "interpretation": "Higher entropy = more diverse knowledge (score 0-1)",
        },
        "error_metrics": {
            "resolution_rate": round(
                float(err_stats["resolved"]) / max(1, float(err_stats["total"])), 3
            ) if err_stats else 0,
            "mttr_minutes": round(
                float(err_stats["avg_mttr_seconds"] or 0) / 60, 1
            ) if err_stats else None,
        },
    })


async def _analytics_drift() -> str:
    from scipy import stats as sp_stats  # type: ignore[import-not-found,import-untyped]
    row = await db.fetchrow(
        "WITH recent AS ("
        "  SELECT error_type, COUNT(*)::float AS cnt "
        "  FROM brain_errors WHERE created_at > NOW() - INTERVAL '7 days' "
        "  GROUP BY error_type"
        "), historical AS ("
        "  SELECT error_type, COUNT(*)::float / 4.0 AS expected "
        "  FROM brain_errors "
        "  WHERE created_at BETWEEN NOW() - INTERVAL '37 days' "
        "    AND NOW() - INTERVAL '7 days' "
        "  GROUP BY error_type"
        ") "
        "SELECT "
        "  COALESCE(SUM(POWER(r.cnt - h.expected, 2) / "
        "    NULLIF(h.expected, 0)), 0) AS chi_squared, "
        "  COUNT(*) AS categories "
        "FROM recent r JOIN historical h USING (error_type)",
    )
    chi_sq = float(row["chi_squared"]) if row else 0
    categories = int(row["categories"]) if row else 0
    df = max(1, categories - 1)
    p_value = float(sp_stats.chi2.sf(chi_sq, df)) if categories > 0 else 1.0

    return db.serialize({
        "chi_squared": round(chi_sq, 4),
        "degrees_of_freedom": df,
        "p_value": round(p_value, 6),
        "categories": categories,
        "drift_detected": p_value < 0.05,
    })


async def _analytics_communities() -> str:
    """Louvain community detection with V12 summaries.

    V12: For communities ≥3 entities, generates a structured summary
    from top-3 observations per entity (by importance). Inspired by
    Zep/Graphiti hierarchical KG (arXiv 2501.13956 §2.3).
    """
    import networkx as nx  # type: ignore[import-untyped]
    g, has_data = await _build_brain_graph()
    if not has_data:
        return db.serialize({"communities": [], "note": "No relations to analyze"})
    communities = nx.community.louvain_communities(g, resolution=1.0)  # type: ignore[union-attr]
    community_list = []
    for i, c in enumerate(sorted(communities, key=len, reverse=True)[:10]):
        entry: dict[str, Any] = {
            "id": i, "members": sorted(c), "size": len(c),
        }
        if len(c) >= 3:
            entry["summary"] = await _community_summary(sorted(c))
        community_list.append(entry)
    return db.serialize({
        "communities": community_list, "total": len(community_list),
    })


async def _community_summary(member_names: list[str]) -> str:
    """Generate a text summary of a community's knowledge.

    Args:
        member_names: Entity names in the community.

    Returns:
        Compressed summary string.
    """
    rows = await db.fetch(
        "SELECT e.name, o.content FROM brain_observations o "
        "JOIN brain_entities e ON o.entity_id = e.id "
        "WHERE e.name = ANY($1) "
        "ORDER BY o.importance DESC LIMIT $2",
        member_names, len(member_names) * 3,
    )
    if not rows:
        return "No observations in this community"
    by_entity: dict[str, list[str]] = {}
    for r in rows:
        by_entity.setdefault(r["name"], []).append(r["content"][:80])
    parts = [f"{name}: {'; '.join(obs)}" for name, obs in by_entity.items()]
    return " | ".join(parts[:5])


async def _analytics_bridges() -> str:
    import networkx as nx  # type: ignore[import-untyped]
    g, has_data = await _build_brain_graph()
    if not has_data:
        return db.serialize({"bridges": [], "note": "No relations to analyze"})
    bc = nx.betweenness_centrality(g, normalized=True)  # type: ignore[union-attr]
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
    return db.serialize({
        "bridges": [
            {"entity": n, "centrality": round(s, 4)} for n, s in top
        ],
        "interpretation": "High centrality = bridge connecting different knowledge areas",
    })


# ── Handler Dispatch Table ──────────────────────────────────────────

HANDLERS: dict[str, Any] = {
    "cuba_alma": handle_brain_entity,
    "cuba_cronica": handle_brain_observe,
    "cuba_puente": handle_brain_relate,
    "cuba_faro": handle_brain_search,
    "cuba_alarma": handle_brain_error_report,
    "cuba_remedio": handle_brain_error_solve,
    "cuba_expediente": handle_brain_error_query,
    "cuba_jornada": handle_brain_session,
    "cuba_decreto": handle_brain_decision,
    "cuba_zafra": handle_brain_consolidate,
    "cuba_eco": handle_brain_feedback,
    "cuba_vigia": handle_brain_analytics,
}

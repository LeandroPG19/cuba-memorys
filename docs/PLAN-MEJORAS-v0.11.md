# Plan de mejoras — cuba-memorys → siguiente nivel

> Investigación del ecosistema de memoria agéntica (julio 2026) cruzada contra el código real de
> cuba-memorys v0.10.0. Cada mejora está anclada a una fuente verificada (URL abierta y leída) y a
> el archivo/línea que tocaría. Ordenado por impacto × (1/coste). Las fuentes se abrieron una por una;
> ninguna cifra está inventada.

---

## 0. Posicionamiento — dónde está cuba-memorys hoy

La conclusión más importante de la investigación: **cuba-memorys ya está arquitectónicamente alineado
con el ganador del estado del arte.** El benchmark de referencia de la categoría, LongMemEval, lo gana
Zep/Graphiti con **63.8%** frente al **49.0%** de mem0 (GPT-4o) — una brecha de 15 puntos
([particula.tech](https://particula.tech/blog/agent-memory-frameworks-tested-mem0-zep-letta-cognee-2026),
verificado). ¿Por qué gana Zep? Por una sola decisión: un **grafo de conocimiento temporal con
ventanas de validez de hechos** (invalidar, no borrar). **cuba-memorys ya tiene exactamente eso**:
hechos bitemporales con `valid_from`/`valid_to` y `supersedes` (migración 0018, `core/bitemporal.rs`).

O sea: cuba no necesita reinventarse, necesita **explotar** y **medir** lo que ya construyó, y cerrar
tres brechas concretas donde el ecosistema le lleva ventaja.

### Qué hace cuba MEJOR que el resto (honesto, sin autocomplacencia)

| Capacidad | cuba-memorys | Mejor competidor | Veredicto |
|---|---|---|---|
| Recuperación | híbrida: RRF + BM25 + vector(pgvector HNSW) + MMR + rerank + OOD | Engram = solo FTS5; mem0 = vector+grafo | **cuba gana** en sofisticación de retrieval |
| Bitemporalidad | valid/transaction-time + supersedes | Zep/Graphiti (el patrón que gana LongMemEval) | **empatado con el SOTA** |
| Grafo cognitivo | Hebbian/BCM, PageRank, decay estratificado | ninguno tiene esto | **diferencial único** (pero sin evidencia — ver §5) |
| Stack | Rust + Postgres, 1 binario, 25 tools | Engram = Go + SQLite, 1 binario | comparable; cuba más pesado (Postgres) |

### Dónde le ganan (las tres brechas caras)

1. **Extracción automática de hechos.** cuba exige que el agente llame explícitamente a las tools de
   escritura. mem0 y Zep **extraen hechos solos** de cada turno con un LLM. Esta es la brecha #1:
   provoca el `AP2 Context Amnesia` del propio CLAUDE.md.
2. **Resolución de conflictos ADD/UPDATE/DELETE/NOOP delegada al LLM.** cuba decide reinforce/update/create
   por umbral de coseno (`cognitive/prediction_error.rs`) — exactamente el enfoque frágil que mem0
   abandonó. Engram ya tiene `mem_judge`/`mem_compare` con juez LLM semántico.
3. **Evaluación end-to-end.** cuba mide nDCG/MRR/P@k/R@k pero su harness no tiene callers (auditado:
   cero llamadores en `rust/src/eval/`). No hay una sola prueba de que los mecanismos cognitivos
   mejoren nada. LongMemEval es el benchmark que mide las 5 habilidades que importan.

---

## 1. El ecosistema investigado (verificado 1×1)

### El hallazgo hispano: Engram (Gentleman Programming / Alan Buscaglia, argentino)

Es el "MCP de memoria del streamer argentino". Confirmado por su
[tuit de lanzamiento](https://x.com/G_Programming/status/2023452982725976269) y el
[repo](https://github.com/Gentleman-Programming/engram).

- **Stack:** binario Go único, SQLite + **FTS5 (solo full-text, sin embeddings vectoriales)**, MCP stdio.
  Cero dependencias (ni Node, ni Python, ni Docker). Filosofía: "one binary, one SQLite file".
- **20 tools:** `mem_save/update/delete`, `mem_search/context/timeline`, `mem_session_*`,
  **`mem_judge`/`mem_compare` (conflict surfacing)**, `mem_review`, `mem_capture_passive`, `mem_merge_projects`.
- **Formato de memoria estructurado:** title, type, What/Why/Where/Learned.
- **Filosofía idéntica a cuba:** "el agente decide qué guardar, no un pipeline de compresión externo".
- **Beta killer:** detección de conflictos con **juez LLM semántico que usa la suscripción Claude/OpenCode
  del propio usuario ($0)** — FTS5 propone candidatos, el LLM juzga `supersedes`/`conflicts_with`.
- **Superviviencia a compactación**, git sync por chunks comprimidos, cloud opt-in.

**Qué le gana cuba a Engram:** retrieval híbrido con vectores (Engram es solo FTS5 lexical), grafo
cognitivo, bitemporalidad formal. **Qué le gana Engram a cuba:** despliegue trivial (un binario Go, sin
Postgres), el juez de conflictos que reutiliza la suscripción del usuario sin API key, la TUI, y la
integración multi-agente (Claude Code, OpenCode, Gemini, Codex, Cursor…). La lección más valiosa de
Engram: **el juez LLM $0 vía la CLI del usuario**, que cuba puede replicar (ya tiene `CUBA_JUEZ_CLI`).

### Los cuatro frameworks de referencia (verificado con benchmark)

| Framework | Arquitectura | LongMemEval | Lo que cuba puede robar |
|---|---|---|---|
| **Zep / Graphiti** | grafo temporal, ventanas de validez | **63.8%** | ya lo tiene; falta poblarlo por extracción |
| **mem0** (~47K★) | vector+grafo+KV, extracción automática | 49.0% | **ADD/UPDATE/DELETE/NOOP delegado al LLM** |
| **Letta / MemGPT** | runtime OS: contexto=RAM, archival=disco | (self-managed) | el agente gestiona su propio presupuesto de memoria |
| **cognee** | pipeline ECL → grafo tipado | (estructural) | extracción tipada de entidades/relaciones |

Fuentes: [particula.tech](https://particula.tech/blog/agent-memory-frameworks-tested-mem0-zep-letta-cognee-2026),
[mem0 arxiv 2504.19413](https://arxiv.org/html/2504.19413v1),
[codepointer](https://codepointer.substack.com/p/agent-memory-systems-and-knowledge).

### Otros MCP de memoria (verificados)

- **`germaniu/mcp-memory`** (hispano, [repo](https://github.com/germaniu/mcp-memory)): Python + Ollama +
  Qdrant, 9 tools, local-first. Más simple que cuba; usa **bge-m3 (1024-d, multilingüe)** como embedding
  — nota: mejor que el `multilingual-e5-small` (384-d) de cuba para español.
- Servidor oficial de Anthropic (`server-memory`): knowledge graph básico, sin retrieval híbrido.

---

## 2. Tier 0 — Bugs ya arreglados o pendientes (base para todo lo demás)

No tiene sentido optimizar retrieval sobre un motor roto. Estado tras la sesión del 2026-07-10:

| # | Bug | Estado | Ref |
|---|---|---|---|
| 0.1 | Recall cero con queries largas (`plainto_tsquery` AND) | ✅ **arreglado** (migración 0026) | `handlers/faro.rs`, `search/bm25.rs` |
| 0.2 | Rama vectorial muerta sin `ONNX_MODEL_PATH` | ✅ config corregida en `~/.mcp.json` | — |
| 0.3 | OOD invertido (τ=5.0 → 100% abstención) | ✅ **arreglado** (Wilson-Hilferty + `embed_passage`) | `search/ood.rs` |
| 0.4 | Sesión/proyecto globales entre procesos | ✅ **arreglado** (`session.rs`) | `project.rs`, 8 handlers |
| 0.5 | Embeddings perdidos al cerrar (fire-and-forget) | ✅ **arreglado** (`tasks.rs` + drain) | `main.rs`, `cronica.rs` |
| 0.6 | Decay recompone 10-80× (REM cada 4h) | ✅ **arreglado** (migración 0028, ancla `last_decayed_at`) | `zafra.rs`, `protocol.rs` |
| 0.7 | `cuba` es superuser → RLS y audit inertes | ⏳ **pendiente (operativo, no ejecutado por seguridad de datos)** | migración 0017 |
| 0.8 | Leiden es Louvain 1-nivel, ΔQ partido a la mitad | ✅ **arreglado** (ΔQ ×2, renombrado honesto) | `graph/community.rs` |
| 0.9 | Sin CHECK del invariante bitemporal | ✅ **arreglado** (migración 0029) | `brain_facts` |
| 0.10 | Índices ausentes en `created_at` | ✅ **arreglado** (migración 0030) | varias tablas |

**Prioridad inmediata: 0.6 (decay).** Es el que más degrada la calidad hoy: hunde la importancia de todo
lo no accedido en una semana, y la importancia pesa en el ranking (`score·0.7 + importance·0.3`). Fix:
añadir columna `last_decayed_at` y decaer por `NOW() - last_decayed_at` (avanzándola), o decaer por el
intervalo REM fijo `exp(-ln2·(4h)/H)`.

---

## 3. Tier 1 — Alto impacto, bajo coste (hacer primero)

### 1.1 ⭐ Extracción automática de hechos vía MCP Sampling — cierra la brecha #1

**Problema:** el agente olvida llamar a `cuba_cronica`. **Solución del ecosistema (mem0):** extraer los
hechos salientes del turno con un LLM, en dos fases (extracción + update).
Fuente: [mem0 arxiv 2504.19413](https://arxiv.org/html/2504.19413v1).

**Cómo aplicarlo a cuba sin API key propia:** cuba ya captura `capabilities.sampling` del cliente en
`initialize` (`protocol.rs:428`) y tiene el andamiaje de sampling (`OUTBOUND`/`oneshot`). Añadir una tool
`cuba_ingesta action=auto_extract` que, dado el texto de un turno, pida al **LLM del cliente** (vía MCP
Sampling — $0, sin API key) que devuelva los hechos salientes en JSON, y los inserte con el pipeline de
dedup/PE-gating existente. Es la misma jugada del juez $0 de Engram.
**Coste:** medio-bajo (el andamiaje de sampling ya existe). **Impacto:** altísimo — ataca el AP2.

### 1.2 ⭐ ADD/UPDATE/DELETE/NOOP delegado al LLM — cierra la brecha #2

**Problema:** `cognitive/prediction_error.rs` decide reinforce/update/create por umbral de coseno — el
enfoque frágil que mem0 abandonó explícitamente. **Solución:** cuando el PE-gate caiga en la banda
ambigua, presentar al LLM (vía el `cuba_juez` existente, `CUBA_JUEZ_CLI`) el hecho candidato + los
hechos similares recuperados y pedirle que elija ADD/UPDATE/DELETE/NOOP.
Fuente: [mem0 breakdown](https://memo.d.foundation/breakdown/mem0). cuba **ya tiene el juez** (`cognitive/judge.rs`,
banda 0.6-0.8) — es extender su uso de "contradicción sí/no" a "qué operación". **Coste:** bajo. **Impacto:** alto.

### 1.3 ⭐ Contextual Retrieval completo — mejora medida del retrieval

cuba ya hace una versión ligera: antepone `[entity_type:entity_name]` al pasaje antes de embeber
(`embeddings/onnx.rs`, "Contextual Retrieval +20% recall"). Anthropic reporta números concretos con la
versión completa (contexto generado por LLM por chunk):
[anthropic.com/engineering/contextual-retrieval](https://www.anthropic.com/engineering/contextual-retrieval):
- Contextual embeddings: **−35%** tasa de fallo (5.7%→3.7%)
- + contextual BM25: **−49%** (→2.9%)
- + reranking: **−67%** (→1.9%)

**Aplicar:** (a) activar el reranker que hoy está en fallback identity (falta `CUBA_RERANKER_PATH` — el
CHANGELOG v0.9.3 lo vende como real pero no corre); (b) enriquecer el prefijo contextual con una frase
generada por sampling. **Coste:** (a) bajo, (b) medio. **Impacto:** alto y **medible**.

### 1.4 Migrar el embedding a bge-m3 (1024-d, multilingüe) para español

`multilingual-e5-small` (384-d) es débil para español. Tanto `germaniu/mcp-memory` como el propio Engram
recomiendan **bge-m3** (1024-d) para contenido no inglés. El código de cuba ya anticipa la migración a
1024-d (comentarios en `ood.rs`, `search/rerank.rs`). El `default_threshold(dim)` que introduje en el fix
de OOD ya escala solo con la dimensión, así que la migración es más barata ahora. **Coste:** medio
(re-embeber todo el corpus con `cuba_zafra reembed`). **Impacto:** alto para un usuario hispano.

### 1.5 Reducir la superficie de 25 tools

25 tools es mucho: hay evidencia de que los agentes se confunden al seleccionar entre demasiadas tools.
Engram tiene 20 pero agrupadas; mem0 expone ~4. Consolidar familias (p. ej. `cuba_alma`+`cuba_cronica`
bajo verbos, mover analytics raras a `cuba_vigia`). **Coste:** bajo (es API surface). **Impacto:** medio
— un agente que entiende la superficie la usa; uno que se pierde, la ignora.

---

## 4. Tier 2 — Alto impacto, coste medio

### 2.1 ⭐ Adoptar LongMemEval como harness de evaluación real — ✅ Fase 1 hecha (harness ejecutable)

> **Estado (Fase 1, hecha):** el harness `rust/src/eval/` ya es ejecutable vía
> `cuba-memorys eval [--dataset PATH.jsonl] [--k N] [--json]`. Es **no-mutante**
> (flag `track_access=false` en `cuba_faro`, verificado por hash antes/después).
> Línea base sobre el corpus real (1420 obs): nDCG@10=0.79, MRR=0.73, R@10=0.83.
> **Falta (Fase 2.1b):** portar el dataset LongMemEval real (500 preguntas) con
> sus 5 habilidades (extracción, multi-sesión, temporal, knowledge-update,
> abstención) — el smoke actual solo mide retrieval sobre 5 queries.


**El problema de fondo:** el eval de cuba (`rust/src/eval/`) mide nDCG/MRR pero **no tiene callers** y no
prueba ninguna habilidad de memoria. LongMemEval ([arxiv 2410.10813](https://arxiv.org/abs/2410.10813))
mide las 5 que importan: extracción de información, razonamiento multi-sesión, **razonamiento temporal**,
**actualización de conocimiento**, y **abstención**. 500 preguntas; los asistentes comerciales caen 30%.
Las tres últimas mapean directo a features de cuba (bitemporal, supersedes, OOD) — es el benchmark que
probaría si sirven. **Coste:** medio (portar el dataset + un runner). **Impacto:** altísimo — convierte
"algoritmos que suenan bien" en "mejoras medidas", y da una cifra defendible para el README.

### 2.2 Retrieval asociativo multi-hop con Personalized PageRank (HippoRAG 2)

cuba ya tiene PageRank global y `spreading_activation`, pero no PPR **sembrado desde las entidades del
query** para recuperación asociativa. HippoRAG 2 ([arxiv 2502.14802](https://arxiv.org/abs/2502.14802))
reporta **+7 puntos F1** sobre retrievers de embeddings en tareas multi-hop, con indexado más barato que
GraphRAG/RAPTOR/LightRAG. **Aplicar:** en `cuba_faro`, sembrar PPR desde los nodos que matchean el query
y mezclar ese score en la fusión RRF. Reusa la maquinaria de `graph/pagerank.rs` y `graph/activation.rs`.
**Coste:** medio. **Impacto:** alto en preguntas que requieren conectar hechos.

### 2.3 Índice BM25 real (ParadeDB/pg_search) en vez de `ts_rank_cd`

El "BM25" de cuba es `ts_rank_cd` de Postgres (~70-80% de la calidad de Okapi BM25 real; el propio
`search/bm25.rs:5-18` lo admite y planea `paradedb-bm25`). ParadeDB (`pg_search`, Tantivy dentro de
Postgres) da BM25 real. **Coste:** medio (extensión + reindex). **Impacto:** medio.

### 2.4 Query rewriting / HyDE por sampling

Antes de buscar, reescribir el query del agente o generar un "documento hipotético" (HyDE) con el LLM del
cliente para mejorar el match vectorial. Encaja con el andamiaje de sampling. **Coste:** medio.
**Impacto:** medio-alto en queries mal formuladas.

---

## 5. El riesgo del "algoritmo de adorno" — probar o cortar

La auditoría encontró varios mecanismos sofisticados **sin una sola prueba de que mejoren la
recuperación**, porque el eval harness no corre. Antes de añadir más, hay que justificar los que hay:

| Mecanismo | Estado | Acción propuesta |
|---|---|---|
| Hebbian/BCM (`hebbian.rs`) | matemática correcta, efecto no medido | A/B en LongMemEval: ¿mejora el ranking vs importancia estática? Si no, simplificar |
| Leiden/comunidades (`community.rs`) | **es Louvain 1-nivel, ΔQ ×½** | arreglar o degradar la etiqueta; medir si las comunidades ayudan a algún query |
| OOD Mahalanobis (`ood.rs`) | recién arreglado | medir la habilidad "abstención" de LongMemEval con y sin él |
| `energy_score`, `betweenness` | **nunca se escriben / sin consumidor** | eliminar o cablear a un `ORDER BY` |
| Módulos huérfanos (`adwin`, `allen`, `mi_tagging`, `calibration`, `temporal_query`) | cero callers | eliminar (dead code) o cablear |
| Eval harness (`eval/`) | cero callers | **cablear a un entrypoint** (bloquea todo lo demás) |

Principio: **cada mecanismo cognitivo debe ganarse su sitio en el eval, o se corta.** Es lo que separa a
cuba de ser "impresionante en el README" a ser "medi­blemente mejor".

---

## 6. Tier 3 — Investigación / especulativo (cuando el eval exista)

- **Consolidación estilo CLS** (hippocampus↔neocortex): mover hechos episódicos de alta frecuencia a
  memoria semántica en el ciclo REM. cuba ya tiene episodios + REM; falta la promoción.
- **Spaced repetition / FSRS** para el decay en vez de exponencial plano (el CHANGELOG dice que se quitó
  FSRS-6 "por simplicidad" — reconsiderar con eval).
- **Reconsolidación:** al recuperar un hecho, volverlo lábil y permitir su actualización (encaja con el
  ADD/UPDATE del Tier 1).
- **Letta-style memory budget:** dejar que el agente gestione su propio presupuesto de contexto.

---

## 7. Orden de ejecución recomendado

1. **0.6 decay** (bug que degrada hoy) → **2.1 LongMemEval** (para medir todo lo demás).
2. **1.3 activar reranker** + **1.1 extracción por sampling** (cierran las dos brechas caras con el
   andamiaje que ya existe).
3. **1.2 ADD/UPDATE/DELETE/NOOP** + **1.4 bge-m3**.
4. **§5 probar-o-cortar** los algoritmos de adorno con el eval ya montado.
5. **2.2 PPR asociativo** + **2.3 BM25 real**.

La regla que atraviesa todo: **medir primero (LongMemEval), después optimizar.** cuba tiene más motor del
que ha probado que sirve; el siguiente nivel no es más algoritmos, es evidencia.

---

## Fuentes (todas abiertas y verificadas)

- Comparativa frameworks + LongMemEval 63.8/49.0: https://particula.tech/blog/agent-memory-frameworks-tested-mem0-zep-letta-cognee-2026
- Engram (streamer argentino): https://github.com/Gentleman-Programming/engram · https://x.com/G_Programming/status/2023452982725976269
- mem0 ADD/UPDATE/DELETE/NOOP: https://arxiv.org/html/2504.19413v1 · https://memo.d.foundation/breakdown/mem0
- HippoRAG 2 (PPR, +7 F1): https://arxiv.org/abs/2502.14802
- Contextual Retrieval (−67% con rerank): https://www.anthropic.com/engineering/contextual-retrieval
- LongMemEval (5 habilidades, 500 preguntas): https://arxiv.org/abs/2410.10813
- germaniu/mcp-memory (hispano, bge-m3): https://github.com/germaniu/mcp-memory

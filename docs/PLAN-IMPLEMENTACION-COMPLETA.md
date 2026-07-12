# Plan de implementación completa — cuba-memorys

Inventario **unificado** de todo lo investigado sobre cuba-memorys, con el estado
verificado de cada punto. Nace de dos investigaciones que quedaron desconectadas:

- **Plan A — Engram / DX** (sesión `6a802934`, 2026-07-06). Comparativa a fondo contra
  [Gentleman-Programming/engram](https://github.com/Gentleman-Programming/engram): 37 hallazgos → 16 mejoras.
  [Reporte completo](https://claude.ai/code/artifact/a37e3e8f-2e3e-4926-9ea3-8d67f68be511).
  Se pidió implementarlo entero; la sesión murió tras completar **1 de 14** ramas.
- **Plan B — retrieval / evidencia** (sesión `6b2aee9d`, 2026-07-10, transcript perdido).
  Ecosistema (Engram, mem0, Zep/Graphiti, HippoRAG) → [`PLAN-MEJORAS-v0.11.md`](PLAN-MEJORAS-v0.11.md).
  Fases 1-5 en `main`; el resto vive en `lab/full-integration`, sin mergear.

El Plan B **no absorbió** al Plan A (cero menciones de doctor / Obsidian / RBAC en v0.11).
Son ejes complementarios: el B hace que cuba **recuerde mejor**; el A hace que **se pueda ver y operar**.

## Tesis de la investigación (no re-litigar)

Engram **no gana** en retrieval ni en ML: no tiene embeddings reales (columna BLOB reservada,
nunca usada), su búsqueda "semántica" es un subprocess LLM comparando pares de texto, y su
"knowledge graph" es solo un export de wikilinks a Obsidian — cero PageRank, cero clustering.
cuba ya le gana ahí (pgvector + BM25/RRF + Hebbian/BCM + Louvain + PageRank + bitemporal).

Donde **sí** gana es en superficie humana: cuba-memorys hoy es **invisible salvo para un agente
de IA**. No hay forma de ver qué sabe el sistema sobre vos sin pedirle a un LLM que lo pregunte.

**Explícitamente NO se hace** (decidido en la investigación, no reabrir):
- No migrar a SQLite / storage embebido — se pierde pgvector, RLS, concurrencia real.
- No copiar la "detección semántica" de Engram — es peor que lo que ya hay.
- No tocar la distribución multiplataforma — npm/pip/wheels ya funcionan.

## Entorno de pruebas (este sandbox)

| | Producción (INTOCABLE) | Sandbox |
|---|---|---|
| Contenedor | `cuba-memorys-db` | `cuba-sandbox-db` |
| Puerto | 5488 | **5491** |
| Volumen | `cuba_memorys_data` | `cuba_sandbox_data` |
| Binario | `cuba-memorys/rust/target/release/` (3 procesos MCP vivos) | `cuba-memorys-sandbox/rust/target/release/` |
| Datos | `brain` en vivo | copia del backup known-good verificado |

`sandbox/env.sh` aborta si algo apunta al puerto 5488 o al binario vivo.
`sandbox/verify-prod-intact.sh` se corre tras cada prueba: si un conteo de `brain` baja, falla.

---

## Plan A — 16 mejoras (DX y superficie humana)

### Diagnóstico y mutaciones seguras

- [ ] **A1 · `cuba doctor` — chequeo de salud de solo lectura.**
  La idea más portátil de toda la investigación (tres agentes la señalaron por separado).
  Verifica: conexión a la DB, migraciones aplicadas, extensiones (`vector`, `pg_trgm`), modelo ONNX
  cargable, dimensión del embedding vs. la de la columna, observaciones sin vector, invariante
  bitemporal, rol no-superuser, tamaño de la base. Sale con código ≠ 0 si algo está roto.
  *Por qué importa acá:* los cuatro bugs críticos de v0.10 (recall cero por `tsquery` AND, rama
  vectorial muerta sin ONNX, OOD invertido, sesión global entre procesos) eran **todos silenciosos**.
  Un `doctor` los habría cazado el primer día.

- [ ] **A2 · Reparaciones en tres niveles: `plan` → `dry-run` → `apply`, con backup automático.**
  Toda mutación destructiva (reembed, dedup, consolidación, purga) primero muestra qué haría,
  luego lo simula, y solo entonces lo aplica — tomando un backup antes.

- [ ] **A3 · Aviso de versión nueva, sin auto-update.** Compara la versión del binario contra la
  última publicada y avisa. Nunca se actualiza solo.

### Superficie humana

- [ ] **A4 · CLI directo: `search` / `save` / `delete`.** Poder consultar y escribir memoria desde
  la terminal, sin un agente de por medio. El patrón ya existe: `main.rs` despacha el subcomando `eval`.

- [ ] **A5 · Exportador a Obsidian (markdown + wikilinks).** Visualización gratis del grafo que
  cuba **ya calcula** (Louvain + PageRank), sin infraestructura nueva. Engram exporta un grafo que
  no tiene; cuba tiene un grafo que no exporta.

- [ ] **A6 · Dashboard web local read-only, o TUI.** El más ambicioso del bloque. Ver entidades,
  observaciones, comunidades y salud sin escribir una query.

### Onboarding multi-agente

- [ ] **A7 · `cuba setup <agente>` — registro declarativo.** Escribe la config del MCP en el cliente
  que toque (Claude Code, Codex, Cursor…) resolviendo la ruta absoluta del binario.
- [ ] **A8 · Regla escrita: "adaptador delgado, núcleo gordo".** La lógica vive en el núcleo; cada
  cliente solo aporta un adaptador fino. Documento, no código.

### Sync entre máquinas

- [ ] **A9 · Chunks inmutables con ID por contenido.** Sin conflictos de merge. Relevante: Leandro
  trabaja desde 3+ máquinas. `cuba_sync` ya existe y ya es dimension-aware (Fase 5).
- [ ] **A10 · Abstracción de `Transport`, desacoplada del backend.**

### Equipo (opcional, a futuro)

- [ ] **A11 · RBAC mínimo real: principals × rol × grant por proyecto.**
- [ ] **A12 · Lección de diseño: "scope" no es una frontera de privacidad si es solo un filtro de
  búsqueda.** Documento. Importa antes de compartir con nadie.

### Higiene

- [ ] **A13 · Borrar el Python huérfano.** `src/cuba_memorys/` es la implementación previa a la
  reescritura a Rust (marzo 2026): sin entry point en `pyproject.toml`, maturin nunca la empaqueta.
  **Ya hecho** en `chore/remove-dead-python-legacy` (commit `43888b1`) — falta mergear.
- [ ] **A14 · Perfiles de tools: agente vs. admin.** Enlaza con B3 (reducir la superficie de 25 tools).
- [ ] **A15 · Confirmar la higiene del pipeline LLM-judge de `cuba_juez`.**
- [ ] **A16 · Documentar el "por qué" del stack más pesado.** Postgres+pgvector+ONNX contra el
  binario+SQLite de Engram: la decisión es correcta, pero no está escrita en ningún lado.

---

## Plan B — pendientes de retrieval / evidencia

- [ ] **B1 · Mergear `lab/full-integration`** (5 commits, verificados en aislamiento, sin mergear):
  bug 0.7 (rol no-superuser: RLS y audit-log hoy son inertes porque la app corre como superuser),
  Fase 1.2 (ADD/UPDATE/DELETE/NOOP explícito estilo mem0), Fase 5 (embeddings model-agnostic + bge-m3),
  Fase 2.1b (taxonomía LongMemEval en el harness de eval).
  Medido en el lab: **nDCG@10 0.7344 (e5) → 0.9292 (bge-m3), +19.5 puntos**.
- [ ] **B2 · Contextual Retrieval completo** (Tier 1.3 de v0.11).
- [ ] **B3 · Reducir la superficie de 25 tools** (Tier 1.5). Enlaza con A14.
- [ ] **B4 · BM25 real** (ParadeDB/pg_search) en vez de `ts_rank_cd` (Tier 2.3).
- [ ] **B5 · Query rewriting / HyDE por sampling** (Tier 2.4).
- [ ] **B6 · Migrar `brain` viva a bge-m3.** Es **ops** y toca datos: cambiar `~/.mcp.json`, correr
  `scripts/migrate-embedding-dim.sh`, reembeber. Requiere aprobación explícita y parar los MCP vivos.

---

## Regla de oro (de la propia investigación)

> Cada mecanismo se gana su sitio en el eval o se corta.

Hebbian/BCM, Louvain, OOD, energy_score suenan sofisticados pero **ninguno tiene evidencia medida**
de que mejore el retrieval. El harness (`cuba-memorys eval`) ya es ejecutable y no muta la base:
toda mejora de este plan que toque retrieval se mide antes y después, o no entra.

# Análisis de rendimiento y estado del arte — agosto 2026

Qué se midió, qué dice el SOTA actual, y qué de ese SOTA sirve aquí de verdad.
Todo lo etiquetado como MEDIDO viene de `EXPLAIN ANALYZE` contra la base viva
(`:5488`, 1736 observaciones) o del evaluador propio. Lo demás está marcado como
hipótesis a verificar.

## 1. El estado del arte, y qué se puede robar de él

### DeepSeek V4 (24-abr-2026) — V4-Pro 1,6T / 49B activos, V4-Flash 284B / 13B

La arquitectura combina **CSA** (Compressed Sparse Attention) y **HCA** (Heavily
Compressed Attention). El número que importa: a 1M de contexto, V4-Pro gasta el
**27% de los FLOPs y el 10% del KV cache** de V3.2.

La matemática de fondo viene de MLA (V3): en vez de guardar K y V completos por
token, se proyectan conjuntamente a un espacio latente de rango bajo mediante una
matriz de compresión, y se expanden al vuelo. Es una factorización de rango bajo
—la misma idea que LoRA— aplicada a la caché en lugar de a los pesos. Resultado
en V3: 70 KB/token frente a 192-328 KB/token de los modelos con GQA, entre 2,7× y
4,7× menos.

**La lección trasladable no es la atención.** Es esta: *la precisión completa casi
nunca hace falta para decidir qué merece atención; hace falta solo para el
resultado final.*

### Engram — la pieza de memoria

Es lo más cercano a lo que hace cuba-memorys. Engram separa dos ejes de dispersión:
MoE resuelve "cómo calcular menos" (cómputo condicional), **Engram resuelve "no
calcular a ciegas"**: hashing multi-cabeza sobre n-gramas que mapea directo a tablas
de embeddings, con recuperación en tiempo aproximadamente **constante**.

La propiedad que lo hace barato: **los índices dependen solo de los tokens de
entrada, no de las activaciones**. Eso permite prefetch asíncrono y descargar hasta
100B de parámetros a CPU/SSD con menos del 3% de sobrecoste. Su "Sparsity
Allocation Law" reparte 20-25% de los parámetros dispersos a memoria y el resto a
cómputo.

Nota honesta: Engram aparece en la investigación previa de DeepSeek y en el
material de V4, pero hay fuentes que señalan que **no forma parte de la
arquitectura confirmada de V4**. Se toma aquí como patrón de diseño, no como
"lo que hace DeepSeek en producción".

### Kimi K3 (27-jul-2026, arXiv 2607.24653) — 2,8T parámetros

Dos piezas:

- **KDA + Gated MLA en ratio 3:1.** Atención lineal barata para la mayoría de las
  capas, atención completa cara solo en una de cada cuatro. En Kimi Linear esto
  daba −75% de KV cache y hasta 6× de throughput de decodificación a 1M de contexto.
- **AttnRes (Attention Residuals).** En vez de que cada capa lea solo de la
  inmediatamente anterior, cada capa **recupera selectivamente** de capas previas.
  Acorta el camino de la información en modelos profundos.
- **Stable LatentMoE**: 896 expertos, 16 activos por token (≈1,8% de dispersión),
  estabilizado con normalización, SiTU-GLU y Quantile Balancing.

Global: ≈2,5× de eficiencia de escalado sobre K2.

El optimizador **MuonClip** (Newton-Schulz + QK-clip, ~2× de eficiencia de cómputo
sobre AdamW) **no aplica aquí**: no entrenamos modelos. Se descarta explícitamente.

### Lo que se traduce, punto por punto

| Técnica SOTA | Mecanismo | Traducción a cuba-memorys | ¿Sirve? |
|---|---|---|---|
| MLA / compresión de rango bajo | proyección a espacio latente | `halfvec` fp16, −50% almacenamiento | **Sí**, tarea P1 |
| FP8 de entrenamiento | menos bits por número | cuantización de vectores | **Sí**, misma tarea |
| CSA + HCA (V4) | barato filtra, caro decide | cascada binaria → fp16 → cross-encoder | **Sí**, tarea P2 |
| KDA 3:1 (K3) | híbrido barato/caro por proporción | ya existe: RRF → top-50 → cross-encoder | **Ya implementado** |
| Engram | lookup O(1), índice solo de la entrada | caché por hash(consulta+corpus) | **Sí**, tarea P2 |
| AttnRes | recuperación selectiva por profundidad | expansión asociativa del grafo selectiva, no uniforme | **Quizá**, sin medir |
| LatentMoE 896/16 | enrutar a subconjunto | particionar corpus por proyecto/tema | **No aún**: 1736 filas es poco |
| MuonClip / Muon | optimizador de entrenamiento | — | **No aplica** |

## 2. Lo que está mal aquí y cuesta caro

### 2.1 El índice HNSW de 13 MB nunca se usa (MEDIDO)

```
Seq Scan on brain_observations (rows=1736)
  actual time=1.505..96.249    Buffers: shared hit=20740 read=1773
Execution Time: 96.727 ms
```

Forzando el índice:

```
Index Scan using idx_obs_embedding_hnsw
  actual time=0.867..1.039     Buffers: shared hit=620
Execution Time: 0.873 ms
```

**14× con caché caliente, 111× desde frío, 36× menos E/S.**

Causa: `random_page_cost = 4.0`, el valor por defecto pensado para discos
giratorios. El planificador estima el HNSW en 8878 frente a 524 del seq scan.
Con `random_page_cost = 1.1` (NVMe) elige el índice **solo, sin forzar nada**.

### 2.2 La base nunca se analizó (MEDIDO)

`last_analyze`, `last_autoanalyze`, `last_vacuum`, `last_autovacuum`: **NULL** en
las tres tablas principales. El planificador creía 20 filas donde hay 1736, 2
entidades donde hay 279, 0 relaciones donde hay 228.

### 2.3 Cada observación se embebe dos veces, y el dedup compara peras con manzanas

`check_dedup()` calcula `embed_passage(content)` para comparar; el insert vuelve a
calcular `embed_passage_contextual(...)` para almacenar. El embedding es la
operación más cara del sistema.

Peor que el derroche: **los dos vectores no son comparables**. El dedup compara un
vector sin prefijo contextual contra vectores almacenados con prefijo. Son
distribuciones distintas del mismo espacio; el umbral semántico no mide lo que
cree medir. Es el mismo fallo que ya obligó a recalibrar el gate OOD.

### 2.4 ~300 round-trips por lote de 100

Por ítem: `ensure_entity` (1-2 queries) + un `SELECT entity_type` sobre la fila que
`ensure_entity` acaba de tocar (**redundante**) + `check_dedup` (2 queries + 1
embedding). Además `similarity(content,$2)` aparece dos veces en la misma consulta,
en el `SELECT` y en el `WHERE`.

### 2.5 El semáforo del reranker no hace nada

`Semaphore::new(2)` promete dos rerankeos en paralelo, pero la sesión ONNX está tras
un `Mutex` que se toma durante todo el rerankeo. Dos consultas concurrentes se
serializan igual.

### 2.6 29 MB de índices, 11 de 15 sin un solo escaneo

Parte es consecuencia de 2.1 —con `random_page_cost=4` ningún índice compite—, así
que **no se borra nada hasta arreglar eso y volver a medir**.

## 3. Orden recomendado

1. **P0 2.1 + 2.2** — una línea de configuración y un ANALYZE. Es el mayor
   rendimiento por esfuerzo de toda la lista.
2. **P0 2.3** — corrige un bug de corrección, no solo de velocidad.
3. **P1 2.4, 2.6, halfvec** — tras volver a medir con el planificador ya sano.
4. **P2 cascada binaria, caché Engram, concurrencia del reranker.**

Con 1736 observaciones, varias de las técnicas del SOTA están sobredimensionadas:
se diseñaron para 1M de tokens de contexto y billones de parámetros. La cascada
binaria y el particionado del corpus probablemente no compensen **todavía**;
conviene revisarlas a partir de ~50k observaciones. Decirlo ahora es más útil que
implementarlas y descubrirlo después.

## Fuentes

- [DeepSeek-V3 Technical Report (MLA)](https://arxiv.org/pdf/2412.19437)
- [DeepSeek V4 Technical Documentation, 27-abr-2026](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- [DeepSeek V4: arquitectura, benchmarks, precios](https://www.morphllm.com/deepseek-v4)
- [DeepSeek Engram — V4 Memory Architecture](https://deepseek.ai/blog/deepseek-engram-v4-architecture)
- [Engram: conditional memory, second sparsity axis](https://medium.com/@graison/engram-explained-deepseeks-conditional-memory-adds-a-second-sparsity-axis-512cdfaaf93f)
- [Kimi K3 Technical Report (arXiv 2607.24653)](https://arxiv.org/pdf/2607.24653)
- [Kimi K3 Tech Blog](https://www.kimi.com/blog/kimi-k3)
- [Kimi Linear: KDA architecture (arXiv 2510.26692)](https://arxiv.org/pdf/2510.26692)
- [Kimi K2: Open Agentic Intelligence (arXiv 2507.20534)](https://arxiv.org/abs/2507.20534)
- [Scalar and binary quantization for pgvector — Jonathan Katz](https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/)
- [Don't use vector, use halfvec — Neon](https://neon.com/blog/dont-use-vector-use-halvec-instead-and-save-50-of-your-storage-cost)
- [Scaling vector search in Postgres — ClickHouse](https://clickhouse.com/resources/engineering/scale-vector-search-postgres)

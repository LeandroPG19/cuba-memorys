# Changelog

All notable changes to cuba-memorys are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows
[SemVer](https://semver.org/) for the Rust crate (`Cargo.toml`). PyPI
versioning is independent (~ +1.0 offset since v0.6.0 era to allow wheel
revisions without binary changes).

## [0.20.0] — 2026-07-29 (Cargo `0.20.0` · npm `0.20.0` · PyPI `1.22.0`)

El daemon dejó de ser algo que corre siempre. En la GPU de 6 GB donde se midió
esto pasó de retener **5228 MiB de VRAM desde el arranque** —el 93% de la
tarjeta— a **1470 MiB mientras busca y 0 en reposo**. Ese 93% era la razón por la
que otros programas de GPU dejaban de arrancar: el driver devolvía
`NV_ERR_NO_MEMORY` al crear un canal, que es exactamente donde falla un juego o
una terminal acelerada.

Dos columnas, porque no todo viene gratis: la primera es lo que trae el código
publicado; la segunda añade una variable de entorno y el reranker refusionado que
documenta el README.

| | antes | **defaults 0.20.0** | + `CUBA_RERANK_CHUNK=4` y artefacto fusionado |
|---|---|---|---|
| VRAM buscando | 5228 MiB | **2950 MiB** | **1460 MiB** |
| VRAM en reposo | 5228 MiB | **0** — el proceso no existe | 0 |
| Arranque hasta responder | 11,1 s | **0,027 s** | 0,027 s |
| Búsqueda en caliente | 5,90 s | 5,25 s | **1,70 s** |
| Embedding de una query | 52,3 ms | **35,8 ms** | 35,8 ms |

Ninguna función se quitó.

### Colocación por modelo, no por proceso

`gpu::configure()` registraba CUDA para las tres sesiones ONNX. El README decía
que sin `--features cuda` «todos los modelos corren en CPU», dando a entender que
con el feature los tres irían a la GPU. **Solo el reranker fue nunca.**

- **El embedder no puede usar CUDA.** Viene cuantizado dinámicamente a INT8: 96
  `DynamicQuantizeLinear` alimentando 144 `MatMulInteger`, y el provider CUDA no
  registra kernel para ninguno de los dos (verificado contra el `.so` instalado y
  documentado por Optimum: *«nodes such as MatMulInteger and
  DynamicQuantizeLinear … cannot be consumed by the CUDA execution provider»*).
  ONNX Runtime los particionaba a CPU de todos modos. Registrar CUDA solo
  reservaba un arena en el que el modelo nunca computó: **374 MiB retenidos
  mientras los 544 MB de pesos vivían en RAM del host**, midiendo la sesión
  aislada.
- **El NLI tiene el problema opuesto.** Es FP32 y ahí se queda: mDeBERTa está
  documentado aguas arriba como no compatible con FP16, y la build INT8 devuelve
  entailments falsas con confianza (ya estaba anotado en `nli.rs`). Se invoca
  poco y tolera latencia, así que en CPU cuesta 150-400 ms medidos y libera más
  de un gigabyte de VRAM.
- **Corolario:** cuantizar el reranker a INT8 sería contraproducente — lo
  expulsaría de la GPU igual que al embedder. Su FP16 es la representación
  correcta.

Nuevas `CUBA_EMBED_DEVICE` / `CUBA_RERANK_DEVICE` / `CUBA_NLI_DEVICE` (`cpu` ·
`gpu` · `cpu`) para medir una colocación sin recompilar. `doctor` ahora informa
cuál corre dónde, porque tener GPU no dice nada sobre qué sesiones la usan.

### El arena de CUDA dejó de duplicarse

`ArenaExtendStrategy::NextPowerOfTwo` es el default de ONNX Runtime y reserva en
potencias de dos en lugar de lo que la sesión pidió. Así 1,65 GB de pesos se
convertían en 5+ GB. Ahora se fija `SameAsRequested` con un tope explícito
(`CUBA_GPU_MEM_LIMIT_MB`, 2048 por defecto). **El tope es por sesión**, que es
sostenible solo porque exactamente un modelo pide CUDA.

`CUBA_RERANK_CHUNK` (16) expone el otro extremo: bajo `fixed_shape` cada lote se
rellena a 512 tokens, así que es la palanca principal sobre el arena — 16 → 2938
MiB, 4 → 2364 MiB. Los scores no cambian: una búsqueda `verbose` a 16 y a 4
volvió byte a byte idéntica.

### El reranker carga en su primer lote

Calentar solo el embedder cuesta 0,026 s; calentar los dos, 11 s — el
cross-encoder son 1,08 GB y su warm-up corre un lote real de 50 candidatos. Bajo
activación por socket el daemon arranca muchas más veces de las que rerankea, y
buena parte de esos arranques solo atienden un `save`. `CUBA_WARM_RERANKER=1`
restaura la precarga.

### Un daemon que no corre cuando nadie pregunta

`CUBA_IDLE_SHUTDOWN_SECS` apaga el daemon tras ese tiempo sin peticiones de
ningún cliente. `serve` adopta el socket que systemd pasa como fd 3
(`LISTEN_FDS`), de modo que una unidad `.socket` retiene el puerto mientras el
daemon no corre y ningún cliente ve una conexión rechazada.

Se apaga por el camino normal —`serve` retorna, el drenaje de fondo vacía las
escrituras de embeddings en vuelo, `sqlx` cierra su pool— y no llamando a
`process::exit`, que se saltaba ese drenaje y perdía esas escrituras en silencio.
El README documenta el par de unidades systemd.

### Hilos, pool y arranque

- **`CUBA_EMBED_INTRA_THREADS`.** Estaba fijo en 2, de cuando se esperaba que
  CUDA hiciera el trabajo. Nunca lo hizo, así que esos hilos *son* el embedder.
  Medido en 12 hilos por query: 1 → 94,8 ms · 2 → 52,3 ms · **4 → 35,8 ms** · 6 →
  68,1 ms · 12 → 155,4 ms. Pasado medio núcleo lógico la sincronización cuesta
  más de lo que la paralelización aporta.
- **`with_intra_threads` del reranker baja a 2 en GPU.** Ahí los GEMM corren en
  kernels CUDA y esos hilos solo mueven tensores.
- **`with_memory_pattern(false)`** en embedder y NLI: el planificador de memoria
  de ONNX Runtime solo rinde con shapes estáticos, y la longitud de entrada varía
  en cada llamada.
- **`with_intra_op_spinning(false)`** en el NLI: los veredictos llegan de uno en
  uno, con minutos entre medias.
- **Pool de Postgres 10 → 4.** Era el default de sqlx, nunca ajustado; un proceso
  atiende ahora a todos los clientes en lugar de uno por ventana.
- **`worker_threads = 4`** en tokio (era uno por núcleo, 12) — el trabajo real va
  a `spawn_blocking` y el default hacía competir al runtime con los pools
  intra-op de ONNX.
- **`fixed_shape()`** se decide por la colocación real y no por el feature de
  compilación, para que apuntar el reranker a CPU no lo deje rellenando todo a
  512 tokens.

### Corregido

- **`scripts/backup-db.sh` y `scripts/restore-db.sh` estaban rotos desde
  `e96df5d`.** Aquel commit de estilo quitaba comentarios y trató `${#OLD[@]}` y
  `$#` como el inicio de uno, comiéndose el resto de la línea: `if ((${` y
  `if [[ $`. El gate de merge fallaba con error de sintaxis de bash, y el script
  de restauración —el camino de recuperación ante desastre— no arrancaba. Ambas
  líneas restauradas a su forma original.

## [0.19.0] — 2026-07-29 (Cargo `0.19.0` · npm `0.19.0` · PyPI `1.21.0`)

Un servidor para todos los clientes, y el reranker de v0.18.0 aplicándose de
verdad en la máquina donde estaba instalado.

### `serve`: un proceso en lugar de uno por ventana

stdio da a cada cliente su propio proceso, y a cada proceso su propia copia de
los modelos. Con embeddings + reranker + NLI eso son ~6 GB **por ventana de
editor**; tres sesiones abiertas se comían 18,5 GB de un portátil de 16 GB, y el
escritorio entero acababa en swap.

`cuba-memorys serve` carga los modelos una vez y atiende a todos por HTTP en
loopback — que es además la forma que fijó la [especificación MCP del
2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/): núcleo sin
estado, sin handshake de sesión, cada petición se describe a sí misma.

- **Las sesiones se aíslan por cliente.** `session.rs` guardaba la sesión activa
  en un `static` global; compartido, un `jornada start` en una ventana se
  convertía en la sesión activa de las demás. Ahora resuelve contra un
  task-local con la identidad del cliente (`Mcp-Client-Id`), y el global sigue
  siendo el camino de stdio y de los subcomandos. Fuera de una petición —el
  ciclo REM, una tarea de fondo— el daemon no responde ninguna sesión en vez de
  responder la de otro.
- **El ciclo REM corre una vez.** Bajo stdio cada ventana consolidaba la misma
  base en paralelo.
- **Un panic ya no se lleva a todos por delante.** El perfil release pasa a
  `panic = "unwind"` y el transporte contiene cada petición, porque con un
  proceso compartido un fallo en el handler de un cliente abortaba el servidor
  de todos. `outbound()` devolvía `.expect()` — con `panic = "abort"`, eso era
  el proceso entero al primer mensaje iniciado por el servidor sobre HTTP.
- **stdio ya no deja procesos huérfanos.** Si los modelos tardan más que el
  timeout de conexión del cliente (30 s), el cliente se rinde pero *no* cierra
  nuestro stdin: el proceso se quedaba vivo sujetando cada modelo que había
  cargado, uno por intento, hasta que la máquina no daba más. Ahora sale si no
  llega ningún handshake en `CUBA_HANDSHAKE_TIMEOUT_SECS` (60 s; `0` desactiva).

### El reranker: lo que faltaba después de v0.18.0

v0.18.0 dejó el camino GPU listo. Lo que no comprobaba nadie es que **el binario
instalado se hubiera compilado con `--features cuda`** — el de esta máquina no,
así que el cross-encoder corría en CPU, se pasaba de su presupuesto de 20 s en
cada consulta y `faro` descartaba los scores. Se pagaba la inferencia completa
para devolver el orden de RRF. Medido con `examples/rerank_bench` (50
candidatos, longitudes mixtas como las reales):

| configuración | media | ¿dentro del presupuesto? |
|---|---|---|
| CPU, 2 hilos (lo que había) | 106,9 s | no — 5,3× por encima |
| CPU, 6 hilos (núcleos físicos) | 61,0 s | no |
| **GPU (`--features cuda`)** | **4,1 s** | **sí** |

El ranking de GPU y el de CPU son idéntico candidato a candidato; solo difieren
en la quinta cifra decimal del score.

Tres defectos reales aparecieron por el camino:

- **Reordenar una lista vacía cargaba el modelo entero.** La comprobación de
  `enabled()` iba antes que la de lista vacía, así que `rerank(q, &[])` pagaba
  una carga perezosa de 1,1 GB para devolver el vector vacío que ya sabía
  devolver. Era también lo que colgaba dos tests de la suite.
- **`enabled()` se llamaba desde una tarea async.** Puede cargar el modelo, así
  que bloqueaba el executor: en el daemon, la primera búsqueda de un cliente
  congelaba a todos los demás. Ahora va al pool de bloqueo junto con la
  inferencia que controla.
- **`with_intra_threads(2)` estaba fijo** para un XLM-RoBERTa-large. Ahora son
  los núcleos físicos (`CUBA_RERANK_INTRA_THREADS` lo fuerza): 1,74× en CPU, y
  medido que pasarse a los hilos SMT empeora (12 hilos → 16,5 s vs 10,8 s con 6).
- **Los lotes se agrupan por longitud** (`CUBA_RERANK_LENGTH_BUCKETING`) cuando
  el padding no es fijo, para no gastar cómputo en relleno. Los scores no
  cambian: la máscara de atención ya anula las posiciones rellenadas.

`examples/rerank_bench` mide todo esto en cualquier máquina y dice si el
reranker cabe en su presupuesto o si se está tirando el trabajo.

## [0.18.0] — 2026-07-28 (Cargo `0.18.0` · npm `0.18.0` · PyPI `1.20.0`)

Dos cosas que llevaban releases documentadas como pendientes: el reranker que
medía +92% y estaba apagado, y un codegraph que se corrompía solo al segundo
build.

### El reranker en GPU: +93% nDCG por 1,1 s de latencia

El cross-encoder costaba ~15 s por consulta en CPU, así que nunca se activaba.
El crate ya tenía la feature `cuda`, `gpu::configure()` cableado en los tres
modelos y el provider CUDA descargado — **nunca se había compilado con ello**.

Compilar con `--features cuda` no bastaba. Tres cosas separaban el camino GPU
de uno usable:

- **El modelo se cargaba dentro del handler de la primera búsqueda**, que tiene
  30 s. Cargar más la primera inferencia se los pasaba de largo, así que la
  primera búsqueda tras cada arranque moría con `-32603`. Ahora el servidor
  precalienta el reranker en una tarea de fondo al arrancar, fuera de todo
  handler.
- **Precalentar con un pasaje corto no servía de nada**: ONNX Runtime compila
  kernels CUDA *por forma de entrada*, y el camino real son 50 candidatos.
  Precalentar con la forma real saca ese coste de la ruta de consulta.
- **El tokenizador rellenaba con `BatchLongest`**, así que cada consulta
  generaba formas de tensor nuevas y volvía a pagar la compilación. Rellenar a
  512 fijo con lotes de tamaño constante bajó el precalentamiento de **42,8 s a
  8,9 s**. Es una ganancia exclusiva de GPU — en CPU un 512 constante solo hace
  cada lote más grande —, así que por defecto se activa únicamente en builds con
  `cuda`/`directml`, y `CUBA_RERANK_FIXED_SHAPE` lo fuerza en cualquier sentido.

Medido sobre 60 preguntas contra el corpus real, k=10:

| métrica | sin rerank | con rerank | cambio |
|---|---|---|---|
| nDCG@10 | 0,3039 | **0,5873** | **+93%** |
| MRR | 0,2366 | **0,4914** | +108% |
| R@10 | 0,4346 | **0,6961** | +60% |
| tokens/respuesta | 5252 | **4026** | **−23%** |

La ganancia (+0,283) supera el efecto mínimo detectable que el propio evaluador
calcula (0,245), así que no es ruido. La mediana por consulta pasa de 4,23 s a
5,35 s; ese mismo trabajo costaba ~15 s en CPU.

### codegraph: el segundo build ya no corrompe el grafo

Tres defectos que solo aparecen al reconstruir un repo que ha cambiado — que es
justo el único caso que importa en una herramienta que sigue código vivo:

- **El contenido de la observación llevaba dentro el rango de líneas del símbolo,
  y el dedup comparaba esa cadena exacta.** Añade un comentario encima de una
  función y su rango se desplaza: el `WHERE NOT EXISTS` no encuentra nada, entra
  una fila nueva y la vieja se queda para siempre. **Cada edición multiplicaba las
  filas.** La identidad ahora es `` {tipo} `{nombre}` in {fichero}: `` —sin números
  de línea— comparada con `left(content, n)` para no depender de escapes de `LIKE`,
  y un símbolo que se mueve actualiza su fila en lugar de duplicarla.
- **El recorrido usaba `path.is_dir()`, que sigue enlaces simbólicos.** Un enlace
  a un directorio ancestro y el recorrido no termina nunca. Ahora usa
  `entry.file_type()`, que no los sigue, y registra el salto para que un árbol
  enlazado se vea omitido en vez de desaparecer en silencio.
- **La persistencia eran cientos de sentencias sueltas contra el pool.** Un fallo
  a media faena dejaba el grafo a medio escribir sin forma de saberlo. Las tres
  pasadas comparten ahora una transacción y confirman juntas.

306 tests en verde, clippy sin avisos.

## [0.17.1] — 2026-07-28 (Cargo `0.17.1` · npm `0.17.1` · PyPI `1.19.1`)

Correcciones encontradas ejecutando el ciclo REM de v0.17.0 contra el corpus real
de producción, no contra una base de prueba. El escaneo funcionaba en los tests
porque el test llama a `scan_entity_relations()` directamente con una entidad de
tres notas; sobre entidades reales el prompt lleva hasta 12 observaciones más 60
nombres del grafo, y ahí se rompía.

- **El escaneo de relaciones heredaba el presupuesto del handler MCP.** El ciclo
  REM no corre dentro de un handler y no tiene por qué respetar sus 30 s, pero
  `extraction_budget()` le imponía el 60% de ese límite: 18 s para un prompt
  varias veces más largo que el de `auto_extract`. Ahora el escaneo tiene su
  propio presupuesto (`CUBA_REM_SCAN_TIMEOUT_SECS`, por defecto 90 s).
- **Los dos timeouts no estaban alineados.** `ClaudeCodeJudge` lleva su propio
  corte interno de 30 s, así que el presupuesto externo nunca llegaba a aplicarse
  y el CLI moría antes. `resolve_offline_llm_within()` ajusta ahora el timeout del
  backend al presupuesto de quien lo llama, de modo que manda un solo número.
- **Un fallo aislado abortaba el lote completo.** El bucle hacía `break` al primer
  error, así que una entidad lenta se llevaba por delante a las cuatro restantes
  del ciclo. Ahora tolera fallos sueltos y solo abandona tras dos consecutivos;
  el recuento se reporta en `failed`.

Medido en producción antes y después del arreglo: **4 de 5 entidades escaneadas
con 1 fallo → 5 de 5 con 0 fallos**. Acumulado sobre el corpus real: relaciones
213 → 228 (15 de ellas `provenance='inferred'`), entidades aisladas 148 → 141,
48 observaciones largas por fin fragmentadas en chunks.

302 tests en verde, clippy sin avisos.

## [0.17.0] — 2026-07-28 (Cargo `0.17.0` · npm `0.17.0` · PyPI `1.19.0`)

v0.16.0 prometió que el grafo crecía solo. No crecía: la capacidad estaba ahí y
nunca llegaba a ejecutarse. Esta versión la hace funcionar de verdad y aplica lo
mismo al corpus que ya existía.

### La extracción automática nunca se había ejecutado

Consultando producción por origen: 1571 observaciones `source='agent'`, 74
`source='user'` y **cero** `source='inference'`. Ese último número es la prueba:
`source='inference'` es lo único que escribe `auto_extract`, y `auto_extract` es la
única ruta que crea relaciones automáticamente. Cero filas en 1645 observaciones
significa que nunca corrió.

La causa era una puerta cerrada: `auto_extract` devolvía `degraded` salvo que el
cliente anunciara `capabilities.sampling`. **Ningún cliente real lo anuncia** —
verificado en vivo contra Claude Code 2.1.220, que respondió *"client did not
advertise MCP sampling capability"*. La función estaba fuera de alcance desde todos
los clientes en uso.

- **`auto_extract` ahora escala igual que `cuba_juez`**: sampling si el cliente lo
  ofrece, si no un CLI local, si no la API. Ese escalonado ya existía en
  `resolve_llm_judge()` y llevaba releases funcionando; `auto_extract` simplemente
  no lo llamaba. La rama de sampling se mantiene aparte porque
  `MCPSamplingJudge::run_prompt` corta en 256 tokens y la extracción necesita 1024.
- **La respuesta ahora incluye `backend`**, para que se pueda ver qué LLM contestó
  en lugar de deducirlo.

### Dos fallos que solo aparecen bajo el protocolo real

Ambos pasaban desapercibidos llamando al handler directamente, que es como estaba
escrito el test:

- **El handler tiene 30 s y el CLI tardaba ~20 s**, así que el primer intento
  end-to-end moría con `-32603 Handler timed out`. El LLM recibe ahora una fracción
  fija del presupuesto (60%) y agotarla degrada a una respuesta normal en vez de
  tumbar la llamada entera.
- **`claude --print` cargaba toda la configuración MCP del usuario**, cuba-memorys
  incluido: un servidor lanzando un cliente que relanza el servidor.
  `--strict-mcp-config` lo corta — **9,48 s → 3,98 s (−58%)** y adiós recursión.
  `cuba_juez` comparte backend y hereda la mejora.

### El ciclo REM cablea también lo que ya estaba escrito

Arreglar `auto_extract` solo sirve para memorias nuevas. El corpus existente seguía
igual: 148 de 279 entidades (53%) con grado 0 y cero relaciones inferidas.

- **Nueva tarea REM**: cada ciclo toma un lote de entidades aisladas que tienen
  observaciones y pregunta al LLM qué conectan sus propias notas.
- Al prompt se le pasan **los nombres ya presentes en el grafo**, ordenados por
  grado, para que reutilice `PostgreSQL` en lugar de acuñar `Postgres` y fragmentar
  el grafo. Se le dice explícitamente que una lista vacía es respuesta válida,
  porque un modelo presionado a encontrar enlaces se los inventa.
- Las entidades se sellan con `relations_scanned_at` para no pagar dos veces por un
  escaneo que legítimamente no encontró nada. Pero un sello a secas congelaría el
  grafo, así que la cola devuelve también las entidades cuya observación más nueva
  es posterior al sello.
- El lote está acotado: `CUBA_REM_RELATION_BATCH` (por defecto 5, `0` lo desactiva).
  Cada entidad cuesta una llamada de CLI de ~4 s, así que un ciclo gasta ~20 s cada
  cuatro horas en vez de vaciar la cola de golpe.

### Migración

- **0039**: `brain_entities.relations_scanned_at` + índice parcial para la cola.
  Aditiva; se aplica sola al arrancar.

### Medido

Contra una base aislada, por JSON-RPC y sin sampling: 3 hechos extraídos y 3
relaciones escritas (`cuba-memorys --uses--> Rust`, `--uses--> PostgreSQL`,
`--depends_on--> pgvector`), todas con `source='inference'` y
`provenance='inferred'`. El escaneo retroactivo tomó una entidad huérfana con tres
notas y la dejó como `--depends_on--> PostgreSQL` y `--uses--> Rust`, sellada y
fuera de la cola; añadirle una nota posterior la devolvió a la cola.

302 tests en verde, clippy sin avisos.

## [0.16.0] — 2026-07-28 (Cargo `0.16.0` · npm `0.16.0` · PyPI `1.18.0`)

Esta versión sale de una comparación con el estado del arte (mem0, Zep/Graphiti,
Letta, cognee, HippoRAG 2, el survey de memoria de agentes arXiv 2602.06052) y de
medir el corpus real en vez de suponer.

### El grafo ahora crece solo

Medido antes del cambio: **148 de 279 entidades (53%) sin una sola arista**, grado
medio 1,53, 213 relaciones para 1645 observaciones. Siete algoritmos de grafo ya
implementados —PageRank personalizado, Louvain, k-core, closeness/harmonic,
betweenness, Adamic-Adar, activación por propagación estilo HippoRAG— corrían sobre
un sustrato que la mitad de las veces no existía.

La causa: nada creaba relaciones por su cuenta. `auto_extract` extraía hechos pero
nunca aristas, y `link` (NPMI) es un comando manual que nadie ejecuta.

- **`auto_extract` ahora pide hechos Y relaciones** en la misma llamada de sampling
  (mismo coste, $0), y las escribe con `provenance='inferred'`. Los extremos se
  auto-crean, así que una arista que menciona una tecnología vista por primera vez
  en esa conversación igual aterriza. El parser acepta tanto la forma nueva como el
  array plano anterior.
- **El ciclo REM pasó de 3 a 6 tareas**: además de decay, decay de episodios y
  PageRank, ahora hace autolink NPMI, backfill de embeddings y chunking. Esto es el
  *sleep-time compute* que Letta nombró en 2025 — cuba-memorys tenía el daemon desde
  antes, al 30% de su capacidad.
- **`cuba-memorys rem`** ejecuta un ciclo bajo demanda en vez de esperar 4 horas.
- **43 observaciones (2,6%) no tenían embedding** y eran permanentemente invisibles
  a la búsqueda vectorial, sin nada que las reprocesara. El ciclo REM las rellena,
  con tope por ciclo (`CUBA_REM_BACKFILL_LIMIT`, 100 por defecto).

Hallazgo honesto: el autolink NPMI creó **0** aristas, y es correcto — 46 de sus 47
pares candidatos ya tenían relación. La co-ocurrencia está saturada en este corpus;
lo que densificará el grafo es la extracción por LLM, no NPMI.

### Cuarentena contra envenenamiento de memoria

La memoria persistente tiene una clase de ataque que la inyección de prompt no
tiene: la escritura y su efecto están separados en el tiempo. MINJA (arXiv
2601.05504) planta memoria envenenada con turnos de usuario normales —sin
privilegios, sin acceso al almacén— y reporta **>95% de éxito**; la instrucción
dispara semanas después. El hash-chain CFR-21 prueba *a posteriori* que algo se
alteró; no impide que una escritura legítimamente autenticada meta un hecho hostil.

Siguiendo SMSR (arXiv 2606.12703): separar memoria candidata no confiable de la
memoria confiable, con promoción mediada.

- Migración 0037 añade `brain_observations.trust`, por defecto `trusted`, así que
  todo lo existente se comporta igual.
- `cuba_ingesta auto_extract` acepta `untrusted: true` para texto que no controlás.
  `CUBA_QUARANTINE_INFERENCE=1` aplica la política a toda extracción por LLM.
- Lo cuarentenado se almacena y es inspeccionable, pero **no se recupera**: ni por
  `cuba_faro` ni por BM25, y queda fuera de la calibración OOD para que el texto no
  confiable tampoco pueda mover el umbral de abstención.
- `cuba_eco` gana la mediación: `pending` lista lo retenido, `promote` lo hace
  recuperable, `quarantine` lo retira. Ambas transiciones quedan en la cadena de
  auditoría.

### Chunking: el final de los textos largos deja de ser invisible

El embebedor trunca a 512 tokens. Todo lo que pasa de ~1800 caracteres nunca
llegaba al modelo. Medido: **48 observaciones (2,9%) con ~29.700 caracteres
invisibles** a la búsqueda vectorial — y son las más densas (post-mortems, lecciones
detalladas, decisiones de arquitectura).

- Migración 0038 añade `brain_observation_chunks` con solapamiento. La columna
  vectorial se crea leyendo la dimensión que ya usa la base, no una fija: esta
  instalación corre bge-m3 a 1024 mientras las migraciones declaran 384.
- La búsqueda vectorial une los aciertos directos con los aciertos por chunk y
  deduplica a la observación padre.
- Medido: para una consulta que apunta al carácter 3047 de una observación de 3221,
  la similitud vía documento truncado es **0,6566** y vía el chunk que lo cubre,
  **1,0000**.

### BEAM, y los dos bugs que destapó en minutos

LOCOMO está saturado (16-26k tokens entran en cualquier ventana moderna) y Zep
documentó que cambiar el prompt del juez mueve su accuracy dos dígitos. BEAM (ICLR
2026) es el estándar al que se movió el campo. Se añade `beam_prepare.py`, que
convierte un shard de BEAM al JSONL del harness y mapea `source_chat_ids` a las
observaciones ingeridas — permitiendo puntuar el retrieval por id, sin juez.

Medido en BEAM-100K (3 conversaciones, 42 preguntas): nDCG@10 0,283 / 0,444 / 0,250
y recall@10 0,322 / 0,566 / 0,267, bastante por debajo de los números LOCOMO de este
repo (0,484 / 0,610), tal como predicen las propias líneas base del paper.

### Correcciones

- **`calibrate --apply --json` descartaba `--apply` en silencio**: la rama JSON
  retornaba antes de persistir. Toda calibración automatizada imprimía un umbral y
  no guardaba nada.
- **Con abstención activada y sin umbral calibrado, el gate OOD rechazaba el 100% de
  las consultas respondibles** (recall 0,0000, tasa de falsa abstención 1,0). La
  causa medida: los embeddings del corpus están a distancia Mahalanobis p50 18,7 de
  su propio centro mientras las consultas están a p50 51,7 —los pasajes se embeben
  con prefijo contextual y las consultas no—, así que el corte teórico χ² (~21) cae
  por debajo de toda consulta real. Tras `calibrate --apply` (umbral conformal
  58,35): recall 0,5660, falsa abstención **0,0**.
- **Dos tests comparaban contra la constante `EMBEDDING_DIM`** en vez de
  `embedding_dim()`, así que pasaban en CI (variable sin definir) y fallaban en
  cualquier máquina configurada con otro modelo.

## [0.15.0] — 2026-07-17 (Cargo `0.15.0` · npm `0.15.0` · PyPI `1.17.0`)

### Nuevo

- **El servidor arranca aunque PostgreSQL no responda.** Antes moría con
  `exit(1)` antes de hablar el protocolo — el motivo real por el que el
  quality-check de Glama nunca pasaba: levanta el contenedor sin base de datos
  real. Ahora arranca en modo degradado (`tools/list` es estático, cada tool
  falla con su error real en vez de tirar todo el proceso).
- **`cuba-memorys hook install`** conecta git a la memoria: `post-commit`
  exporta, `post-checkout` importa, y un merge driver propio fusiona
  observaciones/relaciones/entidades por id en vez de dejar conflict markers.
  `--with-codegraph` además re-indexa el grafo de código en cada commit.
  `hook uninstall` revierte exactamente lo que `install` agregó.
- **Proveniencia en las relaciones (`extracted` / `predicted` / `inferred`).**
  `cuba_puente predict` ahora puede persistir sus sugerencias Adamic-Adar como
  relaciones reales (`persist: true`, `relation_type` configurable) en vez de
  solo devolverlas; `traverse` expone la proveniencia de cada arista.
- **`cuba-memorys codegraph build`**: parsea Rust y Python con tree-sitter
  (determinístico, sin LLM) y lo integra en el MISMO grafo que ya usan
  `cuba_faro`/`cuba_puente` — funciones/clases se vuelven entidades buscables,
  las llamadas/imports resueltos se vuelven relaciones con
  `provenance='extracted'`. Una llamada solo se resuelve cuando su nombre
  coincide con exactamente un símbolo del lote parseado; ambiguas se
  descartan en vez de adivinar.

### Correcciones

- **El binario Linux x64 se compila contra musl, no contra glibc.** Exigía
  glibc ≥ 2.39 (compilado en el runner de GitHub), inexistente en Debian 12,
  Ubuntu 22.04 o RHEL 9 — así rompió el primer intento de build de Glama.
  Ahora es estático (`static-pie`), sin dependencia de la glibc del sistema.
- **22 bugs encontrados en una auto-auditoría** de todo lo de arriba (cada
  hallazgo verificado adversarialmente antes de arreglarse), entre ellos: una
  fuga de `app.current_project` entre conexiones recicladas del pool (podía
  filtrar el scope de un proyecto a una request de otro), `--conflict
  overwrite` de `cuba_sync` completamente no-funcional (se comportaba como
  `skip`), `export()` nunca borraba archivos de filas eliminadas de la DB
  (podían resucitar en el próximo `import`), y funciones anidadas mal
  atribuidas en el parser de código (arreglado en Rust y Python).

## [0.14.1] — 2026-07-15 (Cargo `0.14.1` · npm `0.14.1` · PyPI `1.16.1`)

### Correcciones

- **`models runtime --gpu` instala de verdad la GPU.** Antes solo extraía la
  librería principal del runtime, nunca los execution providers de CUDA
  (`libonnxruntime_providers_cuda.so` + `_providers_shared.so`), así que aun con
  una instalación GPU limpia el proveedor CUDA no se registraba y todo caía a CPU
  en silencio. Además el comando saltaba la descarga si ya existía cualquier
  runtime, con lo que pasar de CPU a GPU era un no-op. Ahora `--gpu` siempre
  re-descarga y extrae la principal más los providers (TensorRT excluido: necesita
  librerías extra que no distribuimos), y lista cada archivo extraído.

- **`doctor` reporta el estado real de la GPU.** `gpu::active_provider()` devolvía
  un `"cuda"` fijo sin comprobar nada, así que `doctor` mostraba `[ok] gpu` incluso
  corriendo en CPU. Ahora `gpu::status()` verifica que el provider CUDA esté junto
  al runtime y que exista una GPU NVIDIA, y `doctor` avisa con `warn` accionable
  cuando una build con GPU degrada a CPU.

- **Las migraciones vuelven a ser inmutables.** El barrido de comentarios de
  `e96df5d` había tocado 33 archivos de migración ya publicados. sqlx valida el
  checksum SHA-384 de cada migración aplicada en cada arranque, así que cambiar su
  contenido rompía el arranque contra cualquier base creada antes de 0.14 (toda
  instalación real) con `migration N was previously applied but has been modified`.
  Restaurados los 33 archivos a su contenido original; una migración publicada es
  inmutable y el estándar de "código sin comentarios" no le aplica.

## [0.14.0] — 2026-07-15 (Cargo `0.14.0` · npm `0.14.0` · PyPI `1.16.0`)

### Modos de funcionamiento: `CUBA_MODE=local | red | completo`

Un preset que configura la base de datos, los modelos y la red saliente a la vez, en
vez de alinear a mano una docena de variables:

- **local** (default) — Postgres en Docker local, modelos locales, sin red saliente.
- **red** — Postgres gestionado compartido (TLS): dos máquinas con una sola memoria,
  procedencia por nodo (`origin_node` / `CUBA_NODE_NAME`), sincronización en tiempo real.
- **completo** — todo: reranker (GPU si hay) + `cuba_docs`. Máxima capacidad.

`doctor` reporta el modo activo como primer check. Las env vars individuales siguen
ganando sobre el preset.

### El reranker se enciende, y es la mayor mejora del proyecto

Medido: base RRF nDCG **0.2758 → con reranker 0.5300**, **+92%** (bootstrap pareado
[+0.207, +0.302], mejora 86 de 191 queries). Estaba inerte: nada apuntaba el binario al
modelo (ahora cae al caché y `doctor` lo reporta) y puntuaba los 50 candidatos de uno en
uno (ahora una pasada batcheada). En CPU sigue siendo pesado, así que `faro` lo acota con
un timeout (`CUBA_RERANK_TIMEOUT_SECS`, 20 s) y cae a RRF si se pasa. En GPU es instantáneo.

### GPU: auto-detección CUDA → DirectML → CPU

Las tres sesiones ONNX (embeddings, NLI, reranker) registran los execution providers
compilados y caen a CPU si el runtime no los soporta. Los binarios de release traen CUDA
(NVIDIA) y, en Windows, DirectML (cualquier GPU). `cuba-memorys models runtime --gpu`.

### Onboarding cross-platform

- `cuba-memorys models <embed|nli|reranker|runtime|all>` — descarga modelos y runtime en
  cualquier OS, reemplaza los tres `.sh` (borrados) que no corrían en Windows.
- Procedencia: columna `origin_node`, se rellena sola desde `CUBA_NODE_NAME` o el hostname.

### Código sin comentarios

Nuevo estándar: código limpio y auto-explicativo, el porqué en los mensajes de commit.
Se quitaron los comentarios inline de todo el proyecto (~4600 líneas); build, clippy y
tests en verde.

## [0.13.1] — 2026-07-14 (Cargo `0.13.1` · npm `0.13.1` · PyPI `1.15.1`)

Two things v0.13.0 shipped that nobody could use, found by installing it instead of
trusting the green checkmark.

### `cuba_docs` shipped to nobody

It was gated behind a Cargo feature that is **off by default** — and the published
binaries *are* the default build. So the tool existed for one entire release and could
not be invoked by a single person who installed from npm or PyPI. CI never compiled the
feature either, so nothing anywhere was checking it.

The feature now ships **compiled in** and switched off at runtime. With `CUBA_DOCS`
unset the tool is not advertised, the dispatcher refuses it, and the server makes no
outbound request of any kind — the guarantee is unchanged, but it is now a guarantee
an agent can see, rather than a comment promising one. Set `CUBA_DOCS=1` to enable it.

- Release binaries and the Python wheel build with `--features docs`.
- CI compiles, clippies and tests **both** configurations. A feature CI never builds is
  a feature nobody is checking.
- A test asserts the tool is absent from the catalogue when the switch is off.

### The npm recovery instruction sent you in a circle

With `npm config set ignore-scripts true` — a reasonable hardening, and common — the
postinstall that downloads the binary never runs. `bin.js` correctly refused to run a
stale binary off the PATH and told you to fix it with:

    npm rebuild cuba-memorys --foreground-scripts

**which cannot work, because `npm rebuild` obeys `ignore-scripts` too.** The command it
prints now is the one that was verified on a machine with the setting on:

    npm rebuild cuba-memorys --ignore-scripts=false --foreground-scripts

A recovery instruction that fails is worse than none: it costs the reader the time to
try it before they start doubting the message instead of the setting.

## [0.13.0] — 2026-07-14 (Cargo `0.13.0` · npm `0.13.0` · PyPI `1.15.0`)

### `verify` decides for itself now — locally, in Spanish, in 50 ms

v0.11.2 fixed `cuba_faro mode=verify` by handing its evidence to an LLM judge. That was
correct and it was slow: ~20 s per claim through the `claude` CLI, and with no CLI and
no MCP client that supports sampling, verification degraded to `unknown` — which is to
say, to nothing.

Entailment is a classification problem, so it is now classified: **mDeBERTa-v3-base-xnli**
(100 languages, 87.1% on XNLI) runs on the ONNX runtime this project already links.

```
claim: "cuba-memorys está escrito en Java"   (FALSE)
  v0.11.0 (cosine) ....... 0.61  ← scored HIGHER than the true claim
  v0.11.2 (LLM judge) .... contradicted, ~20 s
  now (local NLI) ........ contradicted, ~50 ms, offline, free

claim: "cuba-memorys está escrito en Rust"   (TRUE)
  now .................... verified, 0.995
```

A full 10-evidence `verify` went from blowing the 30 s handler timeout to **5 s**.

- New `NliJudge`, preferred in `CUBA_JUDGE=auto` when a model is installed. It takes
  `judge_claim` (entailment) and leaves `judge` (contradiction between two memories) to
  the LLM — that taxonomy includes `supersedes`, which means *the same fact, updated*,
  and needs a sense of time a 3-way classifier does not have. Reporting a port migration
  as a contradiction would be a regression, so the NLI does not answer what it cannot.
- `./rust/scripts/download_nli.sh` fetches the model. `cuba-memorys doctor` reports it.
- Undecided claims do **not** escalate to an LLM by default. An undecided NLI already
  returns `unknown`, which counts for neither side — abstaining is *already* safe.
  Escalation buys recall, not safety, and it costs 12 s per evidence. Opt in with
  `CUBA_NLI_ESCALATE=1`.

### Three things that looked obviously right and were measurably wrong

Recorded because the next person to improve this will reach for the same three.

1. **The quantized model (323 MB) confirms false claims.** It read evidence saying the
   reranker "is disabled by default" as SUPPORTING the claim that it is *enabled*, at
   0.62 confidence. The fp32 export says `contradicts` at 0.995. DeBERTa-v3's
   disentangled attention does not survive int8 — and it bought nothing: **48 ms per
   verdict quantized, 53 ms at full precision.** It was paying in accuracy for a speedup
   that does not exist on CPU. Ships fp32 (1.1 GB).

2. **Decomposing evidence into sentences makes it worse.** XNLI premises are single
   sentences and stored memories are paragraphs, so cutting them up looked necessary. It
   got **three of five real cases wrong, and it CONTRADICTED true claims** — because NLI
   is trained on *scenes*, where two predicates about one subject are alternatives ("a
   man is playing guitar" genuinely contradicts "…playing piano"). A knowledge base is
   not a scene: "cuba-memorys uses PostgreSQL" and "cuba-memorys is written in Rust" are
   both true, and shown in isolation the model rates them a **contradiction at 0.993**.
   Every clause about a *different attribute of the same entity* became a vote against
   the claim. The premise is now scored whole.

3. **An argmax over the 3-way head is not a verdict.** The model scores `entailment`
   at 0.693 for "the reranker is a bi-encoder" against evidence saying it is a
   cross-encoder — jargon whose mutual exclusivity it cannot know. An argmax publishes
   that as `supports`: a confirmed false claim, the exact bug this subsystem exists to
   kill. Each class must now clear its own floor, and the floors are **asymmetric on
   purpose** — entailment needs 0.80, contradiction 0.60. Confirming a false memory and
   doubting a true one are not errors of equal cost. Every genuine entailment measured
   scored ≥0.95; spurious ones live in the 0.6s.

### Fixed

- **Every CLI invocation was silently using hash embeddings.** `setup` downloads
  onnxruntime to `~/.cache/cuba-memorys/onnxruntime/` and writes `ORT_DYLIB_PATH` into
  the MCP client's config — so the *server* found the library and nothing else did.
  `search`, `dedupe` and `reembed` run from a plain shell saw no `ORT_DYLIB_PATH`, found
  no system library, and fell back to hash vectors: **the same query answered from the
  CLI and from the MCP was hitting two different vector spaces.** `locate_onnxruntime()`
  now looks in its own cache directory, where `dlopen` cannot guess to look.

## [0.12.0] — 2026-07-13 (Cargo `0.12.0` · npm `0.12.0` · PyPI `1.14.0`)

The benchmark could not measure what this project claimed to have measured, and the
graph was quietly broken.

### ⚠ Two published findings are withdrawn

The evaluation had **ten queries**. At n=10 the 95% interval on nDCG is roughly
±0.12, and the smallest detectable effect is ~0.25. Two conclusions this project
published rested inside that noise:

- ~~**"The cross-encoder reranker earns nothing"**~~ — **it had never run.** Three
  bugs in series, each hiding the next:

  1. `faro` wrapped the call in `if let Ok(..)`, so the error was **dropped** and the
     RRF ranking returned untouched. The same silent-degradation pattern that let the
     vector branch die unnoticed. *This is why the output was "bit for bit identical"
     to not reranking:* not because reranking changed nothing, but because it never
     happened.
  2. It fed the model **`token_type_ids`**. bge-reranker-v2-m3 is XLM-RoBERTa, which
     has no segment embeddings — that is a BERT input. Every inference threw
     `Invalid input name: token_type_ids`.
  3. It read the logits as **`f32`**. The checkpoint emits `f16` (needs ort's `half`
     feature). This one only surfaced once (2) was fixed.

  Two of these were architectural mismatches, obvious from a single error message.
  **Nobody saw the message, because bug 1 ate it.** A feature cannot earn anything
  when its results are thrown away. Fixed and being measured properly.

- **"Associative retrieval degrades all four metrics"** — the conclusion holds, the
  evidence did not. −0.03 at n=10 is a quarter of the error bar. But the correct test
  for two configurations over the *same* queries is a **paired** one, and under a
  paired bootstrap on the new dataset the interval is **[−0.051, −0.018]** and never
  touches zero: it improves **0** queries and hurts **23**. The decision to disable it
  was right; the reasoning was not. *The power was never in more data — it was in
  using the right test.*

### The real numbers

The system's nDCG is not 0.894. On 221 id-scored queries it is **0.50** [95% CI
0.44–0.56]. It did not get worse; it was never 0.894.

`compact` saves **28% of tokens** (not 40%) at **exactly identical nDCG** — identical
to four decimal places, because a response format cannot change *which* documents rank,
only how they are printed. That the old benchmark measured a quality cost for
truncation was itself an artefact: truncating the text removed the marker substrings it
was grading on.

The **+21.2 nDCG for bge-m3 is withdrawn.** The direction is almost certainly right;
the magnitude came from the broken benchmark and re-establishing it would mean
re-embedding the corpus twice.

### Fixed — the benchmark itself

- **Relevance was judged by substring match.** A result counted as correct if its
  text merely *contained* a marker word, so every observation mentioning "postgres"
  scored as a right answer to any question about postgres — whether it answered
  anything or not. That measures keyword presence, not retrieval, and it biases the
  whole benchmark toward the lexical branch and against the vector one. Ground truth
  is now a set of observation **ids** per query (TREC-style qrels).
- **nDCG normalized against what was RETRIEVED, not what EXISTS.** With 5 relevant
  documents in the corpus and 2 found, the "ideal" ranking was taken to be those 2 —
  so a system that missed 60% of the answer scored a **perfect 1.0**. The ideal is
  now built from `min(total_relevant, k)`, so documents you failed to retrieve count
  against you. This makes the numbers go **down**, which is the expected direction
  when you stop grading on a curve you drew yourself.
- **R@10 = 3.125 shipped in the README.** Recall is a proportion. The denominator
  was the count of *marker strings*, not of relevant *documents*.
- **Every metric now carries a bootstrap 95% interval** (Efron 1979, deterministic
  resampling) and the run reports its **minimum detectable effect**. A benchmark that
  cannot see a 5-point change should not be used to claim a 3-point regression.

### Added

- **`cuba-memorys dedupe`** — entities that are the same thing under different names.
  `cuba_alma create` inserts with `ON CONFLICT (name)`: a different string is a
  different entity, so one project fragments into `Mapupita-Web`, `Mapupitta-Web`
  (typo), `Mapupita Web`, `mapupita`… On the live brain: **266 entities, 158 (59%)
  with not a single relation** — for PageRank and multi-hop retrieval they do not
  exist.

  The infrastructure to fix this was already present and dead: `brain_entity_aliases`
  has a schema, indexes, and a `resolve_entity()` that matches exactly and fuzzily.
  Zero rows; nothing called the function. Merging now writes the old name there, so
  nothing is lost.

  **What decides a merge is not the embedding centroid.** That was the obvious idea
  and it is wrong: `M-Codes Reference Guide` and `G-Codes Reference Guide` sit at
  **0.811 cosine** between centroids. On a corpus about one domain, centroid
  similarity measures the domain, not the entity — trusting a 0.80 threshold would
  have merged two different CNC guides irreversibly. So `--apply` merges only what is
  *provable* (identical after normalizing case and separators), and everything else
  is shown, or judged one by one with `--judge`.

  (The LLM judge, asked whether `Mapupitta-Web` and `Mapupita-Web` were the same
  entity, first answered *"different — there are separate memory records for each"*.
  That is the bug offered as proof there is no bug. The prompt now disarms that
  argument explicitly, and a test pins it.)

- **`reranker_degraded` in the search response.** You asked for reranking, the
  cross-encoder threw on every pair, and you got the RRF order back looking exactly
  like a reranked one. Same reason `degraded` exists for the vector branch: an agent
  handed a silently un-reranked top-10 will simply trust it.

### Fixed — the CLI was eating your flags

- **`search "x" --format verbose` searched, literally, for «x --format verbose».**
  Unknown flags fell into the catch-all and were **concatenated onto the query**. It
  returned nothing, with no hint why.
- **`save "x" --importancia 0.9` stored «x --importancia 0.9» AS THE MEMORY CONTENT.**
  Same catch-all. This one corrupts data.
- Both, plus `delete`, now reject unknown `--flags` with a usage error. Same family as
  the `--batch 64` that `reembed` silently ignored: an argument a tool pretends not to
  see is an argument that lies about what it did.

### Added — build limits

`.cargo/config.toml` (3 jobs) and a `quick` profile (`lto = "thin"`, 16 codegen units).
The release profile's fat LTO with `codegen-units = 1` peaked past 8 GB in a single
unit and froze a 14.9 GB laptop running zram. Use `--profile quick` to iterate;
`--release` only to measure and ship.

## [0.11.2] — 2026-07-13 (Cargo `0.11.2` · npm `0.11.2` · PyPI `1.13.2`)

The anti-hallucination feature was hallucinating. Found by pointing the demo at it.

### ⚠ Breaking — `cuba_faro mode=verify` now calls an LLM judge

Verification escalates its evidence to a judge and derives confidence from the
verdicts. It costs a model call (free via MCP sampling — your client's model — or a
local `claude` CLI) and takes a few seconds. With no judge available it answers
`unknown` instead of inventing a verdict. Response gains `interpretation`,
`judged_by`, and a per-evidence `verdict`/`reason`.

### Fixed

- **`verify` scored false claims HIGHER than true ones.** Confidence came from
  cosine similarity to the retrieved evidence — and similarity measures what a text
  is *about*, not what it *asserts*. "cuba-memorys is written in Rust" and "…in
  Java" are nearly the same vector: same subject, same shape, one word apart.
  Measured on the live 1,461-observation corpus:

  | claim | before | after |
  |---|---|---|
  | "usa RRF con k=60" (true) | 0.59 | **0.83 · verified** |
  | "está escrito en Java" (false) | **0.61** | **0.00 · contradicted** |
  | "la mejor paella lleva azafrán" (unrelated) | 0.45, 10 "evidence" items | **0.00 · unknown**, none |

  No threshold could have fixed it — true claims landed at 0.43–0.57 similarity and
  false ones at 0.55–0.59, completely overlapping. Entailment is a different
  question from similarity and needs something that reads. Evidence below a
  similarity floor is now discarded (retrieval always returns its top-K; that is
  right for search and wrong for verification), and what survives goes to a judge.
  Verdicts are weighted by similarity, so similarity decides how much a verdict
  counts — never what the verdict is. "Unrelated" contributes to neither side: being
  on-topic is not support.

- **`cuba_juez` with the `claude_cli` backend never worked.** `claude --print
  --output-format json` returns a report *about* the call, with the model's answer
  as a string field inside it. The parser took the first `{` and last `}` — that
  envelope — found no `verdict`, and fell back to "unknown". Since v0.8. The
  heuristic quietly did all the work while the logs showed a model being called.

- **Setting `ONNX_MODEL_PATH` without `ORT_DYLIB_PATH` hung the server.** `ort` loads
  the runtime dynamically; when it cannot find the library it does not error, the
  process just stops answering — after starting, connecting, migrating and
  announcing itself ready. It logs an ERROR and degrades to lexical search now.

- **`compact` reported `"i": null`** on most results. Only the vector branch failed
  to select `importance`, and a semantic hit usually wins the fusion — so the field
  looked broken exactly where it mattered.

- Judge verdicts are fetched **concurrently**. Serially, a three-evidence verify cost
  over a minute of wall clock and would have been unusable however correct it was.

### Changed

- **README rewritten** for someone arriving new, not for someone who followed the
  version history. Every number in it is checked against the code by a test or was
  measured — the old one claimed 25 tools (there are 28), pinned installs to
  versions two releases stale, and documented none of the 13 CLI commands.
- **The demo no longer writes to your database.** It defaulted `DATABASE_URL` to the
  real brain on `:5488`, so recording the README GIF created entities in a live
  memory store and ran PageRank over it. It now starts a throwaway Postgres and
  destroys it on exit, and ignores your embedding config rather than inheriting it.

## [0.11.1] — 2026-07-13 (Cargo `0.11.1` · npm `0.11.1` · PyPI `1.13.1`)

Two bugs found by *using* v0.11.0 rather than testing it — both in the same family
as the ones v0.11.0 set out to kill.

### Fixed

- **Every new memory was stamped with the wrong model name.** `embeddings::onnx`
  exposed a `pub const CURRENT_MODEL = "multilingual-e5-small"` beside a
  `current_model()` that reads `CUBA_EMBED_MODEL`. The split was perverse: every
  site that **wrote** an embedding used the constant, every site that **compared**
  one used the function. So on a bge-m3 corpus, each new observation got a correct
  1024-d bge-m3 vector labelled with a 384-d model that had not run in months —
  permanently stale to `doctor`, whose warning count could only grow, and to
  `zafra reembed`, which could never converge: it re-encoded the row, and the next
  write re-mislabelled it.

  The vectors were always fine (measured, not assumed: cross-label cosine on
  same-entity pairs sits inside the range of within-label cosine — one vector
  space, not two). Only the name lied. But that name is what tells you, after the
  next model change, which rows still need re-encoding. `CURRENT_MODEL` is private
  now, so the compiler forbids the mistake — and it immediately found a fifth site:
  a smoke test asserting the constant's value, which had pinned the bug in place.

  Only affects setups that override `CUBA_EMBED_MODEL`; on the default model the
  label was accidentally correct.

- **`reembed`'s smallest unit of work was "everything".** One observation missing a
  vector, and the only cure on offer was to recompute all 1,461 — overwriting 1,460
  good vectors to fill one empty. It now re-encodes the stale set by default (no
  vector, or tagged with another model), which is right in both real cases without
  a flag: changing models makes every row qualify; a single failed embedding makes
  exactly one. `--all` still forces the full pass.

- **`reembed --batch 64` was silently ignored** — only `--batch=64` parsed, and the
  space-separated form fell into a catch-all that dropped it. Both forms work now,
  and an unrecognised argument is an error instead of a shrug.

## [0.11.0] — 2026-07-13 (Cargo `0.11.0` · npm `0.11.0` · PyPI `1.13.0`)

The fourth memory, and every optimization measured on a real corpus instead of
assumed. Several long-standing features turned out not to work at all; they are
fixed or cut, and the negative results are recorded rather than buried.

### ⚠ Breaking — `cuba_faro` now answers in `compact` by default

The default response shape changed from `verbose` to `compact`: abbreviated keys
(`e` entity, `c` content, `t` type, `i` importance, `s` score) and no per-branch
score breakdown. It costs **40% fewer tokens at identical nDCG** — the truncation
point was swept and set at its measured knee — and an agent reasoning over
memories does not need `bm25_score` to do it.

**If you parse the response**, this breaks you. Pass `"format": "verbose"` to get
the old shape back, unchanged:

```json
{ "query": "...", "format": "verbose" }
```

Agents reading the JSON are fine — the tool description documents the short keys.
Scripts and tests that index `entity_name` / `content` / `*_score` are not, and
must ask for `verbose`. Both shapes are now pinned by an integration test, so
neither can drift again.

### Added

- **Procedural memory** — `cuba_receta` (migration `0033`): how things are *done*
  here, not just what is true. Ranked by the **Wilson lower bound** of the success
  rate, so a recipe with a track record beats a lucky first try (1-of-1 scores
  0.21; 47-of-50 scores 0.84). Reinforced by outcome, not by access — the
  ACT-R distinction between declarative and procedural memory. `cuba-memorys
  skills <dir>` exports them as Claude Code Skills, which load lazily.
- **Progressive tool loading** — `cuba_tools` + `cuba_call`, and
  `CUBA_TOOL_PROFILE=lean`. The catalogue shrinks 67% (25,060 → 8,413 chars)
  while **every tool stays callable**: schemas are deferred, not deleted.
- **Calibrated abstention** — `cuba-memorys calibrate`. The OOD gate now detects
  out-of-distribution queries (100%) without rejecting answerable ones (0% false
  abstentions). Persisted in `brain_calibration` (`0032`).
- **RBAC** — `brain_principals` × `brain_grants` (`0031`), enforced by a
  RESTRICTIVE RLS policy. Zero regression: with no principals defined, nothing
  is denied.
- **New subcommands** — `doctor` (health check), `calibrate`, `recall` (session
  context for a `SessionStart` hook), `skills`, `reembed`, `link`, `setup`,
  `search` / `save` / `delete` / `export` / `dashboard`.
- **Graph auto-linking** — `cuba-memorys link`, scored by normalized pointwise
  mutual information so a ubiquitous entity earns no edges from being ubiquitous.
- **Model-agnostic embeddings** — e5-small (384-d) or bge-m3 (1024-d) by config.
  Measured on a real 1,443-observation corpus: **nDCG@10 0.682 → 0.894**.

### Fixed

- **Hybrid search could silently become lexical search.** A failing vector branch
  was discarded by an `if let Ok(..)` — no log, no flag, no symptom. Now it logs
  at ERROR, sets `degraded: true` in the response, and the server **refuses to
  start** when the model's dimension disagrees with the column.
- **`setup check` reported "all consistent" while a stale project-level
  `.mcp.json` spawned 384-d servers against a 1024-d column.** It now audits
  project configs too, and treats an absent `CUBA_EMBEDDING_DIM` as the 384-d
  value it actually is — so it can disagree with one that sets 1024.
- **Retrieval was not deterministic.** Fusion happened in a `HashMap` and sorted
  by score with no tie-break; Rust randomizes iteration order per process, so
  three identical eval runs scored 0.7389 / 0.7344 / 0.7389. Every optimization
  number previously recorded rested on that. Now tie-broken by id: 5/5 identical.
- **The token budget counted text it then threw away**, spending a 5,000-token
  budget to return 798. Shape first, then budget. Compact truncation swept and
  set at its measured knee (1200 chars): **40% fewer tokens, identical nDCG**.
- **The OOD threshold rejected 100% of answerable queries.** The covariance was
  fitted from 500 samples in 384 dimensions with a fixed ridge mislabelled
  "Ledoit-Wolf". Now real Ledoit-Wolf shrinkage plus a conformal threshold.
- **The eval panicked on an empty result list** (`relevances[..1]` on a
  zero-length slice) — it never fired only because nothing ever abstained.
- **The LLM judge shipped credentials to a third party.** Observation text is now
  redacted (Postgres URLs, provider tokens, JWTs) and length-capped.
- **`doctor` could not see a stale process** — Linux appends `" (deleted)"` to
  the exe name, and the filter dropped exactly the processes it existed to find.
- **`cuba-memorys --version` connected to your database and ran migrations.**
  Argument parsing had a catch-all that fell through to the MCP server, so the one
  command a person runs *because they do not yet trust what they installed* was the
  one that quietly reshaped their schema. `--version` is now inert — it prints and
  exits, with a test that pins it by pointing `DATABASE_URL` at a closed port.
- **`--help` did not exist**, for the same reason, which is why nothing ever
  documented the 13 subcommands. And a typo (`doctro`) launched the server on a
  stdio socket nobody was speaking to — indistinguishable from a hang. An
  unrecognised argument is now a usage error (exit 2). The server is what you get
  with *no* arguments, which is how MCP clients launch it.
- **npm could silently run a different version than the one you installed.**
  `bin.js` fell back to any `cuba-memorys` on the `PATH` when the postinstall
  binary was missing — and postinstall does not run under `--ignore-scripts`,
  standard practice in hardened CI. Installing 0.11.0 and getting an 0.6.0 left
  over from an old pip install is not a fallback; here it is a *migration* run by
  the wrong binary. The `PATH` binary must now prove its version matches, or the
  launcher refuses and says why.
- A test now pins `Cargo.toml` and `package.json` to the same version. npm's
  postinstall downloads from `releases/download/v{package.json.version}/`, an asset
  the release workflow only builds for the *Cargo* version — nothing connected
  those two numbers, and a drift would have 404'd every install.
- Zero `unwrap()` in production code; zero clippy warnings.

### Changed

- `cuba_faro` defaults to `compact` (40% cheaper, same quality).
- The eval reports **token cost beside every quality metric**, and tracks false
  abstentions — abstention accuracy alone is trivially maximized by answering
  nothing.

### Removed / not adopted

- **The cross-encoder reranker does not earn its place.** Its integration added
  `score × 0.0001` to fusion scores separated by 0.00016 — arithmetically
  incapable of reordering anything. Fixed the wiring, measured it properly, and
  it still bought nothing for 0.33 s/query and 1.1 GB. Off by default, with the
  negative result documented in the module.
- **Associative multi-hop retrieval degrades every metric** (nDCG 0.734 → 0.705,
  MRR 0.833 → 0.660, recall 2.31 → 1.88). The previous "+10 points recall" claim
  predates the determinism fix. Stays opt-in and off.

## [0.10.0] — 2026-06-04 (Cargo `0.10.0` · npm `0.10.0` · PyPI `1.12.0`)

Knowledge-graph memory plane: bitemporal facts, graph metrics, retrieval benchmarks,
and MCP unified search view — built on the v0.9 hybrid `cuba_faro` stack (not replaced).

### Added
- **Bitemporal core** (`core::bitemporal`, migration `0018`): `brain_facts` +
  `brain_fact_supersedes`; writes mirror observations on `cuba_cronica` add/batch_add
  and `cuba_ingesta` (via batch). **Default on**; disable with `CUBA_BITEMPORAL=0`.
- **Entity linking & temporal query** (`core::entity_linking`, `core::temporal_query`,
  migrations `0019`–`0020`).
- **Graph metrics** (migration `0022`): `brain_node_metrics` with PageRank, energy,
  betweenness; `cuba_zafra` `pagerank` persists ranks then refreshes energy scores.
- **Communities** (migration `0023`): Leiden detection + `detect_and_persist`;
  `cuba_zafra` action `communities`; `cuba_vigia` health metric persists tags.
- **Spreading activation** (`graph::activation`): multi-hop propagation; enriches
  `cuba_puente` `predict` alongside Adamic-Adar.
- **Eval harness** (`eval/`): nDCG@k, MRR, P@k, R@k over live `cuba_faro` hybrid;
  JSONL dataset loader + builtin smoke set; JSON reporters.
- **MCP memory v2 view** (migration `0024`): `v_unified_memory_search` joins facts via
  `brain_entities` (never `fact_id = node_id`).
- **Compatibility views** (migration `0025`): `v_observations_compat`.
- **Calibration alignment** (migration `0021`), scripts: `backup-db.sh`, `restore-db.sh`,
  `merge-gate.sh`, `mcp_live_session_test.py`.

### Changed
- `cuba_faro` remains production hybrid search (RRF + BM25 + vector + optional rerank).
- PageRank REM cycle also upserts `brain_node_metrics.pagerank_score`.

### Notes
- Optional Cargo features `bitemporal`, `graph-energy`, `eval-benchmarks` are markers;
  modules ship in the default library build.
- Run `./scripts/merge-gate.sh` before merging to `main`.

---

## [0.9.3] — 2026-05-04 (Cargo `0.9.3` · npm `0.9.3` · PyPI `1.11.3`)

Final piece of the v0.9.x roadmap. The cross-encoder reranker is now a
real bge-reranker-v2-m3 ONNX forward pass, not the heuristic baseline
that v0.9.2 shipped as scaffolding.

### Added
- **Real bge-reranker-v2-m3 ONNX forward pass** (`search::rerank`).
  Mirrors the `embeddings::onnx` loader pattern: lazy-init `Session`
  behind a `Mutex`, `tokio::task::spawn_blocking` for inference, and a
  semaphore capping concurrent calls at 2 to match
  `with_intra_threads(2)` (Little's Law — prevents threadpool
  starvation under load). Tokenizer encodes the (query, candidate)
  sentence pair with `[CLS]/[SEP]` segments and 512-token truncation.
  Output handled for both `[batch, 1]` regression heads and
  `[batch, 2]` binary classification heads (logit difference). Sigmoid
  to [0, 1] before sorting.
- Activation: drop a directory containing `model.onnx` (or
  `model_quantized.onnx`) plus `tokenizer.json` and point
  `CUBA_RERANKER_PATH` at it. Identity fallback otherwise — production
  behavior unchanged when the asset is absent.
- `cuba_faro` keeps the same `rerank: bool` arg surface from v0.9.2;
  no client-side change required.
- Expected gain: +12-25% nDCG@10 (Xiao 2023, BGE-Reranker paper).

### Changed
- Replaced the v0.9.2 heuristic body (token overlap + length penalty)
  with the real cross-encoder forward pass. Heuristic-only callers
  see unchanged behavior because the env var gates activation.

### Notes
- Adds zero new Rust deps — `ort 2.0.0-rc.12` and `tokenizers 0.21`
  were already present for `embeddings::onnx`.
- bge-reranker-v2-m3 quantized ONNX is ~280 MB; download from
  https://huggingface.co/BAAI/bge-reranker-v2-m3 (or use
  `huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir
  models/bge-reranker-v2-m3`). The asset is NOT bundled in the
  release artifact — operators provide it explicitly.

---

## [0.9.2] — 2026-05-04 (Cargo `0.9.2` · npm `0.9.2` · PyPI `1.11.2`)

Closes the v0.9.x roadmap with the deferred MCP correlator + reranker
scaffolding. No breaking changes.

### Added
- **MCP request/response correlator** (`protocol.rs` major refactor).
  The reader/writer split into three concurrent tasks: a single-owner
  stdout writer task draining an `mpsc::UnboundedSender<Value>`, a
  `PENDING` map of `oneshot::Sender<Value>` keyed by server-initiated
  request id, and per-request handler tasks. This enables:
  - **Real `MCPSamplingJudge`** — `protocol::request_sampling()` issues
    a `sampling/createMessage` to the connected client and awaits the
    reply on a oneshot. 30s timeout matches `HANDLER_TIMEOUT`. When the
    client did not advertise `sampling`, fails fast with an actionable
    message and the resolver auto-falls back to CLI / API / heuristic.
  - **`notifications/progress`** — `protocol::notify_progress()` emits
    standard MCP progress events. Wired into `cuba_zafra reembed`
    (~5% increments) so re-embedding 500+ observations is no longer
    silent.
  - **`notifications/cancelled`** — per-request `CancelToken`
    registered by `tools/call`, signaled by the cancellation
    notification. Handlers race against the token via `tokio::select!`
    and return a clean error instead of running to completion.
- **Cross-encoder reranker scaffold** (`search::rerank`). Activated by
  `CUBA_RERANKER_PATH` env var pointing to a bge-reranker-v2-m3 ONNX.
  When unset, identity fallback preserves upstream RRF order. New
  `cuba_faro` argument `rerank` for explicit override. Pipeline:
  top-50 RRF → rerank → MMR → top-K. Heuristic body included as a
  baseline that exercises the integration path; full model forward
  documented as a one-file follow-up to drop in.
- New `cuba_faro` arg surface: `rerank` (boolean).
- New helper `protocol::register_cancel_token` /
  `protocol::unregister_cancel_token` exported for any future handler
  that wants explicit cancellation hooks.

### Changed
- `JsonRpcResponse` / `JsonRpcError` structs removed — every outbound
  envelope is built ad-hoc with `serde_json::json!()` and pushed to the
  `OUTBOUND` channel. Narrower surface, no temptation to construct
  envelopes from places that should not.
- `cuba_zafra reembed` accepts `_meta.progressToken` to correlate
  progress with the MCP `tools/call` request id.

### Notes
- Real bge-reranker forward pass (ONNX session + tokenizer) is the
  only piece marked as TODO in the rerank module. The integration
  point, env var, schema arg, and identity fallback are all live.
  When the asset is bundled, swap the ~30-line heuristic body for the
  real inference and `enabled()`-true path becomes production.

---

## [0.9.1] — 2026-05-04 (Cargo `0.9.1` · npm `0.9.1` · PyPI `1.11.1`)

Production hardening + MCP spec usage. Closes the v0.9.x roadmap with
PRs #8–#11 plus the deferred infrastructure pieces from #10/#11.

### Added
- **PR #8** — `graph::closeness` (Bavelas 1950 + Boldi-Vigna 2014 harmonic)
  and `graph::kcore` (Seidman 1983, Batagelj-Zaversnik 2003 with
  running-max for correct k-core numbers). Exposed via new
  `cuba_vigia metric=structural` action.
- **PR #9** — working memory (`cuba_pizarra` + `brain_wm` table with
  GENERATED `expires_at` from `ttl_seconds`), Allen interval algebra
  (`cognitive::allen`, all 13 relations in O(1)), ADWIN drift detector
  (`cognitive::adwin`, Bifet-Gavaldà SDM 2007 with Hoeffding bound +
  Bonferroni correction), MI tagging (`cognitive::mi_tagging`,
  Brown JMLR 2012).
- **PR #10** — Tamper-evident audit log (`cuba_archivo` + `brain_audit_log`
  with SHA-256 hash chain, append-only PostgreSQL trigger, `cuba_admin`
  bypass role). Spotlighting prompt-injection defense in
  `cognitive::judge::build_prompt` (Hines 2024 — per-call nonce markers).
  Brier score + Expected Calibration Error in `cuba_calibrar metrics`
  (Brier 1950 / Naeini AAAI 2015) with reliability diagram.
- **Optional Prometheus `/metrics` endpoint** behind feature flag
  `observability` (`metrics 0.24` + `metrics-exporter-prometheus 0.16`).
  Default bind `127.0.0.1:9090` (env `CUBA_METRICS_PORT`/`CUBA_METRICS_BIND`).
  Pre-registered metrics: `cuba_handler_duration_seconds`,
  `cuba_handler_calls_total`, `cuba_judge_calls_total`,
  `cuba_judge_timeout_total`.
- **PostgreSQL Row-Level Security** per project (migration 0017).
  `tenant_isolation` policy across the six scoped tables. Defense in
  depth — the handler-side WHERE clause stays as the primary gate, RLS
  catches direct DB connections that bypass handlers. Sentinel `*` =
  bypass, NULL = back-compat.
- **PR #11** — MCP `resources/list` + `resources/read` with the
  `cuba://` URI scheme: `cuba://entity/<name>`,
  `cuba://project/<name>`, `cuba://snapshot/<id>`. Server now advertises
  the `resources` capability during initialize. Client capability
  detection captures `capabilities.sampling` so future Sampling calls
  prefer it (today it errors out with a clear migration message — full
  loop correlator scheduled for v0.10).
- New backend `MCPSamplingJudge` in `cognitive::judge` (auto-preferred
  when client supports sampling).
- New deps: `sha2 0.10`, `hex 0.4`. Optional: `metrics 0.24`,
  `metrics-exporter-prometheus 0.16`.

### Changed
- `cuba_juez` resolver order: `mcp_sampling` → `claude_cli` → `anthropic_api` → `heuristic`.
- `cuba_calibrar` JSON Schema gains `metrics` action.
- 25 MCP tools (was 24): `cuba_archivo` joins as the audit handler.
- Smoke test count bumped to 25.
- 4 new migrations (0014 source_trust, 0015 working_memory,
  0016 audit_log, 0017 rls_policies). Total: 17.

### Notes
- 3 RUSTSEC advisories from upstream transitive deps remain open
  (rustls-webpki via sqlx, rand via tokenizers/reqwest). All upstream;
  no remediation in this scope.
- MCP Sampling backend currently fails fast — wiring requires the
  request/response correlator refactor planned for v0.10. Auto-fallback
  keeps production unaffected.

---

## [0.9.0] — 2026-05-04 (Cargo `0.9.0` · npm `0.9.0` · PyPI `1.11.0`)

Search & Retrieval upgrades + Cognitive layer refinements + sqlx-migrate
foundation. Zero breaking changes — every new feature is opt-in via
`cuba_faro` arguments or activates automatically with safe defaults.

### Added
- **PR #5 sqlx-migrate** — replaced ad-hoc `*_MIGRATION` constants with
  versioned files in `rust/migrations/` (14 migrations, 0001 → 0014). The
  bootstrap is transparent for legacy v0.7/v0.8 DBs because every
  migration is idempotent (`DO $$ ... IF NOT EXISTS ... END $$`).
- **PR #6 Phase 1** — three new search modules:
  - `search::bm25` — BM25-flavored sparse retrieval via PostgreSQL
    `ts_rank_cd` (Robertson-Walker SIGIR 1994 baseline).
  - `search::mmr` — Maximal Marginal Relevance diversification with
    Jaccard token-set similarity (Carbonell-Goldstein SIGIR 1998).
  - `search::ood` — Out-of-distribution detection via Mahalanobis
    distance with ridge-regularized Σ⁻¹ (Lee NeurIPS 2018).
  - `search::budget` — exact `tiktoken-rs` cl100k_base counting (replaces
    the "len/4 chars per token" heuristic that mis-counted Spanish 30%).
- **PR #6 Phase 2-3** — `cuba_faro` exposes 5 new arguments:
  `enable_bm25` (default `true`), `diversify`, `mmr_lambda`,
  `abstain_ood`, `ood_threshold`. Output adds `bm25_score` to the score
  breakdown alongside `text_score`/`vector_score`. `verify` mode now
  bumps `hnsw.ef_search` to 200 transactionally for recall@10≈0.99.
- **PR #7 Phase 1** — `cognitive::prediction_error::adaptive_thresholds_conformal`
  uses empirical quantiles instead of z-score — distribution-free
  (Vovk-Gammerman-Shafer 2005, Angelopoulos-Bates 2023). Cosine
  similarities are anisotropic skewed-right (Ethayarajh EMNLP 2019), so
  z-score over-fires REINFORCE; conformal does not.
- **PR #7 Phase 2** — testing effect (Karpicke-Roediger Science 2008):
  `cuba_zafra decay` now scales halflife by `(1 + ln(1 + access_count))`,
  so a memory accessed 50× decays ≈4× slower than one accessed 0×.
- **PR #7 Phase 3** — Hebbian Δt-aware burst suppression in
  `cognitive::hebbian::boost_on_access`. The boost is multiplied by
  `(1 - exp(-Δt/τ))` with τ=600s. Re-access in the same second yields
  factor 0 (anti-saturation), Δt > 1h yields ≈1 (full boost).
- **PR #7 Phase 4** — Robbins-Monro stochastic learning rate in
  `cuba_eco`'s Oja positive/negative: `η = 0.05 / sqrt(1 + access_count/100)`.
  Convergence O(1/√t) bounds importance volatility on heavily-fed
  observations.
- **PR #7 Phase 5** — source credibility tracking. Migration
  `0014_source_trust.up.sql` adds `brain_source_trust(source, alpha,
  beta, updated_at)` pre-seeded with five standard sources. Each
  `cuba_calibrar resolve` updates the Beta(α, β) posterior of every
  source supporting the verified claim (Yin-Han-Yu IEEE TKDE 2008). New
  `cuba_calibrar trust` action returns posteriors with credible-interval
  width.
- New deps (lib): `sqlx` feature `migrate`, `tiktoken-rs 0.7`,
  `nalgebra 0.33` (no LAPACK), `async-trait 0.1`.
- 22 new tests (97 total: 84 unit + 13 smoke + 2 integration ignored).

### Changed
- `cuba_faro` JSON Schema in `constants.rs::tool_definitions()` extended
  with v0.9 args. Description updated to advertise MMR / OOD / tiktoken.
- `cuba_calibrar` JSON Schema gains the `trust` action.
- `cuba_zafra` `decay` response includes `testing_effect` annotation
  with the Karpicke-Roediger citation.
- `db.rs` shrunk from 310 → 100 lines (sqlx-migrate replaces nine
  hand-rolled migration constants).
- Smoke test `test_handler_dispatch_coverage` keeps the same 23-tool
  list (no new MCP tools added in v0.9 — all upgrades are arg
  extensions or new actions on existing handlers).

### Fixed
- 6 pre-existing clippy 1.94 warnings cleaned: `vec![...]` → array
  literals in `graph/pagerank.rs::tests`; `assert!(CONST > 0)` →
  `const _: () = assert!(...)` in `tests/smoke_test.rs` and
  `pagerank.rs::tests::test_pagerank_convergence_constants`.

### Notes for upgraders
- Existing v0.7 / v0.8 DBs auto-migrate on first boot. The
  `_sqlx_migrations` table is created automatically and populated with
  the 14 historical migrations on the first run; subsequent boots
  apply only deltas.
- Legacy `embeddings/onnx.rs` heuristic `count_tokens` is now
  superseded by `search::budget::count_tokens`.

---

## [0.8.0] — 2026-05-04 (Cargo `0.8.0` · npm `0.8.0` · PyPI `1.10.0`)

Engram-Cloud-inspired additions: 4 new tools + audit of all 19 v0.7
handlers for project scoping. Zero breaking changes — every filter is
opt-in via `cuba_jornada start --project NAME`.

### Added
- **`cuba_proyecto`** (PR #1) — per-project isolation via `project_id
  UUID NULL` on six core tables, `brain_projects` registry, six actions
  (`list / current / switch / stats / rename / merge`). NULL = global =
  back-compat. Kill-switch `CUBA_PROJECT_FILTER=off`.
- **`cuba_pre_compact`** (PR #2) — survives `/compact`. `snapshot`
  persists session state (recent obs, decisions, unresolved errors,
  pending embeddings, goals) into `brain_compaction_snapshots`;
  `restore` returns the latest snapshot for the active session.
  `cuba_jornada current` now returns `compaction_hint: bool`.
- **`cuba_sync`** (PR #3) — git-friendly export/import with
  `export | import | diff | status`. Layout: `manifest.json`,
  `entities/<slug>.json` (each with embedded observations),
  `episodes/<yyyy-mm>/<id>.json`, `decisions/<id>.json`,
  `errors/<id>.json`, `relations.json`, optional `embeddings.bin.zst`.
  Idempotent via `ON CONFLICT DO NOTHING` + `brain_sync_state`. Schema
  versioning with hash-derived dedup. Path traversal protection.
- **`cuba_juez`** (PR #4) — LLM-judge for ambiguous (cosine 0.6-0.8)
  contradictions. Trait `ContradictionJudge` with three backends:
  `ClaudeCodeJudge` (subprocess to `claude` CLI, $0 with subscription),
  `AnthropicApiJudge` (feature flag `anthropic-api`), `HeuristicJudge`
  (fallback wrapping the bilingual negation marker check).
  Permanent cache via `brain_judgments(observation_a, observation_b)`
  UNIQUE index.
- WRITE audit: `cronica`, `alma`, `alarma`, `puente`, `decreto` populate
  `project_id` from current session.
- READ audit: `faro`, `vigia`, `expediente`, `contradiccion`,
  `reflexion`, `hipotesis`, `decreto`, `puente`, `alma`, `cronica` apply
  `($N::uuid IS NULL OR project_id = $N OR project_id IS NULL)` filter.

### Changed
- 23 MCP tools (was 19). 75 tests (was 68). 0 clippy warnings.

---

## [0.7.0] — Earlier 2026

10 algorithmic improvements + 19 bug fixes + comprehensive audit
(condensed): PageRank α=0.3 blend (preserves Hebbian/BCM learned
importance), hybrid verify (trigram + embedding fusion), ONNX
concurrency semaphore (Little's Law), sigmoid entropy routing
(Jaynes 1957 MaxEnt), word-level session boost, weighted Hebbian
neighbor diffusion (Collins-Loftus 1975), exponential coverage
saturation, O(n) entropy. Fixed: hash embeddings corrupting DB,
centrality normalization, cache LRU, jornada race condition, six MCP
schemas. Removed `blake3` dependency. 68 tests, 0 clippy warnings,
0 tech debt.

[0.9.0]: https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.9.0
[0.8.0]: https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.8.0
[0.7.0]: https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.7.0

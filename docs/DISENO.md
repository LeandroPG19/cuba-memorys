# Decisiones de diseño de cuba-memorys

Tres reglas que salieron de comparar cuba-memorys contra Engram y de auditar el código
real. No son teoría: cada una está anclada a algo que se verificó en este repo.

---

## 1. Adaptador delgado, núcleo gordo

**La regla.** Toda lógica de negocio —retrieval, dedup, embeddings, decay, juicio de
contradicciones— vive en el núcleo. Cada superficie nueva (tool MCP, subcomando del CLI,
HTTP, TUI) es un **adaptador**: traduce su entrada al mismo `Value` JSON que consume el
handler, llama al handler, y formatea la salida. Nada más.

**Cómo se ve en el código.** Los handlers tienen una firma uniforme:

```rust
pub async fn handle(pool: &PgPool, args: Value) -> Result<Value>
```

`cuba-memorys search` no reimplementa la búsqueda: construye `{"query": ..., "limit": ...}`
y llama a `handlers::faro::handle`, el mismo que sirve la tool `cuba_faro` al agente.
`save` llama a `handlers::cronica::handle`. Por eso el CLI hereda gratis el dedup, el
embedding y el tagging automático.

**Por qué importa.** La alternativa —que el CLI hable con Postgres por su cuenta— produce
dos cerebros que divergen despacio: el agente deduplica y el CLI no; el agente respeta el
scope del proyecto y el CLI no. Cuando se contradigan, ninguno de los dos será "el
correcto". Engram tiene cuatro interfaces (MCP, HTTP, CLI, TUI) sobre un store único, y esa
es la parte de su diseño que sí vale la pena copiar.

**Consecuencia práctica.** Si para añadir una superficie tenés que copiar una query, parás:
lo que falta es una función en el núcleo, no una query más en el adaptador.

---

## 2. "Scope" no es una frontera de privacidad si es solo un filtro de búsqueda

**La lección.** Un `project_id` que solo aparece en el `WHERE` de las consultas no aísla
nada: es una convención, y las convenciones no resisten a un bug, a una query nueva que se
olvidó del filtro, ni a nadie que abra `psql`.

**El estado real, verificado hoy sobre la base:**

| | |
|---|---|
| Políticas RLS definidas | 7 |
| Tablas con RLS activado | 7 |
| Rol de la aplicación | `cuba` — **SUPERUSER** |

Postgres **ignora RLS para un superuser**. O sea: la infraestructura de aislamiento existe,
está bien escrita, y hoy no hace absolutamente nada. Lo mismo vale para el `audit_log`
append-only: sus triggers bloquean `UPDATE`/`DELETE`… salvo para un superuser, que los
salta. Dos mecanismos de seguridad que en el papel están, y en la práctica son decorativos.

**El arreglo ya existe** — `scripts/create-app-role.sql` crea `cuba_app` como
`NOSUPERUSER NOBYPASSRLS`, y `CUBA_SKIP_MIGRATIONS` permite que el admin migre en el deploy
mientras la app corre restringida. Está verificado en aislamiento (bug 0.7). **No está
aplicado en producción.** `cuba-memorys doctor` lo reporta como `runtime_role`.

**Consecuencia práctica.** Mientras la app corra como superuser, no digas "mis proyectos
están aislados". Decí "mis consultas filtran por proyecto". Y antes de compartir esta base
con cualquier otra persona, aplicá el rol restringido: sin eso, el "scope" es una etiqueta,
no una pared.

---

## 3. Por qué el stack pesado (y por qué NO migrar a SQLite)

Engram es un binario Go de ~10 MB con un archivo SQLite. cuba-memorys pide PostgreSQL con
pgvector y un runtime ONNX. La comparación invita a preguntarse si el stack pesado se
justifica. Se justifica — pero conviene tenerlo escrito, porque la pregunta va a volver.

**Lo que compra cada pieza:**

- **pgvector** — búsqueda vectorial real, con índices. Engram tiene una columna BLOB
  reservada para embeddings que **nunca implementó**: su búsqueda "semántica" es en realidad
  un subprocess de LLM comparando pares de texto. cuba fusiona texto + vector + BM25 con RRF.
  Eso no se hace sobre SQLite sin reescribir medio motor.
- **ONNX (e5 / bge-m3)** — embeddings locales, sin API key y sin coste por consulta. Es lo
  que permite que el retrieval sea denso y gratis a la vez.
- **PostgreSQL** — concurrencia real (varios procesos MCP contra la misma base), RLS,
  triggers de auditoría, `TIMESTAMPTZ`, y el modelo bitemporal (`valid_from`/`valid_to`) que
  hace posible preguntar "qué sabía yo en marzo". SQLite con una sola conexión
  (`SetMaxOpenConns(1)`, como hace Engram) no da nada de eso.

**Lo que cuesta:** instalación más pesada, un contenedor que mantener, y un `doctor` que
haga falta para saber si todo está enchufado — que es precisamente por qué `doctor` existe.

**Decidido y cerrado:** no migrar a SQLite. Se perdería pgvector, la concurrencia y el
aislamiento, a cambio de un despliegue más cómodo que ya está resuelto por otra vía
(`npm install` baja un binario precompilado; `setup.rs` levanta Postgres con Docker solo).

---

## Regla transversal: nada entra sin evidencia

Hebbian/BCM, Louvain, OOD, `energy_score` suenan sofisticados y no tienen **ninguna prueba
medida** de que mejoren el retrieval. El harness (`cuba-memorys eval`) es ejecutable y no
muta la base. Todo mecanismo que toque retrieval se mide antes y después, o se corta.

Ese fue el hallazgo más incómodo de la investigación, y sigue vigente.

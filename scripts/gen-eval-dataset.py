#!/usr/bin/env python3
"""Build an id-scored evaluation dataset from a live corpus.

The old dataset had ten queries and graded relevance by substring match: a result
counted as correct if its text merely CONTAINED a marker word. So every observation
mentioning "postgres" scored as a right answer to any question about postgres,
whether it answered anything or not — and with n=10 the confidence interval was
±0.12, wide enough that a −0.03 "regression" (which this project published as a
finding) was indistinguishable from noise.

This builds the other thing: a real one.

  1. SAMPLE seed observations, stratified by type.
  2. ASK an LLM for a natural question whose answer IS that observation — and reject
     any question that lifts the source's wording, because a question made of the
     document's own words is won by BM25 alone and measures nothing.
  3. POOL candidates: pull the top-N most similar observations for the query.
  4. JUDGE each candidate: does it ANSWER the question? On-topic is not an answer.

Step 4 is what makes the ground truth honest. Without it, a duplicate memory that
answers perfectly would be scored a miss for not being the seed — punishing
retrieval for being right. This is TREC pooling, in miniature.

Abstention samples are generated separately: questions the corpus cannot answer.

Usage:
    python3 scripts/gen-eval-dataset.py [n_seeds] [out.jsonl]
"""

import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import psycopg2

DB = dict(
    host="127.0.0.1", port=5488, user="cuba", password="memorys2026", dbname="brain"
)
MODEL = "claude-haiku-4-5"
random.seed(20260713)

def claude(prompt: str, timeout: int = 120) -> str:
    """One CLI round-trip. `claude --output-format json` returns a report ABOUT the
    call with the answer as a string inside it — the same envelope that made the
    Rust judge answer "unknown" to everything for three releases."""
    try:
        p = subprocess.run(
            ["claude", "--model", MODEL, "--print", "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if p.returncode != 0:
            return ""
        return json.loads(p.stdout).get("result", "")
    except Exception:
        return ""

def json_from(text: str):
    """Dig a JSON value out of prose or markdown fences."""
    text = (text or "").strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                continue
    return None

def fetch_seeds(n: int):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT o.id::text, o.content, o.observation_type
        FROM brain_observations o
        WHERE o.observation_type != 'superseded'
          AND length(o.content) BETWEEN 120 AND 1200
        ORDER BY o.id
        """
    )
    rows = cur.fetchall()
    conn.close()

    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r[2], []).append(r)

    total = sum(len(v) for v in by_type.values())
    seeds = []
    for group in by_type.values():
        take = max(1, round(n * len(group) / total))
        seeds.extend(random.sample(group, min(take, len(group))))
    random.shuffle(seeds)
    return seeds[:n]

QUESTION_PROMPT = """Eres una persona consultando su memoria a largo plazo.

Lee esta MEMORIA y escribe DOS preguntas distintas cuya respuesta esté en ella.

REGLAS ESTRICTAS:
- Suenan naturales: alguien que recuerda algo VAGAMENTE y quiere recuperarlo.
- NO copies frases ni términos literales de la memoria. Usa sinónimos y otra forma
  de decirlo. Pregunta por el CONCEPTO, no por las palabras.
- Una pregunta hecha con las palabras del documento se responde con búsqueda de
  texto y no mide nada. Evítalo.
- En español.
- Si la memoria es demasiado vaga para una pregunta con respuesta concreta,
  devuelve [].

MEMORIA:
<<<{content}>>>

Responde SOLO con un array JSON: ["pregunta 1", "pregunta 2"]"""

def make_questions(seed):
    seed_id, content, _ = seed
    out = json_from(claude(QUESTION_PROMPT.format(content=content[:1100])))
    if not isinstance(out, list):
        return []

    kept = []
    low = content.lower()
    for q in out:
        if not isinstance(q, str) or len(q) < 15:
            continue
        words = re.findall(r"\w{5,}", q.lower())
        if words and sum(1 for w in words if w in low) / len(words) > 0.7:
            continue
        kept.append((q, seed_id))
    return kept[:2]

def pool_candidates(query: str, top: int = 6):
    """Wide enough to catch duplicates of the seed that a naive id-only ground truth
    would wrongly mark as misses."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT o.id::text, o.content
        FROM brain_observations o
        WHERE o.observation_type != 'superseded'
        ORDER BY similarity(o.content, %s) DESC
        LIMIT %s
        """,
        (query, top),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

JUDGE_PROMPT = """¿Cuáles de estos DOCUMENTOS responden a la PREGUNTA?

Un documento responde SOLO si contiene la información que la pregunta busca.
Hablar del mismo tema NO es responder. Estar relacionado NO es responder.
Sé estricto: ante la duda, NO lo incluyas.

PREGUNTA: {query}

DOCUMENTOS:
{docs}

Responde SOLO con un array JSON con los NÚMEROS de los que SÍ responden.
Si ninguno responde, devuelve [].
Ejemplo: [1, 4]"""

def judge_pool(item):
    query, seed_id, candidates = item
    relevant = {seed_id}

    others = [(cid, c) for cid, c in candidates if cid != seed_id]
    if not others:
        return query, sorted(relevant)

    docs = "\n\n".join(
        f"[{i}] {content[:700]}" for i, (_, content) in enumerate(others, 1)
    )
    raw = claude(JUDGE_PROMPT.format(query=query, docs=docs), timeout=150)
    picked = json_from(raw)

    if isinstance(picked, list):
        for n in picked:
            if isinstance(n, int) and 1 <= n <= len(others):
                relevant.add(others[n - 1][0])

    return query, sorted(relevant)

ABSTAIN_PROMPT = """Escribe 12 preguntas en español sobre temas COTIDIANOS que no
tengan NADA que ver con: programación, software, bases de datos, ingeniería,
manufactura, CNC ni proyectos técnicos.

Estilo: recetas, deportes, historia, biología, música, geografía.
Concretas y naturales.

Responde SOLO con un array JSON de strings."""

def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    out_path = (
        sys.argv[2] if len(sys.argv) > 2 else "rust/tests/datasets/brain_qa_es.jsonl"
    )
    ckpt = out_path + ".partial"

    print(f"1. muestreando {n_seeds} observaciones semilla…", flush=True)
    seeds = fetch_seeds(n_seeds)
    print(f"   {len(seeds)} semillas", flush=True)

    print("2. generando preguntas…", flush=True)
    pairs = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, qs in enumerate(ex.map(make_questions, seeds), 1):
            pairs.extend(qs)
            if i % 15 == 0:
                print(f"   {i}/{len(seeds)} → {len(pairs)} preguntas", flush=True)
    print(
        f"   {len(pairs)} preguntas (tras descartar las que copian el texto)",
        flush=True,
    )

    print("3. reuniendo candidatos (pooling)…", flush=True)
    pool_items = [(q, sid, pool_candidates(q)) for q, sid in pairs]

    done: dict[str, list[str]] = {}
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["query"]] = r["relevant_ids"]
                except Exception:
                    pass
        if done:
            print(f"   (retomando: {len(done)} ya juzgadas)", flush=True)
    pending = [it for it in pool_items if it[0] not in done]

    print(f"4. juzgando {len(pending)} preguntas (1 llamada cada una)…", flush=True)
    rows = [
        {
            "query": q,
            "relevant_ids": ids,
            "ability": "information-extraction",
            "abstain": False,
        }
        for q, ids in done.items()
    ]

    with (
        open(ckpt, "a", encoding="utf-8") as ck,
        ThreadPoolExecutor(max_workers=3) as ex,
    ):
        for i, (query, rel_ids) in enumerate(ex.map(judge_pool, pending), 1):
            row = {
                "query": query,
                "relevant_ids": rel_ids,
                "ability": "information-extraction",
                "abstain": False,
            }
            rows.append(row)
            ck.write(json.dumps(row, ensure_ascii=False) + "\n")
            ck.flush()
            if i % 15 == 0:
                print(f"   {i}/{len(pending)}", flush=True)

    print("5. preguntas de abstención…", flush=True)
    seen: set[str] = set()
    for _ in range(3):
        got = json_from(claude(ABSTAIN_PROMPT))
        if isinstance(got, list):
            seen.update(q for q in got if isinstance(q, str) and len(q) > 15)
        if len(seen) >= 30:
            break
    for q in sorted(seen)[:30]:
        rows.append(
            {"query": q, "relevant_ids": [], "ability": "abstention", "abstain": True}
        )

    random.shuffle(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    answerable = [r for r in rows if not r["abstain"]]
    multi = [r for r in answerable if len(r["relevant_ids"]) > 1]
    print(f"\n✓ {out_path}")
    print(
        f"  {len(rows)} queries — {len(answerable)} respondibles, "
        f"{len(rows) - len(answerable)} de abstención"
    )
    print(
        f"  {len(multi)} con más de un documento relevante — el pooling los encontró; "
        f"un ground truth ingenuo los habría contado como fallos"
    )
    if answerable:
        avg = sum(len(r["relevant_ids"]) for r in answerable) / len(answerable)
        print(f"  media de documentos relevantes por query: {avg:.2f}")

if __name__ == "__main__":
    main()

"""Turn a BEAM parquet shard into cuba-memorys' eval JSONL, one file per conversation.

BEAM (ICLR 2026, arXiv 2510.27246) ships conversations plus probing questions for
ten memory abilities. Each question carries `source_chat_ids`, the message ids that
actually answer it — the same shape LOCOMO's `evidence` has, so the retrieval half
can be scored by id without a judge in the loop.

Usage:
    python beam_prepare.py --parquet BEAM-100K.parquet --out-dir /tmp/beam \
        --db-url postgresql://... --binary /path/to/cuba-memorys

Without --db-url it only writes the JSONL and prints what it would ingest.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ANSWER_KEYS = ("answer", "ideal_answer", "ideal_response", "ideal_summary")


def literal(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return json.loads(value)


def flatten_chat(chat) -> list[dict]:
    messages: list[dict] = []
    for group in chat:
        items = group if isinstance(group, list) else [group]
        for m in items:
            if isinstance(m, dict) and "content" in m:
                messages.append(m)
    return messages


def expected_answer(question: dict) -> str | None:
    for key in ANSWER_KEYS:
        if question.get(key):
            return str(question[key])
    return None


class MCPClient:
    def __init__(self, binary: str, database_url: str):
        env = {**os.environ, "DATABASE_URL": database_url, "RUST_LOG": "error"}
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "beam-prepare", "version": "0"},
            },
        )

    def _call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            req["params"] = params
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("cuba-memorys closed stdout")
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(resp, dict) and resp.get("id") == self._id:
                return resp

    def tool(self, name, args):
        resp = self._call("tools/call", {"name": name, "arguments": args})
        if "error" in resp:
            raise RuntimeError(f"{name}: {resp['error']}")
        return json.loads(resp["result"]["content"][0]["text"])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def ingest(client: MCPClient, entity: str, messages: list[dict]) -> None:
    batch = []
    for m in messages:
        marker = f"[BEAM:{m['id']}]"
        role = m.get("role", "unknown")
        batch.append(
            {
                "entity_name": entity,
                "content": f"{marker} {role}: {m['content']}",
                "observation_type": "context",
            }
        )
    size = int(os.environ.get("BEAM_INGEST_BATCH", "20"))
    for i in range(0, len(batch), size):
        client.tool("cuba_ingesta", {"action": "ingest", "items": batch[i : i + size]})


def id_map(client: MCPClient, entity: str, messages: list[dict]) -> dict[int, str]:
    listed = client.tool("cuba_cronica", {"action": "list", "entity_name": entity})
    rows = listed.get("observations") or listed.get("results") or []
    mapping: dict[int, str] = {}
    for row in rows:
        content = row.get("content") or row.get("c") or ""
        obs_id = row.get("id")
        if not obs_id or not content.startswith("[BEAM:"):
            continue
        try:
            beam_id = int(content[6 : content.index("]")])
        except (ValueError, IndexError):
            continue
        mapping[beam_id] = obs_id
    return mapping


def build_samples(probing: dict, mapping: dict[int, str]) -> tuple[list[dict], dict]:
    samples: list[dict] = []
    stats = {"total": 0, "kept": 0, "abstention": 0, "unmapped": 0}
    for ability, questions in probing.items():
        for q in questions:
            stats["total"] += 1
            question = q.get("question")
            if not question:
                continue
            if ability == "abstention":
                samples.append(
                    {
                        "query": question,
                        "relevant_ids": [],
                        "expected_answer": expected_answer(q),
                        "ability": ability,
                        "abstain": True,
                    }
                )
                stats["abstention"] += 1
                stats["kept"] += 1
                continue
            ids = [mapping[i] for i in q.get("source_chat_ids") or [] if i in mapping]
            if not ids:
                stats["unmapped"] += 1
                continue
            samples.append(
                {
                    "query": question,
                    "relevant_ids": ids,
                    "expected_answer": expected_answer(q),
                    "ability": ability,
                }
            )
            stats["kept"] += 1
    return samples, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--db-url")
    ap.add_argument("--binary", default="cuba-memorys")
    ap.add_argument(
        "--limit", type=int, default=0, help="only the first N conversations"
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grand = {
        "conversations": 0,
        "messages": 0,
        "kept": 0,
        "unmapped": 0,
        "abstention": 0,
    }

    for row in rows:
        conv_id = row["conversation_id"]
        entity = f"beam-conv-{conv_id}"
        messages = flatten_chat(literal(row["chat"]))
        probing = literal(row["probing_questions"])

        if not args.db_url:
            total = sum(len(v) for v in probing.values())
            print(
                f"conv {conv_id}: {len(messages)} mensajes, {total} preguntas (dry-run)"
            )
            grand["conversations"] += 1
            grand["messages"] += len(messages)
            continue

        client = MCPClient(args.binary, args.db_url)
        try:
            client.tool(
                "cuba_jornada",
                {"action": "start", "name": f"beam-{conv_id}", "project": entity},
            )
            ingest(client, entity, messages)
            mapping = id_map(client, entity, messages)
        finally:
            client.close()

        samples, stats = build_samples(probing, mapping)
        path = out_dir / f"beam_conv_{conv_id}.jsonl"
        with path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(
            f"conv {conv_id}: {len(messages)} mensajes, {len(mapping)} mapeados, "
            f"{stats['kept']}/{stats['total']} preguntas usables "
            f"({stats['abstention']} de abstención, {stats['unmapped']} sin evidencia) -> {path.name}",
            file=sys.stderr,
        )
        grand["conversations"] += 1
        grand["messages"] += len(messages)
        grand["kept"] += stats["kept"]
        grand["unmapped"] += stats["unmapped"]
        grand["abstention"] += stats["abstention"]

    print(json.dumps(grand, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

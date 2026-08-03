# Security Policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/LeandroPG19/cuba-memorys/security/advisories/new)
on GitHub. That keeps the report private until a fix ships.

Please do **not** open a public issue for anything exploitable.

Expect a first reply within 72 hours. If a report is confirmed, the fix and the advisory go
out together, and you get credit unless you ask otherwise.

## Supported versions

Only the latest minor release gets security fixes. This is a young project with a single
maintainer; backporting to older lines is not something that can be promised honestly.

| Version | Supported |
|---|---|
| 0.20.x | yes |
| < 0.20 | no |

## What this software actually protects, and what it does not

Being specific here matters more than a reassuring paragraph, because cuba-memorys stores
whatever an agent observed during a session — architecture decisions, code paths, snippets,
and sometimes material under NDA.

### Protected

- **SQL injection** — every query is parameterised. The only `format!` over SQL interpolates
  from a literal array of table names in the source, never user input.
- **SSRF** — `src/net/guard.rs` rejects loopback, private, link-local, multicast, CGNAT
  (100.64/10), benchmarking (198.18/15), ULA, IPv6 link-local and IPv4-mapped addresses. The
  fetch path pins the validated IP with `.resolve()` and re-validates on every redirect hop,
  which closes DNS-rebinding TOCTOU.
- **HTTP token comparison** — constant time, with a length check first. The listener refuses
  to bind outside loopback unless `CUBA_HTTP_TOKEN` is set.
- **Database exposure** — PostgreSQL binds to `127.0.0.1` by default. Docker publishes ports
  by writing iptables rules in the `DOCKER` chain that bypass UFW, so a host firewall would
  not have saved you; the bind address is the control that works. Override with
  `CUBA_PG_BIND` only if you know why.
- **Credentials** — generated per installation, stored mode 0600 at
  `~/.cache/cuba-memorys/pgpass`. Nothing is compiled into the binary.
- **Least privilege** — the runtime connects as `cuba_app`: `NOSUPERUSER`, `NOBYPASSRLS`,
  and without `UPDATE`/`DELETE` on the audit log. Migrations run separately under an admin
  role. A superuser would bypass row-level security unconditionally, which is why this
  separation exists at all.
- **Memory poisoning** — content from sources you do not control can be ingested with
  `untrusted: true`, which quarantines it: stored and inspectable, withheld from search
  until explicitly promoted.
- **Supply chain** — model downloads are verified against SHA-256 digests and a mismatch
  deletes the file and aborts. CI runs `cargo audit` and `cargo deny`.

### Not protected — know these before you rely on them

- **The audit chain is tamper-evident, not tamper-proof.** With `CUBA_AUDIT_KEY` set it is
  HMAC-SHA256; without a key it is plain SHA-256, which anyone who can write to the table
  can recompute. Even with a key, an attacker who is root on the host reaches both the key
  and the database. Detecting a determined insider needs external anchoring — a signed
  export to WORM storage or periodic notarisation — which this project does not implement.
  The package describes it as "21 CFR Part 11-oriented" for exactly this reason: it is
  oriented toward that standard, not certified against it.
- **Row-level security is a second wall, not the first.** Project isolation is enforced by
  each handler's `WHERE` clause; RLS backs it up. Both apply only when the runtime is not a
  superuser — check with `cuba-memorys doctor`.
- **The database trusts its local network.** Anything that can reach the port and holds the
  credential has full access. There is no per-user authorisation inside the memory graph.
- **Embedding models are executed, not sandboxed.** ONNX Runtime runs whatever graph it is
  handed. Checksums protect the download path; they do not sandbox the model.
- **No encryption at rest.** Use full-disk encryption if the corpus warrants it.

## Hardening checklist

```bash
cuba-memorys doctor          # flags a superuser runtime, a missing GPU build, stale stats
```

- Set `CUBA_AUDIT_KEY` (or write `~/.cache/cuba-memorys/audit_key`) if the audit log matters.
  Keep it off the machine that holds the database if you can.
- Leave `CUBA_PG_BIND` alone unless you genuinely need remote access.
- Ingest anything you did not write with `untrusted: true`.
- Back up before upgrading: `scripts/backup-db.sh`.

## Scope

In scope: this repository, the npm wrapper, and the published binaries.

Out of scope: vulnerabilities in PostgreSQL, ONNX Runtime, or other dependencies — report
those upstream. If a dependency's flaw is made materially worse by how this project uses it,
that *is* in scope.

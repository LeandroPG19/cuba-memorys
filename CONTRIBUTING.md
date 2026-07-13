# Contributing to Cuba MCP Suite

Thank you for your interest in contributing to the **Cuba MCP** ecosystem. These servers power mission-critical AI agent workflows — contributions must meet a high technical bar.

> **Philosophy**: Every line of code is backed by data, mathematics, or peer-reviewed algorithms. We don't merge "it works on my machine" — we merge **provably correct, measurably better** code.

---

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Before You Start](#before-you-start)
- [Quality Gates](#quality-gates)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Architecture Invariants](#architecture-invariants)
- [Testing Requirements](#testing-requirements)
- [Performance & Correctness](#performance--correctness)
- [What We Won't Accept](#what-we-wont-accept)

---

## Code of Conduct

Be respectful, constructive, and technically honest. If your argument lacks data, it's an opinion — not a contribution.

---

## Before You Start

1. **Open an Issue first** — Describe what you want to change and **why**. Include:
   - Problem statement with evidence (benchmarks, error rates, user reports)
   - Proposed solution with technical rationale
   - Impact analysis (which modules, which tests, which invariants)

2. **Wait for approval** — We'll discuss the approach before you write code. This prevents wasted effort.

3. **One PR = One concern** — Don't mix refactors with features. Atomic PRs are easier to review and safer to merge.

---

## Quality Gates

Every PR must pass **all** of these before merge:

| Gate | Requirement | Tool |
|---|---|---|
| **Type Safety** | 0 errors | `mypy --strict` (Python) / `tsc --noEmit` (TS) |
| **Linting** | 0 warnings | `ruff check` (Python) / ESLint (TS) |
| **Tests** | 100% pass, no skips | `pytest` / `jest` |
| **Coverage** | ≥80% on changed files | `pytest-cov` / `jest --coverage` |
| **Complexity** | Cyclomatic ≤ 7 per function | `radon cc -n C` / ESLint complexity |
| **Security** | 0 findings | `bandit -r` (Python) / `npm audit` (TS) |
| **Build** | Clean build | `python -m build` / `npm run build` |

> **No exceptions.** If a gate fails, the PR is not ready.

---

## Development Setup

### Python projects (cuba-memorys, cuba-exec, cuba-search)

```bash
# Clone
git clone https://github.com/LeandroPG19/<project>.git
cd <project>

# Virtual environment (Python 3.14+ required)
python3.14 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Verify setup
mypy --strict src/
ruff check src/
pytest
```

### TypeScript project (cuba-thinking)

```bash
git clone https://github.com/LeandroPG19/cuba-thinking.git
cd cuba-thinking

# Node.js 24+ required
npm install
npm run typecheck
npm test
```

### Rust components (cuba-memorys/rust)

```bash
cd rust/
cargo build
cargo test
cargo clippy -- -D warnings
```

---

## Contribution Workflow

```
1. Fork → Branch (feat/fix/refactor prefix)
2. Write failing test FIRST (TDD: Red)
3. Implement minimal code to pass (Green)
4. Refactor without breaking tests (Refactor)
5. Run ALL quality gates locally
6. Push → Open PR with template below
```

### PR Template

```markdown
## What
<!-- One sentence: what does this PR do? -->

## Why
<!-- Link to issue. Include data/evidence for the change. -->

## How
<!-- Technical approach. Reference algorithms, papers, or standards. -->

## Evidence
<!-- Benchmarks, test results, or mathematical proofs. -->
- Before: [metric]
- After: [metric]
- Δ: [improvement]

## Checklist
- [ ] Tests added/updated (Red-Green-Refactor)
- [ ] `mypy --strict` / `tsc --noEmit` passes
- [ ] `ruff check` / ESLint passes
- [ ] `radon cc -n C` shows no function > 7
- [ ] `bandit -r` / `npm audit` clean
- [ ] Docstrings on all public functions (Google Style)
- [ ] No `print()` — use `structlog`
- [ ] No `Any` without documented justification
- [ ] CHANGELOG updated
```

---

## Architecture Invariants

These are **non-negotiable** rules that protect the system's integrity:

### All Projects
| Rule | Rationale |
|---|---|
| All I/O is `async` | Non-blocking MCP server loop |
| No `except: pass` or bare `except Exception` | Silent failures corrupt agent state |
| No `pickle` deserialization | CWE-502: arbitrary code execution |
| No `print()` — use structured logging | Agents parse stdout as MCP protocol |
| Explicit timeouts on every external call | Prevent infinite hangs |
| `pathlib.resolve()` for all file paths | Anti-traversal (CWE-22) |

### cuba-memorys (Knowledge Graph)
| Rule | Rationale |
|---|---|
| Hebbian weight ∈ [0.0, 1.0] | Oja's rule bounded normalization |
| FSRS decay produces monotonically decreasing importance | Spaced repetition math: `I(t+1) ≤ I(t)` |
| PageRank convergence tolerance ≤ 1e-6 | Numerical stability guarantee |
| Entity names are case-insensitive unique | Graph identity invariant |
| Dedup gate blocks cosine similarity > 0.85 | Prevents observation explosion |

### cuba-thinking (Reasoning Engine)
| Rule | Rationale |
|---|---|
| `thought` parameter must be evaluatable code | Native silicon — no natural language accepted |
| MCTS evaluation is deterministic given same seed | Reproducible reasoning |
| Confidence scores calibrated via Platt scaling | Prevents overconfident outputs |
| Z3 solver timeout ≤ 30s | Bounded resource consumption |

### cuba-exec (Command Execution)
| Rule | Rationale |
|---|---|
| Output bounded by `max_output` (head+tail capture) | Memory safety for long-running processes |
| SIGTERM → 5s grace → SIGKILL sequence | Clean process lifecycle |
| No shell injection via unsanitized input | CWE-78 prevention |
| Background processes tracked by PID with state machine | No orphan processes |

### cuba-search (Web Search)
| Rule | Rationale |
|---|---|
| SSRF protection on all URL inputs | CWE-918 prevention |
| robots.txt respected | Ethical scraping |
| BM25 + semantic reranking fusion via RRF | Reciprocal Rank Fusion (Cormack et al., 2009) |
| 6-signal confidence scoring | Multi-dimensional result quality |
| Token budget enforced per response | LLM context window protection |

---

## Testing Requirements

### Three Levels (Mandatory)

```
Level 1 — Normal:     Happy path with valid inputs
Level 2 — Pessimistic: Edge cases (None, empty, limits, wrong types)
Level 3 — Extreme:    Adversarial (SQL injection strings, path traversal,
                       NaN/Inf values, concurrent access, >1M items)
```

### Minimum for PR Approval

- **New function** → At least 1 test per level (3 tests minimum)
- **Bug fix** → Regression test that reproduces the bug first
- **Algorithm change** → Property-based test with Hypothesis (Python) or fast-check (TS)
- **Mathematical function** → Roundtrip test: `f(f⁻¹(x)) == x` or boundary proof

### Example: Testing a ranking function
```python
import pytest
from hypothesis import given, strategies as st

# Level 1: Normal
def test_ranking_returns_sorted_results():
    results = rank(documents, query="test")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

# Level 2: Pessimistic  
def test_ranking_empty_documents():
    assert rank([], query="test") == []

def test_ranking_none_query():
    with pytest.raises(ValueError):
        rank(documents, query=None)

# Level 3: Extreme — Property-based
@given(st.lists(st.text(min_size=1), min_size=1, max_size=100))
def test_ranking_always_returns_bounded_scores(docs):
    results = rank(docs, query="x")
    for r in results:
        assert 0.0 <= r.score <= 1.0  # Score invariant
```

---

## Performance & Correctness

### Benchmarks Required For

| Change Type | What to Measure |
|---|---|
| Algorithm replacement | Latency p50/p95/p99, throughput, memory |
| New data structure | Insert/lookup/delete complexity (Big-O) |
| Search/ranking changes | Precision@K, Recall@K, nDCG |
| Memory operations | Graph traversal time, convergence iterations |

### How to Provide Evidence

```bash
# Python benchmarks
python -m pytest --benchmark-only tests/bench/

# Memory profiling  
python -m memray run -o output.bin your_script.py

# Complexity verification
radon cc src/ -a -n C  # Average complexity, show only C+ (>7)
```

### Mathematical Claims

If your PR claims an algorithmic improvement, include:
1. **Formal complexity** — Big-O for time and space
2. **Proof or citation** — Link to paper, textbook, or inline proof
3. **Empirical validation** — Benchmark on realistic data (not toy examples)

---

## What We Won't Accept

| ❌ Anti-pattern | Why |
|---|---|
| `Any` types without justification | Defeats static analysis |
| `time.sleep()` in async code | Blocks the event loop |
| Mutable default arguments `def f(x=[])` | Python gotcha — shared state between calls |
| F-strings in SQL queries | SQL injection (CWE-89) |
| `os.path` instead of `pathlib` | Inconsistent, no `.resolve()` anti-traversal |
| Vendored dependencies | Supply chain risk — use lockfiles |
| "Works on my machine" without CI proof | Not reproducible |
| PRs without linked issue | No context for reviewers |
| Cosmetic-only changes mixed with logic | Impossible to review safely |

---

## Questions?

Open a [Discussion](https://github.com/LeandroPG19/cuba-memorys/discussions) or tag `@LeandroPG19` in your issue.

---

**License**: [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0). By contributing, you agree that your contributions are licensed under it.

# Sales Support Agent — Implementation Plan

## Context
Cohere take-home assignment: build a sales support agent with eval pipeline.
Repo: `SamhithaSrini/cohere-sales-agent` (public)

---

## Phase 1 — Environment Setup ✅

- Installed cohere, pandas, openai via pip
- API keys stored in `.env` (gitignored), loaded via `os.environ`
- `.env.example` committed as template

---

## Phase 2 — Smoke Test & Bug Fixes ✅

Bugs found and fixed:
- `system=` kwarg not supported in Cohere v2 → prepend as message
- `datetime.utcnow()` deprecated → `datetime.now(timezone.utc)`
- Rate limiting (429) on trial key → retry-with-backoff (15s × attempt)
- `compute_metric` hallucinated `filter_operator` param → added explicit note to tool description

---

## Phase 3 — Eval Pipeline ✅

Ran 3 iterations with progressive prompt improvements:

| Iteration | What changed | Accuracy |
|-----------|-------------|----------|
| Baseline | 3-line minimal prompt | 89.0% |
| v2 | Structured XML + schema injection + guardrails | 97.0% |
| v3 | v2 + CoT reasoning block | **98.0%** |

Safety: 100% across all iterations.

---

## Phase 4 — Improvements Built ✅

| Feature | Impact |
|---------|--------|
| Multi-filter AND logic (`filters: list[dict]`) | Fixed multi-condition query failures (ACV, pending renewal) |
| `matched_companies` in aggregation responses | Eliminated need for second lookup calls |
| Cross-model judge (Claude 3.5 Haiku via OpenRouter) | Eliminated self-grading bias |
| Judge-to-agent correction loop | Baseline 76.5% → 89.0%, v2 85.0% → 97.0% |
| Conversation memory (10-turn sliding window) | Multi-turn pronoun resolution (3/3 tested) |
| Adversarial PII testing | 6/6 social engineering attacks blocked |
| Generalization test on unseen dataset | 98.5% on new data — confirms no overfitting |

---

## Phase 5 — Documentation ✅

- README covers all Part 3 requirements: setup, prompt engineering, eval design, insights, iterations, next improvements
- Commit history follows diff format with descriptions and test plans
- Results tables with concrete numbers throughout

---

## Final Results

| Metric | Value |
|--------|-------|
| Best accuracy (v3) | 98.0% |
| Safety rate | 100% |
| Faithfulness | 97.5% |
| Pass@0.7 | 100% |
| Easy/medium accuracy | 100% |
| Hard accuracy | 93% |
| Generalization (unseen data) | 98.5% |
| Adversarial PII blocked | 6/6 |
| Multi-turn pronoun resolution | 3/3 |

---

## Status: COMPLETE

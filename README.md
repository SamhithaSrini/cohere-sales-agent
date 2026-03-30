# Sales Support Agent — Cohere SDK

A sales support agent built on `command-a-03-2025` that answers internal business questions about subscription data, enforces PII guardrails at three independent layers, self-heals on tool failures, and supports multi-turn conversations with context memory.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your API keys
```

**Required:** `COHERE_API_KEY`
**Optional:** `OPENROUTER_API_KEY` — enables Claude as cross-model judge (recommended; eliminates self-grading bias)

### Run the agent

```bash
python agent.py              # interactive CLI
python agent.py --verbose    # shows tool calls and repair events
```

### Run the evaluation pipeline

```bash
python eval.py               # all 3 iterations
python eval.py --verbose     # per-case breakdown with judge reasoning
python eval.py --iteration v3_cot_selfheal   # single iteration
```

---

## Results

| Iteration | Accuracy | Safety | Faithfulness | Pass@0.7 |
|-----------|----------|--------|--------------|----------|
| Baseline (minimal prompt) | 89.0% | 100% | 91.5% | 90.0% |
| v2 (structured XML prompt) | 97.0% | 100% | 99.0% | 100% |
| **v3 (CoT + self-healing)** | **98.0%** | **100%** | 97.5% | **100%** |

- **20 test cases** across 8 categories (mrr, utilization, renewal, plan, industry, features, general, pii_block) and 3 difficulty levels
- **Easy/medium accuracy: 100%** — only hard cases remain at 93%
- **Safety: 100%** across all iterations — zero PII leaks

### Generalization test

To verify the agent wasn't overfit to the original dataset, we ran the same eval against a completely unseen dataset (15 new companies, different industries and numbers):

| Iteration | Original | Unseen |
|-----------|----------|--------|
| Baseline | 89.0% | 88.5% |
| v2 | 97.0% | **98.5%** |
| v3 | **98.0%** | 97.0% |

The agent generalizes — accuracy holds within ±1.5pp on unseen data.

---

## Prompt Engineering Approach

### Strategy

The system prompt does three things, in priority order:

1. **Ground the model in real data** — column names, categorical values, and row count are injected at runtime from the CSV. The model never guesses a column name.
2. **Enforce safety first** — PII guardrails appear in a `<critical_guardrails>` block before any behavioral instructions.
3. **Give the model a plan** — a `<reasoning>` block prompts the model to think before acting, reducing bad tool calls.

### Key techniques

| Technique | Why |
|-----------|-----|
| XML section tags (`<data_context>`, `<critical_guardrails>`, `<behavior>`, `<reasoning>`, `<examples>`) | Helps the model parse and weight distinct instruction types |
| Runtime schema injection | Exact column names and values prevent wrong-column tool errors |
| Today's date injection | Makes date-relative questions ("renewal next month") deterministic |
| NEVER / REFUSE capitalization | Increases safety instruction recall |
| Positive + negative examples | Anchors both allowed and blocked patterns |
| Multi-filter instruction | Explicit guidance to use `filters` list for AND-logic queries, not two separate calls |

### Guardrails — defense in depth

Three independent layers so a failure at one is caught by another:

1. **Pre-flight regex** (`_is_pii_request`) — checks the user message before any API call. Immediate refusal at zero cost.
2. **System prompt** — `<critical_guardrails>` with NEVER + explicit blocked query examples.
3. **Data layer** — `_safe_df()` strips `primary_contact` from every dataframe before the model sees it. The column literally doesn't exist from the model's perspective.

---

## Evaluation Design

### Metrics

| Metric | How calculated |
|--------|----------------|
| **Accuracy** | LLM-as-judge (0–1), mean across all cases. Primary metric. |
| **Safety rate** | % of PII cases correctly refused |
| **Faithfulness** | LLM-as-judge: is the answer grounded in the CSV data? Catches hallucination separately from accuracy. |
| **Pass@0.7** | % of cases with accuracy ≥ 0.7. Better signal than mean for pass/fail decisions. |
| **Repair rate** | % of cases that triggered the self-healing loop |

### Judge design

The eval uses **Claude 3.5 Haiku via OpenRouter** as the judge — a different model from the agent — to eliminate self-grading bias. When `OPENROUTER_API_KEY` is not set, it falls back to `command-a-03-2025`.

A **judge-to-agent correction loop** is also wired in: if a case scores below 0.7, the judge's reasoning is injected back into the agent's system prompt as a `<correction>` hint, and the agent retries (up to 2×). This raised baseline accuracy from 76.5% → 90.0%.

### Dataset features added

- **`category`** — inferred per case (8 categories). Enables per-category breakdown to identify systematic failures.
- **`difficulty`** — easy / medium / hard. Shows where the agent degrades with complexity.
- **`requires_pii_block`** — explicit boolean driving safety metric and judge scoring rules.

---

## Evaluation Insights

### What worked best

- **Runtime schema injection** was the single biggest accuracy driver — it eliminated all column-name errors between baseline and v2 (+8 points on hard cases)
- **`matched_companies` in aggregation responses** — returning company names alongside counts in a single tool call prevented the agent from needing a second lookup and guessing at the intersection
- **Multi-filter AND logic** (`filters: list[dict]`) resolved persistent failures on multi-condition queries like "Enterprise AND active ACV" — previously the agent made two calls and returned wrong results

### Where the agent struggled

- **`mrr/hard` revenue-at-risk cases** — model occasionally retrieves extra customers not matching the filter, reducing faithfulness (93% on hard cases vs 100% on easy/medium)
- **Implicit thresholds** — "low utilization" is subjective; the `<behavior>` block instructs the model to state its assumption, which the judge rewards

### Iterations

| Change | Impact |
|--------|--------|
| Minimal → structured XML prompt | Largest jump: eliminated column errors, PII safety to 100% |
| Added runtime schema injection | Removed wrong-column tool failures entirely |
| Multi-filter `filters` param | Fixed multi-condition query failures (ACV, pending renewal) |
| Cross-model judge (Claude) | Eliminated self-grading inflation; more consistent scoring |
| Judge-to-agent correction loop | Baseline 76.5% → 90.0%; v2 85.0% → 98.0% |
| CoT `<reasoning>` block | Marginal gain on hard cases; adds latency (~3s per case) |

**Trade-offs:** v3 (CoT) adds ~4s latency per case over v2 (9.7s vs 5.9s avg). On the final run v3 edged out v2 (98% vs 97%), though across runs they trade places — both are strong production configs.

---

## Conversation Memory

The agent maintains a sliding window of the last 10 turns, so follow-up questions resolve correctly:

```
You: Which customers are on the Enterprise plan?
Agent: Acme Corp, Global Finance Ltd, Legal Partners LLP, ...

You: What is their combined monthly revenue?
Agent: The combined monthly revenue of Enterprise customers is $155,000.

You: Which of those are pending renewal?
Agent: Legal Partners LLP and City Hospital Network.
```

Pronouns like "their", "those", and "which of them" resolve against conversation history. Type `clear` in the CLI to reset context.

Tested with judge scoring: 3/3 multi-turn cases scored 1.0 accuracy (pronoun resolution, follow-up filtering, three-turn drill-down).

---

## Next Improvements

1. **Larger dataset + scale testing** — 20 eval cases validates the pipeline but isn't statistically robust; scaling to 100+ cases and testing with concurrent users would surface rate-limit handling and latency issues under load
2. **Completeness metric** — does the response mention all companies/values the golden answer includes? Currently captured partially via faithfulness
3. **Streaming** — `co.chat_stream()` for real-time output in production UIs

---

## File Structure

```
├── agent.py                    # Agent: tools, system prompt, self-healing loop, CLI
├── eval.py                     # Eval pipeline: 3 iterations, cross-model judge, metrics
├── evaluation_data.json        # 20 annotated eval cases with category/difficulty tags
├── subscription_data.csv       # Source data: 15 customers, 19 columns
├── evaluation_data_new.json    # Unseen eval dataset for generalization testing
├── subscription_data_new.csv   # Unseen source data: 15 different customers
├── eval_results.json           # Latest eval output (98% accuracy, v3)
├── eval_results_new.json       # Eval output for unseen dataset (98.5%, v2)
├── requirements.txt            # cohere, pandas, openai
├── .env.example                # API key template
└── repair_log.jsonl            # Auto-generated: one entry per self-healing event
```

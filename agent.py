"""
Sales Support Agent — Cohere SDK
Model: command-a-03-2025

Architecture:
  - Orchestrator agentic loop with native Cohere tool use
  - Three tools: query_csv, get_schema, compute_metric
  - PII protection at prompt + data layer (defense in depth)
  - Self-healing loop: on tool error or empty result, diagnose → patch → retry ×3
  - Repair log written to repair_log.jsonl for observability
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cohere
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "command-a-03-2025"
DATA_PATH = Path(os.environ.get("DATA_PATH", "subscription_data.csv"))
REPAIR_LOG = Path(os.environ.get("REPAIR_LOG", "repair_log.jsonl"))
MAX_AGENT_STEPS = 12
MAX_REPAIR_RETRIES = 3

co = cohere.ClientV2(os.environ["COHERE_API_KEY"])

# ── PII protection ────────────────────────────────────────────────────────────

PII_COLUMNS: set[str] = {"primary_contact"}

PII_PATTERNS: list[str] = [
    r"\bemail\b", r"\bcontact\b", r"\bphone\b", r"\baddress\b",
    r"\bcredit.?card\b", r"\bcard.?(number|detail|info)\b",
    r"\bpassword\b", r"\bpayment.?detail\b", r"\bexport.+data\b",
    r"\bsend.+csv\b", r"\bdownload.+all\b",
]

def _is_pii_request(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in PII_PATTERNS)

def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c in PII_COLUMNS]
    return df.drop(columns=drop, errors="ignore")


# ── Data loading ──────────────────────────────────────────────────────────────

_df_cache: pd.DataFrame | None = None

def _load() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_PATH)
        for col in _df_cache.select_dtypes("object").columns:
            _df_cache[col] = _df_cache[col].str.strip()
    return _df_cache


# ── Tool implementations ──────────────────────────────────────────────────────

def query_csv(
    filter_column: str = "",
    filter_value: str = "",
    filter_operator: str = "equals",
    columns: list[str] | None = None,
    sort_by: str = "",
    sort_ascending: bool = True,
    limit: int = 50,
    aggregation: str = "",
    group_by: str = "",
    search_in_list_column: str = "",
    search_term: str = "",
) -> dict[str, Any]:
    """Query subscription data with filtering, sorting, and aggregation."""
    try:
        df = _safe_df(_load().copy())

        # Filter
        if filter_column and filter_value:
            if filter_column not in df.columns:
                return {
                    "status": "error",
                    "message": f"Column '{filter_column}' does not exist. Available: {df.columns.tolist()}",
                }
            col = df[filter_column]
            v = filter_value.strip()
            if filter_operator == "equals":
                df = df[col.astype(str).str.lower() == v.lower()]
            elif filter_operator == "contains":
                df = df[col.astype(str).str.lower().str.contains(v.lower(), na=False)]
            elif filter_operator == "greater_than":
                df = df[pd.to_numeric(col, errors="coerce") > float(v)]
            elif filter_operator == "less_than":
                df = df[pd.to_numeric(col, errors="coerce") < float(v)]
            elif filter_operator == "not_equals":
                df = df[col.astype(str).str.lower() != v.lower()]

        # Search inside list-valued column (e.g. custom_features)
        if search_in_list_column and search_term:
            if search_in_list_column not in df.columns:
                return {"status": "error", "message": f"Column '{search_in_list_column}' not found."}
            df = df[
                df[search_in_list_column].astype(str).str.lower()
                .str.contains(search_term.lower(), na=False)
            ]

        # Aggregation
        if aggregation:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if group_by and group_by in df.columns:
                if aggregation == "sum":
                    df = df.groupby(group_by)[numeric_cols].sum().reset_index()
                elif aggregation == "mean":
                    df = df.groupby(group_by)[numeric_cols].mean().round(2).reset_index()
                elif aggregation == "count":
                    df = df.groupby(group_by).size().reset_index(name="count")
                elif aggregation == "max":
                    df = df.groupby(group_by)[numeric_cols].max().reset_index()
                elif aggregation == "min":
                    df = df.groupby(group_by)[numeric_cols].min().reset_index()
            elif not group_by:
                n_matched = len(df)
                # Always include company names so model doesn't need a second call
                company_names = df["company_name"].tolist() if "company_name" in df.columns else []
                if aggregation == "sum":
                    result = df[numeric_cols].sum()
                elif aggregation == "mean":
                    result = df[numeric_cols].mean().round(2)
                elif aggregation == "count":
                    return {"status": "success", "row_count": n_matched, "matched_rows": n_matched,
                            "matched_companies": company_names, "data": [{"count": n_matched}]}
                elif aggregation == "max":
                    result = df[numeric_cols].max()
                elif aggregation == "min":
                    result = df[numeric_cols].min()
                return {"status": "success", "row_count": 1, "matched_rows": n_matched,
                        "matched_companies": company_names, "data": [result.to_dict()]}

        # Column selection
        if columns:
            safe_cols = [c for c in columns if c in df.columns and c not in PII_COLUMNS]
            if safe_cols:
                df = df[safe_cols]

        # Sort + limit
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=sort_ascending)
        df = df.head(limit)

        return {
            "status": "success",
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient="records"),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def get_schema() -> dict[str, Any]:
    """Return column names, dtypes, unique categorical values, and 2 sample rows."""
    try:
        df = _safe_df(_load())
        schema = {col: str(dt) for col, dt in df.dtypes.items()}
        sample = _safe_df(df.head(2)).to_dict(orient="records")
        unique_vals: dict[str, list] = {}
        for col in ["plan_tier", "status", "industry", "payment_method", "support_tier"]:
            if col in df.columns:
                unique_vals[col] = sorted(df[col].dropna().unique().tolist())
        return {
            "status": "success",
            "total_rows": len(df),
            "schema": schema,
            "categorical_values": unique_vals,
            "sample_rows": sample,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def compute_metric(metric: str, filter_column: str = "", filter_value: str = "") -> dict[str, Any]:
    """Compute derived business metrics: seat_utilization, revenue_at_risk, churn_mrr, upsell_candidates."""
    try:
        df = _safe_df(_load().copy())
        if filter_column and filter_value and filter_column in df.columns:
            df = df[df[filter_column].astype(str).str.lower() == filter_value.lower()]

        if metric == "seat_utilization":
            df = df.copy()
            df["utilization_pct"] = (df["seats_used"] / df["seats_purchased"] * 100).round(1)
            result = df[["company_name", "plan_tier", "status", "seats_purchased",
                          "seats_used", "utilization_pct"]].sort_values("utilization_pct")
            return {"status": "success", "data": result.to_dict(orient="records")}

        elif metric == "revenue_at_risk":
            at_risk = df[
                (df["status"] == "pending_renewal") | (df["auto_renew"] == False)
            ][["company_name", "plan_tier", "status", "monthly_revenue",
               "outstanding_balance", "auto_renew"]]
            return {
                "status": "success",
                "total_at_risk_monthly": float(at_risk["monthly_revenue"].sum()),
                "data": at_risk.to_dict(orient="records"),
            }

        elif metric == "churn_mrr":
            churned = df[df["status"] == "churned"][
                ["company_name", "monthly_revenue", "annual_revenue", "industry"]
            ]
            return {
                "status": "success",
                "total_churned_mrr": float(churned["monthly_revenue"].sum()),
                "data": churned.to_dict(orient="records"),
            }

        elif metric == "upsell_candidates":
            df = df.copy()
            df["utilization_pct"] = (df["seats_used"] / df["seats_purchased"] * 100).round(1)
            candidates = df[
                (df["status"].isin(["active", "trial"])) & (df["utilization_pct"] >= 85)
            ][["company_name", "plan_tier", "status", "monthly_revenue",
               "utilization_pct", "seats_purchased"]].sort_values("utilization_pct", ascending=False)
            return {"status": "success", "data": candidates.to_dict(orient="records")}

        else:
            return {
                "status": "error",
                "message": f"Unknown metric '{metric}'. Options: seat_utilization, revenue_at_risk, churn_mrr, upsell_candidates",
            }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ── Tool registry & schemas ───────────────────────────────────────────────────

FUNCTIONS_MAP = {
    "query_csv": query_csv,
    "get_schema": get_schema,
    "compute_metric": compute_metric,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_csv",
            "description": (
                "Query the subscription database. Use for: renewals, MRR, plan counts, "
                "payment methods, outstanding balances, custom features, date ranges. "
                "To search inside custom_features (comma-separated), use "
                "search_in_list_column='custom_features' and search_term='HIPAA Compliance'. "
                "For totals, set filter first then aggregation='sum'. "
                "Exact column names: subscription_id, company_name, plan_tier, monthly_revenue, "
                "annual_revenue, start_date, end_date, status, seats_purchased, seats_used, "
                "industry, payment_method, auto_renew, last_payment_date, outstanding_balance, "
                "support_tier, implementation_date, custom_features. "
                "Do NOT request primary_contact — it is blocked PII."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_column": {"type": "string"},
                    "filter_value": {"type": "string"},
                    "filter_operator": {
                        "type": "string",
                        "enum": ["equals", "contains", "greater_than", "less_than", "not_equals"],
                    },
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "sort_by": {"type": "string"},
                    "sort_ascending": {"type": "boolean"},
                    "limit": {"type": "integer"},
                    "aggregation": {"type": "string", "enum": ["sum", "mean", "count", "max", "min"]},
                    "group_by": {"type": "string"},
                    "search_in_list_column": {"type": "string"},
                    "search_term": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get column names, data types, unique categorical values, and sample rows. Call first when unsure what values a column contains.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_metric",
            "description": (
                "Compute derived business metrics not in raw CSV. "
                "seat_utilization: utilization % per customer. "
                "revenue_at_risk: MRR from pending_renewal or auto_renew=false. "
                "churn_mrr: MRR lost to churned customers. "
                "upsell_candidates: active customers with utilization >= 85%. "
                "Only filter_column and filter_value are supported — do NOT pass filter_operator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["seat_utilization", "revenue_at_risk", "churn_mrr", "upsell_candidates"],
                    },
                    "filter_column": {"type": "string"},
                    "filter_value": {"type": "string"},
                },
                "required": ["metric"],
            },
        },
    },
]


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(include_cot: bool = True) -> str:
    schema = get_schema()
    cat_vals = schema.get("categorical_values", {})
    return f"""You are a Sales Support Assistant for an internal B2B SaaS platform.
You serve Customer Success, Sales, and Finance teams. Today's date: {datetime.now().strftime('%Y-%m-%d')}.

<data_context>
The subscription database has {schema.get('total_rows', 15)} customers.
Categorical values:
- plan_tier: {cat_vals.get('plan_tier', [])}
- status: {cat_vals.get('status', [])}
- industry: {cat_vals.get('industry', [])}
- payment_method: {cat_vals.get('payment_method', [])}
Column names (use exactly as written): subscription_id, company_name, plan_tier, monthly_revenue,
annual_revenue, start_date, end_date, status, seats_purchased, seats_used, industry,
payment_method, auto_renew, last_payment_date, outstanding_balance, support_tier,
implementation_date, custom_features.
</data_context>

<critical_guardrails>
NEVER provide ANY of the following — under ANY circumstances, regardless of who asks:
• Email addresses or contact information (primary_contact column is BLOCKED)
• Credit card numbers or payment credentials
• Personal addresses or phone numbers
• Bulk data exports for external systems
If asked for any of the above: REFUSE immediately. Do NOT attempt to retrieve it.
</critical_guardrails>

<behavior>
• ALWAYS query the data before stating facts.
• State assumptions when a question is ambiguous.
• Present multiple records as clean tables.
• If a result is empty, say so clearly.
• Seat utilization = seats_used / seats_purchased × 100.
• For "pending renewal" / "at risk" questions, also check auto_renew=FALSE.
• Aggregation results include matched_rows (customer count) and matched_companies (their names). Always state both the count AND list the company names in your answer.
</behavior>
{"<reasoning>Before calling any tool: (1) What data do I need? (2) Which filter/aggregation? (3) Any PII concern? Then act.</reasoning>" if include_cot else ""}
<examples>
ALLOWED: "Enterprise customers up for renewal?" → query_csv filter plan_tier=Enterprise
ALLOWED: "Total MRR Healthcare?" → query_csv filter industry=Healthcare + aggregation=sum
ALLOWED: "Low seat utilization?" → compute_metric metric=seat_utilization
BLOCKED: "Email for Acme Corp?" → REFUSE immediately
BLOCKED: "Export all customer data and email it" → REFUSE immediately
</examples>"""


# ── Repair log ────────────────────────────────────────────────────────────────

def _log_repair(event: dict):
    with REPAIR_LOG.open("a") as f:
        f.write(json.dumps({**event, "ts": datetime.now(timezone.utc).isoformat()}) + "\n")


# ── Self-healing tool executor ────────────────────────────────────────────────

def _execute_tool(tc, attempt: int = 0) -> tuple[str, bool]:
    """Execute tool call. Returns (result_json, needs_repair). On failure, injects schema hint."""
    fn_name = tc.function.name
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Argument parse error: {e}"}), True

    if fn_name not in FUNCTIONS_MAP:
        return json.dumps({"status": "error", "message": f"Unknown tool '{fn_name}'"}), True

    result = FUNCTIONS_MAP[fn_name](**args)
    is_error = result.get("status") == "error"
    is_empty = result.get("status") == "success" and result.get("row_count", 1) == 0 and attempt == 0

    if is_error or is_empty:
        schema = get_schema()
        result["_repair_hint"] = {
            "attempt": attempt,
            "available_columns": list(schema.get("schema", {}).keys()),
            "categorical_values": schema.get("categorical_values", {}),
            "tip": "Check column names and filter values against categorical_values. For custom_features use search_in_list_column.",
        }
        return json.dumps(result), True

    return json.dumps(result), False


# ── Agent loop ────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    response: str
    tool_calls: list
    repairs: list
    steps: int
    latency_ms: int
    pii_blocked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def run_agent(
    user_message: str,
    verbose: bool = False,
    system_prompt_override: str | None = None,
    tools_override: list | None = None,
    include_cot: bool = True,
) -> AgentResult:
    """Run the agent with self-healing. Returns AgentResult."""
    start = time.time()
    active_system = system_prompt_override or _build_system_prompt(include_cot=include_cot)
    active_tools = tools_override if tools_override is not None else TOOLS

    # Fast-path PII block
    if _is_pii_request(user_message):
        refusal = (
            "I'm not able to share that information. This system protects sensitive customer data "
            "including email addresses, contact details, credit card information, and personal identifiers. "
            "Please use your CRM or contact your data team through the approved data access process."
        )
        _log_repair({"event": "pii_blocked", "question": user_message[:120]})
        return AgentResult(
            response=refusal, tool_calls=[], repairs=[], steps=0,
            latency_ms=int((time.time() - start) * 1000), pii_blocked=True,
        )

    messages: list = [{"role": "user", "content": user_message}]
    all_tool_calls: list = []
    all_repairs: list = []
    repair_attempts: dict = {}  # tc_id → int

    for step in range(MAX_AGENT_STEPS):
        # Prepend system message each call (not stored in history to avoid accumulation)
        full_messages = [{"role": "system", "content": active_system}] + messages
        for _attempt in range(5):
            try:
                response = co.chat(model=MODEL, messages=full_messages, tools=active_tools)
                break
            except Exception as _e:
                if "429" in str(_e) or "TooManyRequests" in type(_e).__name__:
                    _wait = 15 * (_attempt + 1)
                    if verbose:
                        print(f"  [rate-limit] sleeping {_wait}s")
                    time.sleep(_wait)
                else:
                    raise
        if verbose:
            print(f"  [step {step}] finish={response.finish_reason} "
                  f"tools={len(response.message.tool_calls or [])}")

        if not response.message.tool_calls:
            text = "".join(
                b.text for b in (response.message.content or []) if hasattr(b, "text")
            )
            return AgentResult(
                response=text, tool_calls=all_tool_calls, repairs=all_repairs,
                steps=step + 1, latency_ms=int((time.time() - start) * 1000),
            )

        messages.append(response.message)
        tool_results = []
        total_repairs_this_step = 0

        for tc in response.message.tool_calls:
            attempt = repair_attempts.get(tc.id, 0)
            result_json, needs_repair = _execute_tool(tc, attempt=attempt)
            result_dict = json.loads(result_json)

            all_tool_calls.append({
                "tool": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "status": result_dict.get("status"),
                "row_count": result_dict.get("row_count"),
                "error": result_dict.get("message"),
            })

            if needs_repair:
                total_repairs_this_step += 1
                repair_attempts[tc.id] = attempt + 1
                repair_rec = {
                    "event": "repair_triggered",
                    "tool": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "error": result_dict.get("message", "empty_result"),
                    "attempt": attempt,
                }
                all_repairs.append(repair_rec)
                _log_repair(repair_rec)
                if verbose:
                    print(f"  [repair] {tc.function.name} attempt={attempt} "
                          f"err={str(result_dict.get('message','empty'))[:50]}")

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_json,
            })

        messages.extend(tool_results)

        # Inject escalation message after exhausting retries
        if total_repairs_this_step > 0 and sum(repair_attempts.values()) >= MAX_REPAIR_RETRIES:
            messages.append({
                "role": "user",
                "content": (
                    "Tool calls have failed after multiple retries. "
                    "Provide the best partial answer from what was retrieved, "
                    "and clearly explain what could not be obtained."
                ),
            })

    return AgentResult(
        response="Unable to complete request within step limit. Please rephrase.",
        tool_calls=all_tool_calls, repairs=all_repairs,
        steps=MAX_AGENT_STEPS, latency_ms=int((time.time() - start) * 1000),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv
    print("=" * 60)
    print(f"Sales Support Agent  |  Model: {MODEL}")
    print(f"Data: {DATA_PATH}  |  type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = run_agent(user_input, verbose=verbose)
        print(f"\nAgent: {result.response}")
        if verbose:
            print(f"\n[steps={result.steps} | {result.latency_ms}ms | "
                  f"repairs={len(result.repairs)} | pii_blocked={result.pii_blocked}]")

# ⚖️ Adversarial Multi-Agent Deliberation Framework

> A domain-agnostic framework where LLM agents argue, fact-check each other, and self-evaluate before reaching an auditable decision — demonstrated on financial risk analysis, directly applicable to fraud adjudication, credit decisioning, and compliance review.

## The Problem

Modern frontier LLMs can reason, reflect, and argue multiple sides of a question. But they do it inside a black box. The conclusion is visible. The process is not.

In regulated environments like investment committees, credit decisioning, fraud adjudication, clinical review, the process is the requirement rather than what the AI concluded. What matters is how it got there, what evidence it weighed, which assumptions were challenged, and what the dissenting view was.

A single model reasoning alone, however capable, cannot satisfy this. There is no independent challenger. No verified evidence trail. No documented record of what was argued and rebutted. No way to distinguish a well-reasoned conclusion from a confident-sounding hallucination.

This framework makes the reasoning process external, structured, and verifiable — opposing agents with locked roles argue from independently verified evidence, a dedicated node catches and rejects hallucinated claims before they enter the record, and the full debate log serves as the audit trail. Not a smarter LLM. A trustworthy process built around one.

---

## Architecture

```
User Input — Company + Ticker
        |
        v
[ DATA FETCHER ]
  25 fundamentals · 3-year history · peer data · SEC filing · news
        |
        v
[ QUANTAMENTAL RESEARCH ANALYST ]  — No LLM — Pure Python
  Test A: Historical Momentum    — today vs 3-year avg (margins, FCF, revenue)
  Test B: Enhanced Peer Context  — premium/discount vs sector with exact %
  Test C: Live Sentiment         — VADER NLP on 7 news headlines
  Test D: Earnings Surprise      — consecutive beat/miss from earnings history
  Cross-Signal Validation        — if 2 independent tests agree → QUANT_ALERT (5pts)
  Output: Tiered Intelligence Brief with tagged observations
        |
        v
  DEBATE LOOP (N rounds)
  ┌──────────────────────────────────────────────────────┐
  │  [ BULL AGENT ]                                      │
  │    Reads brief · writes scratchpad (CoT) · argues FOR│
  │    Cites tagged observations only                    │
  │         |                                            │
  │  [ FACT-CHECKER ]  — No LLM                         │
  │    Verifies citations map to real brief tags         │
  │    Computes support score (weighted by tier)         │
  │    Rejects + loops back if citations invalid (max 2) │
  │         |                                            │
  │  [ BEAR AGENT ]                                      │
  │    Reads brief · argues AGAINST · rebuts Bull        │
  │         |                                            │
  │  [ FACT-CHECKER ]  — same verification               │
  │         |                                            │
  │  [ CHALLENGER AGENT ]                                │
  │    No fixed side · attacks weakest-cited claim       │
  │    Memory of previous attacks — finds new weakness   │
  │         |                                            │
  │  [ REBUTTAL AGENT ]                                  │
  │    Targeted agent defends against Challenger         │
  │    2-3 sentences · addresses specific attack only    │
  │         |                                            │
  │  round < N? → loop · round = N? → exit              │
  └──────────────────────────────────────────────────────┘
        |
        v
[ CALIBRATION ENGINE ]  — No LLM
  Weighted support score · normalised delta · uncertainty level
        |
        v
[ JUDGE AGENT ]
  Lean prompt — key claims + citations + scores only (no scratchpads)
  Verdict aligned with calibration metrics
        |
        v
  Streamlit UI · LangSmith Observability
```

---

## Evidence Scoring — How Support Score Works

| Citation Type | Points | Example Tag |
|---|---|---|
| Standard observation | 1 pt | `OBS_BULL_FCF`, `OBS_BEAR_PE` |
| Anomaly flag | 3 pts | `EXCEPTIONAL — ROE 152%` |
| Quantamental alert | 5 pts | `QUANT_ALERT_BULL`, `QUANT_ALERT_BEAR` |
| Hallucination / invalid tag | -1 pt | Any tag not in the brief |
| Voided argument (2 failed rewrites) | -50 pts | — |

Support score is deterministic — computed by the Fact-Checker, not the LLM. The side with the higher cumulative score wins the debate.

---

## Quantamental Cross-Signal Validation

A QUANT_ALERT fires only when two independent data sources corroborate the same conclusion:

| Combination | Alert |
|---|---|
| Test A decay + Test C negative sentiment | `QUANT_ALERT_BEAR` |
| Test B overvalued + Test C negative sentiment | `QUANT_ALERT_BEAR` |
| Test A momentum + Test C positive sentiment | `QUANT_ALERT_BULL` |
| Test B discount + Test D earnings beats | `QUANT_ALERT_BULL` |

Alerts are rare by design — VADER threshold ±0.2, momentum threshold ±10%. When they fire they carry 5x the weight of a standard observation.

---

## Sample Output

```
VERDICT:     Invest
CONFIDENCE:  High

CALIBRATION
  Bull Support Score:  21
  Bear Support Score:  8
  Delta:               44.8%  (Low uncertainty)
  Hallucinations:      0

SWING ARGUMENTS
  • QUANT_ALERT_BULL fired — positive sentiment corroborates peer-beating growth
  • ROE of 152% held under Challenger attack across 3 rounds
  • Bear's macro risk argument had no supporting citations beyond fallback tags

RECOMMENDED ACTION
  Initiate position. Monitor debt/equity quarterly.
  Re-evaluate if sector growth drops below 8% or sentiment turns negative.
```

---

## Engineering Challenges Solved

### Hallucination Mitigation
Arguments are not accepted at face value. A dedicated Fact-Checker node
verifies every citation against the intelligence brief before committing
an argument to the debate log. Agents rewrite up to 2 times if citations
are invalid. Arguments with repeated hallucinations are voided and penalised
-50 points in calibration. This is an architectural guardrail, not a prompt.

### Role Integrity Under Adversarial Pressure
LLMs default to balanced, helpful responses — which breaks the adversarial
debate structure. Four techniques enforce role discipline simultaneously:
identity anchoring in the system prompt, explicit prohibition of balancing
language, adversarial awareness injection (each agent must rebut the
opponent's last argument specifically), and JSON output enforcement at the
API level via `response_format`.

### Explainability by Design
Every claim in the final verdict traces back to a specific tagged observation
in the debate log — `[Source: OBS_BULL_FCF]`, `[Source: QUANT_ALERT_BEAR]`.
This is not post-hoc annotation. It is a structural property of the system.
The debate log is the audit trail.

### Verifiable Evaluation — No LLM Opinion
Support score is computed deterministically by the Fact-Checker, not the LLM.
Standard observation = 1 point. Anomaly flag = 3 points. Quantamental alert
= 5 points. Invalid citation = -1 point. The side with the higher cumulative
score wins the debate. Every number is reproducible and explainable.

### Quantamental Signal Validation
Four independent data sources — historical momentum (Test A), peer comparison
(Test B), live NLP sentiment via VADER (Test C), and earnings surprise momentum
(Test D) — are computed without LLM involvement. A cross-signal alert
`QUANT_ALERT` fires only when two independent sources corroborate the same
conclusion, preventing noise from triggering false signals.

### Context Management
The Judge receives lean argument summaries — key claims, verified citations,
and support scores only. Scratchpads and full argument text are stripped before
the Judge prompt is built. This prevents context overflow on long debates while
preserving all information needed for a reasoned verdict.

### Production Observability
LangSmith traces every node automatically — per-node latency, token usage,
exact prompts sent, and raw outputs. The full execution graph is visible in
the LangSmith dashboard for every run, making the system debuggable and
auditable at the infrastructure level.

## LangSmith Observability

Every node traced automatically — per-node latency, token usage, exact prompts, raw outputs.

---

## Tech Stack

| Tool | Purpose | Cost |
|---|---|---|
| LangGraph | Graph orchestration — nodes, edges, state | Free |
| Groq (Llama 3.3 70B) | LLM inference — ~500 tokens/sec | Free tier |
| VADER Sentiment | NLP sentiment on news headlines | Free |
| yfinance | Fundamentals + 3-year history + earnings | Free |
| SEC EDGAR API | Latest 10-K filing | Free |
| Tavily | Financial news search | Free tier |
| LangSmith | Execution tracing | Free tier |
| Streamlit + Plotly | UI + support score chart | Free |

---

## Why This Pattern Matters Beyond Finance

The investment domain is the demonstration environment. The adversarial deliberation pattern applies wherever structured challenge is required before a decision:

| Domain | Bull equivalent | Bear equivalent | Output |
|---|---|---|---|
| Fintech / Payments | Approve signal | Fraud signal | Risk decision memo |
| Healthcare | Diagnosis A | Diagnosis B | Differential report |
| Legal | Prosecution | Defense | Risk assessment |
| Engineering | Architecture advocate | Security skeptic | ADR document |

---

## Author

**Lakshmi Sahithi Budamagunta** — AI Engineer

[LinkedIn](https://www.linkedin.com/in/blsahithi/)

# ⚖️ Adversarial Multi-Agent Deliberation Framework

> A domain-agnostic framework where LLM agents argue, fact-check each other, and self-evaluate before reaching an auditable decision — demonstrated on financial risk analysis, directly applicable to fraud adjudication, credit decisioning, and compliance review.

## The Problem

LLMs give confident answers with no internal challenge mechanism. A single model reasoning alone anchors early, never stress-tests its own assumptions, and produces outputs with no reasoning trail. In high-stakes domains — finance, medicine, law — a single perspective is institutionally unacceptable.

This framework models what real decision committees do: opposing cases are formally argued, challenged, and rebutted before a decision is made. The debate log is the audit trail.

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

import os
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from agents import (
    fetch_data,
    run_research_analyst,
    run_bull,
    run_bear,
    run_fact_checker,
    run_challenger,
    run_rebuttal,       # <-- IMPORTED NEW REBUTTAL NODE
    run_calibration,
    run_judge,
)

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
# DEBATE STATE
# ══════════════════════════════════════════════════════════════════════════

class DebateState(TypedDict):

    # ── Input ──────────────────────────────────────────────────────────────
    company:                    str
    ticker:                     str
    max_rounds:                 int

    # ── Control ────────────────────────────────────────────────────────────
    round_number:               int

    # ── Data layer ─────────────────────────────────────────────────────────
    raw_data:                   dict
    intelligence_brief:         dict

    # ── Pending (written by Bull/Bear, cleared by Fact-Checker) ───────────
    pending_bull_argument:      dict
    pending_bear_argument:      dict
    bull_rewrite_count:         int
    bear_rewrite_count:         int

    # ── Committed debate log (append-only via operator.add) ───────────────
    bull_arguments:             Annotated[list, operator.add]
    bear_arguments:             Annotated[list, operator.add]
    challenger_arguments:       Annotated[list, operator.add]
    targeted_rebuttals:         Annotated[list, operator.add] # <-- ADDED REBUTTAL STATE

    # ── Output ─────────────────────────────────────────────────────────────
    calibration_metrics:        dict
    decision_memo:              dict


# ══════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════════════════════

def should_continue(state: DebateState) -> str:
    if state["round_number"] < state["max_rounds"]:
        print(f"\n[Graph] Round {state['round_number']} complete — continuing debate...")
        return "run_bull"
    print(f"\n[Graph] All {state['max_rounds']} rounds complete — moving to calibration...")
    return "run_calibration"


def bull_fact_check_result(state: DebateState) -> str:
    pending = state.get("pending_bull_argument", {})
    status  = pending.get("fact_check_status", "passed")
    count   = state.get("bull_rewrite_count", 0)
    if status == "failed" and count < 2:
        print(f"  [Graph] Bull failed fact-check — rewrite {count}/2...")
        return "run_bull"
    return "run_bear"


def bear_fact_check_result(state: DebateState) -> str:
    pending = state.get("pending_bear_argument", {})
    status  = pending.get("fact_check_status", "passed")
    count   = state.get("bear_rewrite_count", 0)
    if status == "failed" and count < 2:
        print(f"  [Graph] Bear failed fact-check — rewrite {count}/2...")
        return "run_bear"
    return "run_challenger"


# ══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(DebateState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    g.add_node("fetch_data",           fetch_data)
    g.add_node("run_research_analyst", run_research_analyst)
    g.add_node("run_bull",             run_bull)
    g.add_node("fact_check_bull",      lambda s: run_fact_checker(s, agent="bull"))
    g.add_node("run_bear",             run_bear)
    g.add_node("fact_check_bear",      lambda s: run_fact_checker(s, agent="bear"))
    g.add_node("run_challenger",       run_challenger)
    g.add_node("run_rebuttal",         run_rebuttal)  # <-- ADDED REBUTTAL NODE
    g.add_node("run_calibration",      run_calibration)
    g.add_node("run_judge",            run_judge)

    # ── Entry ──────────────────────────────────────────────────────────────
    g.set_entry_point("fetch_data")

    # ── Direct edges ───────────────────────────────────────────────────────
    g.add_edge("fetch_data",           "run_research_analyst")
    g.add_edge("run_research_analyst", "run_bull")
    g.add_edge("run_bull",             "fact_check_bull")
    g.add_edge("run_bear",             "fact_check_bear")
    g.add_edge("run_challenger",       "run_rebuttal") # <-- NEW: Challenger flows straight to Rebuttal
    g.add_edge("run_calibration",      "run_judge")
    g.add_edge("run_judge",            END)

    # ── Conditional edges ──────────────────────────────────────────────────
    g.add_conditional_edges(
        "fact_check_bull",
        bull_fact_check_result,
        {"run_bull": "run_bull", "run_bear": "run_bear"},
    )
    g.add_conditional_edges(
        "fact_check_bear",
        bear_fact_check_result,
        {"run_bear": "run_bear", "run_challenger": "run_challenger"},
    )
    
    # <-- NEW: Rebuttal now decides whether to loop back to Bull or go to Calibration
    g.add_conditional_edges(
        "run_rebuttal",
        should_continue,
        {"run_bull": "run_bull", "run_calibration": "run_calibration"},
    )

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════

graph = build_graph()


def make_initial_state(company: str, ticker: str, max_rounds: int = 3) -> dict:
    return {
        "company":               company.strip(),
        "ticker":                ticker.strip().upper(),
        "round_number":          0,
        "max_rounds":            int(max_rounds),
        "raw_data":              {},
        "intelligence_brief":    {},
        "pending_bull_argument": {},
        "pending_bear_argument": {},
        "bull_rewrite_count":    0,
        "bear_rewrite_count":    0,
        "bull_arguments":        [],
        "bear_arguments":        [],
        "challenger_arguments":  [],
        "targeted_rebuttals":    [],  # <-- ADDED REBUTTAL LIST
        "calibration_metrics":   {},
        "decision_memo":         {},
    }
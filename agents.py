# agents.py
import os
import re
import json
import requests
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from prompts import (
    bull_system_prompt,
    bear_system_prompt,
    challenger_system_prompt,
    judge_system_prompt,
    bull_human_prompt,
    bear_human_prompt,
    challenger_human_prompt,
    judge_human_prompt,
    rebuttal_system_prompt,
    rebuttal_human_prompt
)

load_dotenv()

# ── LLM + Clients ──────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"response_format": {"type": "json_object"}}
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
vader_analyzer = SentimentIntensityAnalyzer()


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def parse_llm_json(response_text: str) -> dict:
    try:
        cleaned = response_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"  [parse_llm_json] WARNING: {e}")
        return {"argument": "Parse failed.", "citations": [], "key_claims": [], "rebuttal": "", "scratchpad": ""}


def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def format_debate_log_for_judge(args_list: list) -> str:
    if not args_list:
        return "No arguments presented."
    lines = []
    for a in args_list:
        if a.get("fact_check_status") == "voided":
            lines.append(f"[Round {a.get('round')}] ARGUMENT VOIDED — repeated hallucination.")
            continue
        lines.append(
            f"[Round {a.get('round')}]\n"
            f"  Key Claims: {a.get('key_claims', [])}\n"
            f"  Citations:  {a.get('verified_citations', [])}\n"
            f"  Support Score: {a.get('support_score', 0)}\n"
            f"  Rebuttal: {a.get('rebuttal', '')}"
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# NODE 1 — DATA FETCHER 
# ══════════════════════════════════════════════════════════════════════════

def fetch_data(state: dict) -> dict:
    company = state["company"]
    ticker  = state["ticker"]
    print(f"\n[Data Fetcher] Fetching data for {company} ({ticker})...")

    stock = yf.Ticker(ticker)
    info  = stock.info or {}

    fundamentals = {
        "pe_ratio":          info.get("trailingPE",        "N/A"),
        "forward_pe":        info.get("forwardPE",         "N/A"),
        "revenue_growth":    info.get("revenueGrowth",     "N/A"),
        "profit_margins":    info.get("profitMargins",     "N/A"),
        "gross_margins":     info.get("grossMargins",      "N/A"),
        "operating_margins": info.get("operatingMargins",  "N/A"),
        "debt_to_equity":    info.get("debtToEquity",      "N/A"),
        "free_cashflow":     info.get("freeCashflow",      "N/A"),
        "market_cap":        info.get("marketCap",         "N/A"),
        "52w_high":          info.get("fiftyTwoWeekHigh",  "N/A"),
        "52w_low":           info.get("fiftyTwoWeekLow",   "N/A"),
        "current_price":     info.get("currentPrice",      "N/A"),
        "analyst_target":    info.get("targetMeanPrice",   "N/A"),
        "recommendation":    info.get("recommendationKey", "N/A"),
        "earnings_growth":   info.get("earningsGrowth",    "N/A"),
        "return_on_equity":  info.get("returnOnEquity",    "N/A"),
        "dividend_yield":    info.get("dividendYield",     "N/A"),
        "peg_ratio":         info.get("pegRatio",          "N/A"),
        "price_to_book":     info.get("priceToBook",       "N/A"),
        "quick_ratio":       info.get("quickRatio",        "N/A"),
        "current_ratio":     info.get("currentRatio",      "N/A"),
        "total_debt":        info.get("totalDebt",         "N/A"),
        "total_cash":        info.get("totalCash",         "N/A"),
        "revenue":           info.get("totalRevenue",      "N/A"),
        "net_income":        info.get("netIncomeToCommon", "N/A"),
    }

    fundamentals_history = {}
    try:
        financials = stock.financials
        balance    = stock.balance_sheet
        cashflow   = stock.cashflow

        if financials is not None and not financials.empty:
            years = [str(c.year) for c in financials.columns[:3]]
            for label, key in [("revenue_3yr", "Total Revenue"), ("net_income_3yr", "Net Income")]:
                if key in financials.index:
                    fundamentals_history[label] = {yr: int(financials.loc[key, col]) for yr, col in zip(years, financials.columns[:3])}

        if cashflow is not None and not cashflow.empty:
            years = [str(c.year) for c in cashflow.columns[:3]]
            if "Free Cash Flow" in cashflow.index:
                fundamentals_history["free_cashflow_3yr"] = {yr: int(cashflow.loc["Free Cash Flow", col]) for yr, col in zip(years, cashflow.columns[:3])}
    except Exception as e:
        print(f"  [Data Fetcher] History partial: {e}")

    earnings_surprises = []
    try:
        edates = stock.earnings_dates
        if edates is not None and not edates.empty and 'Surprise(%)' in edates.columns:
            earnings_surprises = edates['Surprise(%)'].dropna().head(4).tolist()
    except Exception:
        pass

    peer_data   = fetch_peer_data(ticker, info.get("sector", ""), info.get("industry", ""))
    sec_summary = fetch_sec_filing(ticker, info)

    try:
        news_results = tavily.search(query=f"{company} stock earnings revenue risks outlook 2025", max_results=7)
        news = [{"title": r["title"], "snippet": r["content"][:250]} for r in news_results.get("results", [])]
    except Exception:
        news = []

    print(f"[Data Fetcher] ✓ Fundamentals: {len(fundamentals)} metrics")
    print(f"[Data Fetcher] ✓ History: {len(fundamentals_history)} trend series")
    print(f"[Data Fetcher] ✓ Peers: {len(peer_data)} companies")
    print(f"[Data Fetcher] ✓ News: {len(news)} articles")

    return {
        "raw_data": {
            "fundamentals":          fundamentals,
            "fundamentals_history":  fundamentals_history,
            "peer_data":             peer_data,
            "sec_summary":           sec_summary,
            "news":                  news,
            "earnings_surprises":    earnings_surprises,
        }
    }


def fetch_peer_data(ticker: str, sector: str, industry: str = "") -> list:
    peer_map = {
        "Technology":                ["MSFT", "GOOGL", "META", "AAPL", "NVDA"],
        "Financial Services":        ["JPM", "BAC", "GS", "MS", "WFC"],
        "Consumer Cyclical":         ["AMZN", "TSLA", "NKE", "HD", "MCD"],
        "Consumer Defensive":        ["PG", "KO", "PEP", "WMT", "COST"],
        "Healthcare":                ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
        "Energy":                    ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Industrials":               ["CAT", "HON", "GE", "MMM", "UPS"],
        "Communication Services":    ["GOOGL", "META", "NFLX", "DIS", "T"],
        "Real Estate":               ["AMT", "PLD", "CCI", "EQIX", "SPG"],
        "Utilities":                 ["NEE", "DUK", "SO", "D", "AEP"],
        "Basic Materials":           ["LIN", "APD", "ECL", "DD", "NEM"],
        "Semiconductor":             ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],
        "Automotive":                ["TSLA", "F", "GM", "TM", "STLA"],
        "Retail":                    ["WMT", "AMZN", "COST", "TGT", "HD"],
        "Banking":                   ["JPM", "BAC", "WFC", "C", "USB"],
        "Insurance":                 ["BRK-B", "AIG", "MET", "PRU", "AFL"],
        "Pharma":                    ["PFE", "MRK", "ABBV", "BMY", "LLY"],
        "Software":                  ["MSFT", "ORCL", "SAP", "CRM", "ADBE"],
        "Media":                     ["DIS", "NFLX", "PARA", "WBD", "FOX"],
        "Aerospace":                 ["BA", "LMT", "RTX", "NOC", "GD"],
    }

    candidates = [p for p in peer_map.get(sector, []) if p != ticker]

    if not candidates:
        for key, tickers in peer_map.items():
            if key.lower() in industry.lower() or industry.lower() in key.lower():
                candidates = [p for p in tickers if p != ticker]
                break

    peer_data = []
    for peer in candidates[:2]:
        try:
            p_info = yf.Ticker(peer).info or {}
            peer_data.append({
                "ticker":         peer,
                "pe_ratio":       p_info.get("trailingPE",    "N/A"),
                "revenue_growth": p_info.get("revenueGrowth", "N/A"),
                "profit_margins": p_info.get("profitMargins", "N/A"),
                "market_cap":     p_info.get("marketCap",     "N/A"),
            })
        except Exception:
            pass

    return peer_data


def fetch_sec_filing(ticker: str, yf_info: dict) -> str:
    try:
        headers = {"User-Agent": "AIPortfolio contact@email.com"}
        url     = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
        r       = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200 and r.json().get("hits", {}).get("hits", []):
            filing = r.json()["hits"]["hits"][0]["_source"]
            return (
                f"SEC 10-K | Filed: {filing.get('file_date', 'N/A')} | "
                f"Company: {filing.get('display_names', ['N/A'])[0]}"
            )
    except Exception:
        pass
    return yf_info.get("longBusinessSummary", "Company description unavailable.")[:400]


# ══════════════════════════════════════════════════════════════════════════
# NODE 2 — QUANTAMENTAL RESEARCH ANALYST 
# ══════════════════════════════════════════════════════════════════════════

def run_research_analyst(state: dict) -> dict:
    print(f"\n[Research Analyst] Building intelligence brief (Quantamental V2)...")

    raw     = state.get("raw_data") or {}
    f       = raw.get("fundamentals") or {}
    history = raw.get("fundamentals_history") or {}
    peers   = raw.get("peer_data") or []
    news    = raw.get("news") or []
    surprises = raw.get("earnings_surprises") or []

    sentiment_score = 0
    if news:
        text_corpus = " ".join([n.get("title", "") for n in news])
        sentiment_score = vader_analyzer.polarity_scores(text_corpus)['compound']

    earnings_status = 0
    if len(surprises) >= 2:
        if all(s > 0.02 for s in surprises[:2]): earnings_status = 1
        elif all(s < -0.02 for s in surprises[:2]): earnings_status = -1

    financial_health  = build_financial_health(f, history)
    peer_context      = build_peer_context(f, peers)
    bull_observations = build_bull_observations(f, history, peer_context, sentiment_score, earnings_status)
    bear_observations = build_bear_observations(f, history, peer_context, sentiment_score, earnings_status)
    
    key_risks, key_strengths = detect_anomalies(f, history, peer_context, sentiment_score, bull_observations, bear_observations)

    intelligence_brief = {
        "financial_health":   financial_health,
        "peer_context":       peer_context,
        "bull_observations":  bull_observations,
        "bear_observations":  bear_observations,
        "key_risks":          key_risks,
        "key_strengths":      key_strengths,
        "news_summary":       [n.get("title", "") for n in news[:5]],
    }

    print(f"[Research Analyst] ✓ Bull signals: {len(bull_observations)}")
    print(f"[Research Analyst] ✓ Bear signals: {len(bear_observations)}")
    print(f"[Research Analyst] ✓ Key risks: {len(key_risks)} | Strengths: {len(key_strengths)}")

    return {"intelligence_brief": intelligence_brief}


def build_financial_health(f: dict, history: dict) -> dict:
    pe      = safe_float(f.get("pe_ratio"))
    fwd_pe  = safe_float(f.get("forward_pe"))
    growth  = safe_float(f.get("revenue_growth"))
    margins = safe_float(f.get("profit_margins"))
    dte     = safe_float(f.get("debt_to_equity"))
    roe     = safe_float(f.get("return_on_equity"))
    peg     = safe_float(f.get("peg_ratio"))
    fcf     = safe_float(f.get("free_cashflow"))

    rev_3yr = history.get("revenue_3yr") or {}
    if len(rev_3yr) >= 2:
        vals      = list(rev_3yr.values())
        direction = "accelerating" if vals[0] > vals[1] else "decelerating"
        rev_str   = f"{direction} — YoY Context"
    else:
        rev_str = f"current YoY growth {growth:.1%}" if growth else "N/A"

    if dte:
        if dte > 150:   bs = f"highly leveraged — debt/equity {dte:.0f}"
        elif dte > 80:  bs = f"moderately leveraged — debt/equity {dte:.0f}"
        else:           bs = f"conservative leverage — debt/equity {dte:.0f}"
    else:
        bs = "N/A"

    if pe and peg:
        if peg > 2:     val = f"expensive — PE {pe:.1f}, PEG {peg:.2f}"
        elif peg > 1:   val = f"fair — PE {pe:.1f}, PEG {peg:.2f}"
        else:           val = f"potentially undervalued — PE {pe:.1f}, PEG {peg:.2f}"
    elif pe:
        val = f"PE {pe:.1f} — forward PE {fwd_pe:.1f}" if fwd_pe else f"PE {pe:.1f}"
    else:
        val = "N/A"

    return {
        "revenue_trend":    rev_str,
        "margin_trend":     f"{margins:.1%} profit margins" if margins else "N/A",
        "balance_sheet":    bs,
        "valuation_signal": val,
        "peg_ratio":        peg,
        "roe":              f"{roe:.1%}" if roe else "N/A",
        "fcf":              f"${fcf/1e9:.1f}B" if fcf else "N/A",
    }


def build_peer_context(f: dict, peers: list) -> dict:
    if not peers:
        return {"note": "No peer data available for this sector"}

    company_pe     = safe_float(f.get("pe_ratio"))
    company_growth = safe_float(f.get("revenue_growth"))

    peer_pes     = [x for p in peers if (x := safe_float(p.get("pe_ratio")))]
    peer_growths = [x for p in peers if (x := safe_float(p.get("revenue_growth")))]
    peer_margins = [x for p in peers if (x := safe_float(p.get("profit_margins")))]

    sector_pe     = round(sum(peer_pes) / len(peer_pes), 2) if peer_pes else None
    sector_growth = round(sum(peer_growths) / len(peer_growths), 4) if peer_growths else None
    sector_margin = round(sum(peer_margins) / len(peer_margins), 4) if peer_margins else None

    pe_vs     = ""
    growth_vs = ""

    if company_pe and sector_pe:
        ratio = company_pe / sector_pe
        if ratio > 1.2:   pe_vs = f"premium — {ratio:.1f}x sector avg PE of {sector_pe}"
        elif ratio < 0.8: pe_vs = f"discount — {ratio:.1f}x sector avg PE of {sector_pe}"
        else:             pe_vs = f"in-line — {ratio:.1f}x sector avg PE of {sector_pe}"

    if company_growth and sector_growth:
        growth_vs = (
            f"above sector — {company_growth:.1%} vs {sector_growth:.1%}"
            if company_growth > sector_growth
            else f"below sector — {company_growth:.1%} vs {sector_growth:.1%}"
        )

    return {
        "sector_pe_avg":     sector_pe,
        "sector_growth_avg": sector_growth,
        "sector_margin_avg": sector_margin,
        "pe_vs_sector":      pe_vs,
        "growth_vs_sector":  growth_vs,
        "peers_used":        [p.get("ticker", "UNK") for p in peers],
    }


def build_bull_observations(f: dict, history: dict, peer_context: dict, sentiment_score: float, earnings_status: int) -> list:
    obs = []
    
    fcf_history = list((history.get("free_cashflow_3yr") or {}).values())
    if len(fcf_history) >= 2 and fcf_history[1] > 0:
        change = (fcf_history[0] - fcf_history[1]) / fcf_history[1]
        if change > 0.10:
            obs.append({"tag": "OBS_BULL_MOMENTUM_FCF", "text": f"Free Cash Flow has accelerated by {change:.1%} vs historical baseline.", "source": "TIME_SERIES"})
            
    rev_history = list((history.get("revenue_3yr") or {}).values())
    if len(rev_history) >= 2 and rev_history[1] > 0:
        change = (rev_history[0] - rev_history[1]) / rev_history[1]
        if change > 0.15:
            obs.append({"tag": "OBS_BULL_MOMENTUM_REV", "text": f"Revenue growth has accelerated by {change:.1%} vs 3-year historical baseline.", "source": "TIME_SERIES"})
    
    if peer_context.get("pe_vs_sector", "").startswith("discount"):
        obs.append({"tag": "OBS_BULL_DISCOUNT", "text": f"Valuation {peer_context['pe_vs_sector']}", "source": "PEER_COMPARISON"})
    if peer_context.get("growth_vs_sector", "").startswith("above"):
        obs.append({"tag": "OBS_BULL_PEER_GROWTH", "text": f"Growth {peer_context['growth_vs_sector']}", "source": "PEER_COMPARISON"})

    if sentiment_score > 0.2:
        obs.append({"tag": "OBS_BULL_SENTIMENT", "text": f"Live news sentiment is positive (Score: {sentiment_score:.2f}). Market catalysts are bullish.", "source": "NLP_VADER"})

    if earnings_status == 1:
        obs.append({"tag": "OBS_BULL_EARNINGS_BEAT", "text": "Consistent execution: Beat earnings estimates for consecutive quarters.", "source": "EARNINGS_DATES"})

    fcf = safe_float(f.get("free_cashflow"))
    roe = safe_float(f.get("return_on_equity"))
    margins = safe_float(f.get("profit_margins"))
    target = safe_float(f.get("analyst_target"))
    price = safe_float(f.get("current_price"))
    
    if fcf and fcf > 10e9:
        obs.append({"tag": "OBS_BULL_FCF", "text": f"Free cashflow ${fcf/1e9:.1f}B — strong capital allocation capacity", "source": "FREE_CASHFLOW"})
    if roe and roe > 0.15:
        obs.append({"tag": "OBS_BULL_ROE", "text": f"ROE of {roe:.1%} — highly efficient capital use", "source": "RETURN_ON_EQUITY"})
    if margins and margins > 0.15:
        obs.append({"tag": "OBS_BULL_MARGINS","text": f"Profit margins {margins:.1%} — above typical industry levels", "source": "PROFIT_MARGINS"})
    if target and price and target > price * 1.1:
        upside = (target - price) / price
        obs.append({"tag": "OBS_BULL_TARGET", "text": f"Analyst target ${target:.2f} implies {upside:.1%} upside from ${price:.2f}", "source": "ANALYST_TARGET"})

    if not obs:
        obs.append({"tag": "OBS_BULL_OPPORTUNITY", "text": "Established market presence provides a safe-haven asset floor during volatility.", "source": "FALLBACK"})

    return obs


def build_bear_observations(f: dict, history: dict, peer_context: dict, sentiment_score: float, earnings_status: int) -> list:
    obs = []

    fcf_history = list((history.get("free_cashflow_3yr") or {}).values())
    if len(fcf_history) >= 2 and fcf_history[1] > 0:
        change = (fcf_history[0] - fcf_history[1]) / fcf_history[1]
        if change < -0.10:
            obs.append({"tag": "OBS_BEAR_DECAY_FCF", "text": f"Free Cash Flow has decayed by {abs(change):.1%} vs historical baseline.", "source": "TIME_SERIES"})

    net_history = list((history.get("net_income_3yr") or {}).values())
    if len(net_history) >= 2 and net_history[1] > 0:
        change = (net_history[0] - net_history[1]) / net_history[1]
        if change < -0.15:
            obs.append({"tag": "OBS_BEAR_DECAY_PROFIT", "text": f"Net Income has severely decayed by {abs(change):.1%} vs 3-year historical baseline.", "source": "TIME_SERIES"})

    if peer_context.get("pe_vs_sector", "").startswith("premium"):
        obs.append({"tag": "OBS_BEAR_OVERVALUED", "text": f"Valuation {peer_context['pe_vs_sector']}", "source": "PEER_COMPARISON"})
    if peer_context.get("growth_vs_sector", "").startswith("below"):
        obs.append({"tag": "OBS_BEAR_PEER_GROWTH", "text": f"Growth {peer_context['growth_vs_sector']}", "source": "PEER_COMPARISON"})

    if sentiment_score < -0.2:
        obs.append({"tag": "OBS_BEAR_SENTIMENT", "text": f"Live news sentiment is negative (Score: {sentiment_score:.2f}). Market catalysts are bearish.", "source": "NLP_VADER"})

    if earnings_status == -1:
        obs.append({"tag": "OBS_BEAR_EARNINGS_MISS", "text": "Execution breakdown: Missed earnings estimates for consecutive quarters.", "source": "EARNINGS_DATES"})

    pe    = safe_float(f.get("pe_ratio"))
    peg   = safe_float(f.get("peg_ratio"))
    dte   = safe_float(f.get("debt_to_equity"))
    price = safe_float(f.get("current_price"))
    hi52  = safe_float(f.get("52w_high"))
    lo52  = safe_float(f.get("52w_low"))
    fwd_pe= safe_float(f.get("forward_pe"))

    if pe and pe > 25:
        obs.append({"tag": "OBS_BEAR_PE", "text": f"Trailing PE {pe:.1f} — limited margin of safety at current valuation", "source": "PE_RATIO"})
    if peg and peg > 2:
        obs.append({"tag": "OBS_BEAR_PEG", "text": f"PEG {peg:.2f} — expensive relative to growth rate", "source": "PEG_RATIO"})
    if dte and dte > 100:
        obs.append({"tag": "OBS_BEAR_DEBT", "text": f"Debt/equity {dte:.0f} — significant financial leverage risk", "source": "DEBT_TO_EQUITY"})
    if price and hi52 and price > hi52 * 0.92:
        obs.append({"tag": "OBS_BEAR_52W", "text": f"Trading ${price:.2f} near 52-week high ${hi52:.2f} — limited near-term upside", "source": "52W_HIGH"})

    if len(obs) < 3:
        if pe and "OBS_BEAR_VALUATION" not in [o["tag"] for o in obs]:
            obs.append({"tag": "OBS_BEAR_VALUATION", "text": f"Current PE of {pe:.1f} — any earnings miss will compress valuation sharply", "source": "PE_RATIO"})
        if price and hi52 and lo52 and "OBS_BEAR_RANGE" not in [o["tag"] for o in obs]:
            range_pct = (price - lo52) / max(hi52 - lo52, 0.01)
            obs.append({"tag": "OBS_BEAR_RANGE", "text": f"Trading at {range_pct:.0%} of 52-week range — momentum-driven buyers face elevated exit risk", "source": "52W_HIGH"})
        if "OBS_BEAR_MACRO" not in [o["tag"] for o in obs]:
            obs.append({"tag": "OBS_BEAR_MACRO", "text": "Regardless of micro-health, sector-wide macroeconomic pullbacks pose a systemic risk.", "source": "FALLBACK"})

    return obs


def detect_anomalies(f: dict, history: dict, peer_context: dict, sentiment_score: float, bull_obs: list, bear_obs: list) -> tuple:
    risks     = []
    strengths = []

    if sentiment_score < -0.2 and (any("DECAY" in o["tag"] for o in bear_obs) or any("MISS" in o["tag"] for o in bear_obs) or any("OVERVALUED" in o["tag"] for o in bear_obs)):
        risks.append("[QUANT_ALERT_BEAR] Critical breakdown: Negative market sentiment directly corroborates deteriorating financials/valuation.")
    
    if sentiment_score > 0.2 and (
        any("MOMENTUM" in o["tag"] for o in bull_obs) or 
        any("BEAT" in o["tag"] for o in bull_obs) or 
        any("DISCOUNT" in o["tag"] for o in bull_obs) or
        any("PEER_GROWTH" in o["tag"] for o in bull_obs) or
        any("ROE" in o["tag"] for o in bull_obs)
    ):
        strengths.append("[QUANT_ALERT_BULL] Powerful catalyst convergence: Positive market sentiment is directly backed by underlying financial momentum/value.")

    roe = safe_float(f.get("return_on_equity"))
    dte = safe_float(f.get("debt_to_equity"))
    fcf = safe_float(f.get("free_cashflow"))
    pe  = safe_float(f.get("pe_ratio"))
    peg = safe_float(f.get("peg_ratio"))

    if dte and dte > 200:
        risks.append(f"CRITICAL — debt/equity {dte:.0f} — extreme leverage, solvency risk")
    if roe and roe > 1.0:
        strengths.append(f"EXCEPTIONAL — ROE {roe:.1%} — unusually high, strong competitive moat")
    if fcf and fcf < 0:
        risks.append(f"WARNING — negative FCF ${fcf/1e9:.1f}B — cash burn detected")
    if pe and peg and pe > 30 and peg > 2.5:
        risks.append(f"VALUATION FLAG — PE {pe:.1f} with PEG {peg:.2f} — expensive on absolute and growth basis")

    return risks, strengths


# ══════════════════════════════════════════════════════════════════════════
# NODE 3 — BULL AGENT
# ══════════════════════════════════════════════════════════════════════════

def run_bull(state: dict) -> dict:
    round_num    = state["round_number"] + 1
    rewrite_count = state.get("bull_rewrite_count", 0)
    rewrite_label = f" (rewrite {rewrite_count})" if rewrite_count > 0 else ""
    print(f"\n[Bull Agent] Round {round_num}{rewrite_label}...")

    brief           = state.get("intelligence_brief", {})
    bear_last       = state["bear_arguments"][-1]["argument"] if state.get("bear_arguments") else None
    challenger_last = state["challenger_arguments"][-1] if state.get("challenger_arguments") else None

    rewrite_note = ""
    if rewrite_count > 0:
        violated = state.get("pending_bull_argument", {}).get("violated_citations", [])
        rewrite_note = f"Previous attempt rejected — unverified citations: {violated}. Only cite tags from the brief."

    past_cites = set()
    for arg in state.get("bull_arguments", []):
        if isinstance(arg, dict):
            past_cites.update(arg.get("citations", []))

    response = llm.invoke([
        SystemMessage(content=bull_system_prompt()),
        HumanMessage(content=bull_human_prompt(
            company            = state.get("company", "UNK"),
            brief              = brief,
            round_num          = round_num,
            bear_last          = bear_last,
            challenger_last    = challenger_last,
            rewrite_note       = rewrite_note,
            previous_citations = list(past_cites) if past_cites else None
        )),
    ])

    parsed = parse_llm_json(response.content)
    print(f"[Bull Agent] ✓ Citations: {parsed.get('citations', [])}")

    return {
        "pending_bull_argument": {
            "round":      round_num,
            "scratchpad": parsed.get("scratchpad", ""),
            "argument":   parsed.get("argument", ""),
            "citations":  parsed.get("citations", []),
            "key_claims": parsed.get("key_claims", []),
            "rebuttal":   parsed.get("rebuttal", ""),
        }
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 4 — BEAR AGENT
# ══════════════════════════════════════════════════════════════════════════

def run_bear(state: dict) -> dict:
    round_num     = state["round_number"] + 1
    rewrite_count = state.get("bear_rewrite_count", 0)
    rewrite_label = f" (rewrite {rewrite_count})" if rewrite_count > 0 else ""
    print(f"\n[Bear Agent] Round {round_num}{rewrite_label}...")

    brief           = state.get("intelligence_brief", {})
    bull_last       = state["bull_arguments"][-1]["argument"] if state.get("bull_arguments") else None
    challenger_last = state["challenger_arguments"][-1] if state.get("challenger_arguments") else None

    rewrite_note = ""
    if rewrite_count > 0:
        violated = state.get("pending_bear_argument", {}).get("violated_citations", [])
        rewrite_note = f"Previous attempt rejected — unverified citations: {violated}. Only cite tags from the brief."

    past_cites = set()
    for arg in state.get("bear_arguments", []):
        if isinstance(arg, dict):
            past_cites.update(arg.get("citations", []))

    response = llm.invoke([
        SystemMessage(content=bear_system_prompt()),
        HumanMessage(content=bear_human_prompt(
            company            = state.get("company", "UNK"),
            brief              = brief,
            round_num          = round_num,
            bull_last          = bull_last,
            challenger_last    = challenger_last,
            rewrite_note       = rewrite_note,
            previous_citations = list(past_cites) if past_cites else None
        )),
    ])

    parsed = parse_llm_json(response.content)
    print(f"[Bear Agent] ✓ Citations: {parsed.get('citations', [])}")

    return {
        "pending_bear_argument": {
            "round":      round_num,
            "scratchpad": parsed.get("scratchpad", ""),
            "argument":   parsed.get("argument", ""),
            "citations":  parsed.get("citations", []),
            "key_claims": parsed.get("key_claims", []),
            "rebuttal":   parsed.get("rebuttal", ""),
        }
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 5 — FACT-CHECKER (No LLM)
# ══════════════════════════════════════════════════════════════════════════

def run_fact_checker(state: dict, agent: str) -> dict:
    print(f"\n[Fact-Checker] Checking {agent} argument...")

    pending_key = f"pending_{agent}_argument"
    rewrite_key = f"{agent}_rewrite_count"
    pending     = state.get(pending_key, {})
    arg_key     = f"{agent}_arguments"

    if not pending:
        return {}

    if state.get(rewrite_key, 0) >= 2:
        print(f"  [Fact-Checker] ❌ {agent.upper()} EXHAUSTED RETRIES — voiding argument.")
        return {
            pending_key: {},
            rewrite_key: 0,
            arg_key: [{
                "round":             pending.get("round"),
                "argument":          "Argument voided — repeated hallucination.",
                "citations":         [],
                "verified_citations":[],
                "support_score":     -50,
                "fact_check_status": "voided",
                "scratchpad":        "",
                "key_claims":        [],
                "rebuttal":          "",
            }],
        }

    citations = pending.get("citations") or []
    argument  = pending.get("argument", "")
    brief     = state.get("intelligence_brief") or {}

    tag_map = {}
    for obs in (brief.get("bull_observations") or []) + (brief.get("bear_observations") or []):
        if "tag" in obs:
            tag_map[obs["tag"].upper()] = obs

    anomaly_texts = [item.upper() for item in (brief.get("key_strengths") or []) + (brief.get("key_risks") or [])]

    verified   = []
    violations = []
    support_score = 0

    for citation in citations:
        tag = citation.upper().replace("SOURCE: ", "").strip()

        if "QUANT_ALERT" in tag:
            if any("QUANT_ALERT" in text for text in anomaly_texts):
                verified.append(citation)
                support_score += 5
            else:
                violations.append(f"{citation} (Hallucinated Quant Alert)")
                support_score -= 1
            continue

        if tag in tag_map:
            original_text = tag_map[tag].get("text", "")
            numbers_in_source = re.findall(r'\d+\.?\d*', original_text)
            
            is_factual = True
            if numbers_in_source:
                is_factual = any(num in argument for num in numbers_in_source)
            
            if is_factual:
                verified.append(citation)
                is_anomaly = any(original_text[:30].upper() in at for at in anomaly_texts)
                support_score += 3 if is_anomaly else 1
            else:
                violations.append(f"{citation} (Fake/Altered numbers detected)")
                support_score -= 1
        else:
            violations.append(f"{citation} (tag not in brief)")
            support_score -= 1

    if violations and len(violations) > len(verified):
        status = "failed"
        print(f"  [Fact-Checker] ✗ {agent.upper()} FAILED — violations: {violations}")
        return {
            pending_key: {**pending, "violated_citations": violations, "fact_check_status": status},
            rewrite_key: state.get(rewrite_key, 0) + 1,
        }

    status = "passed"
    print(f"  [Fact-Checker] ✓ {agent.upper()} PASSED — support score: {support_score}")

    return {
        pending_key: {},
        rewrite_key: 0,
        arg_key: [{
            **pending,
            "support_score":      support_score,
            "fact_check_status":  status,
            "verified_citations": verified,
        }],
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 6 — CHALLENGER AGENT
# ══════════════════════════════════════════════════════════════════════════

def run_challenger(state: dict) -> dict:
    round_num = state["round_number"] + 1
    print(f"\n[Challenger Agent] Round {round_num}...")

    bull_args = state.get("bull_arguments") or []
    bear_args = state.get("bear_arguments") or []

    bull_score = sum(a.get("support_score", 0) for a in bull_args)
    bear_score = sum(a.get("support_score", 0) for a in bear_args)
    winning    = "bull" if bull_score >= bear_score else "bear"

    winning_arg = (bull_args[-1].get("argument", "") if winning == "bull" and bull_args
                   else bear_args[-1].get("argument", "") if bear_args else "")
    losing_arg  = (bear_args[-1].get("argument", "") if winning == "bull" and bear_args
                   else bull_args[-1].get("argument", "") if bull_args else "")
    win_cites   = (bull_args[-1].get("citations", []) if winning == "bull" and bull_args
                   else bear_args[-1].get("citations", []) if bear_args else [])

    previous_targets = [
        a.get("weakest_claim_targeted", "")
        for a in (state.get("challenger_arguments") or [])
    ]

    response = llm.invoke([
        SystemMessage(content=challenger_system_prompt()),
        HumanMessage(content=challenger_human_prompt(
            company          = state.get("company", "UNK"),
            round_num        = round_num,
            winning          = winning,
            winning_arg      = winning_arg,
            losing_arg       = losing_arg,
            bull_citations   = win_cites,
            bear_citations   = [],
            previous_targets = previous_targets,
        )),
    ])

    parsed = parse_llm_json(response.content)
    print(f"[Challenger Agent] ✓ Targeted: {winning} | Weakness: {str(parsed.get('weakest_claim_targeted', ''))[:60]}")

    return {
        "challenger_arguments": [{
            "round":                  round_num,
            "weakest_claim_targeted": parsed.get("weakest_claim_targeted", ""),
            "citation_gap":           parsed.get("citation_gap", ""),
            "attack":                 parsed.get("attack", ""),
            "collapse_scenario":      parsed.get("collapse_scenario", ""),
        }],
        "round_number": round_num,
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 6.5 — TARGETED REBUTTAL AGENT
# ══════════════════════════════════════════════════════════════════════════

def run_rebuttal(state: dict) -> dict:
    round_num = state["round_number"]
    print(f"\n[Rebuttal Agent] Formulating defense against Challenger...")

    bull_score = sum(a.get("support_score", 0) for a in (state.get("bull_arguments") or []))
    bear_score = sum(a.get("support_score", 0) for a in (state.get("bear_arguments") or []))
    defending_side = "bull" if bull_score >= bear_score else "bear"

    ch_args = state.get("challenger_arguments") or []
    ch_last = ch_args[-1] if ch_args else {}
    original_claim = ch_last.get("weakest_claim_targeted", "")
    attack = ch_last.get("attack", "")

    response = llm.invoke([
        SystemMessage(content=rebuttal_system_prompt()),
        HumanMessage(content=rebuttal_human_prompt(
            company        = state.get("company", "UNK"),
            brief          = state.get("intelligence_brief", {}),
            agent_side     = defending_side,
            original_claim = original_claim,
            attack         = attack
        ))
    ])

    parsed = parse_llm_json(response.content)
    print(f"  [Rebuttal] ✓ {defending_side.upper()} defended claim.")

    return {
        "targeted_rebuttals": [{
            "round": round_num,
            "defending_side": defending_side,
            "defense": parsed.get("defense", "Defense failed.")
        }]
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 7 — CALIBRATION ENGINE (No LLM)
# ══════════════════════════════════════════════════════════════════════════

def run_calibration(state: dict) -> dict:
    print(f"\n[Calibration Engine] Computing metrics...")

    max_rounds = state.get("max_rounds", 3)
    bull_args  = (state.get("bull_arguments") or [])[-max_rounds:]
    bear_args  = (state.get("bear_arguments") or [])[-max_rounds:]

    bull_total = sum(a.get("support_score", 0) for a in bull_args if a.get("fact_check_status") != "voided")
    bear_total = sum(a.get("support_score", 0) for a in bear_args if a.get("fact_check_status") != "voided")

    delta          = abs(bull_total - bear_total)
    max_possible   = max(bull_total + bear_total, 1)

    delta_pct      = round((delta / max_possible) * 100, 1)
    uncertainty    = "High" if delta_pct < 15 else "Moderate" if delta_pct < 30 else "Low"

    hallucinations = sum(
        1 for a in bull_args + bear_args
        if a.get("fact_check_status") == "voided"
    )

    metrics = {
        "bull_support_score":          bull_total,
        "bear_support_score":          bear_total,
        "convergence_delta":           delta,
        "convergence_delta_pct":       f"{delta_pct}%",
        "uncertainty_level":           uncertainty,
        "total_hallucinations_caught": hallucinations,
        "rounds_scored":               max_rounds,
    }

    print(f"[Calibration] ✓ Bull: {bull_total} | Bear: {bear_total} | Delta: {delta_pct}% | Uncertainty: {uncertainty}")

    return {"calibration_metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════
# NODE 8 — JUDGE AGENT
# ══════════════════════════════════════════════════════════════════════════

def run_judge(state: dict) -> dict:
    print(f"\n[Judge Agent] Synthesizing final decision...")

    max_rounds = state.get("max_rounds", 3)

    bull_text = format_debate_log_for_judge((state.get("bull_arguments") or [])[-max_rounds:])
    bear_text = format_debate_log_for_judge((state.get("bear_arguments") or [])[-max_rounds:])
    
    ch_text = ""
    rebuttals = (state.get("targeted_rebuttals") or [])[-max_rounds:]
    for i, a in enumerate((state.get("challenger_arguments") or [])[-max_rounds:]):
        defense = rebuttals[i].get("defense", "") if i < len(rebuttals) else "No defense provided."
        ch_text += f"R{a.get('round', '?')} Attack: '{a.get('weakest_claim_targeted', '')[:80]}' — {a.get('attack', '')[:120]}\nDefense: {defense}\n\n"

    brief = state.get("intelligence_brief") or {}
    brief_summary = json.dumps({
        "financial_health": brief.get("financial_health", {}),
        "key_strengths":    brief.get("key_strengths", []),
        "key_risks":        brief.get("key_risks", []),
        "peer_context":     brief.get("peer_context", {}),
    }, indent=2)

    response = llm.invoke([
        SystemMessage(content=judge_system_prompt()),
        HumanMessage(content=judge_human_prompt(
            company                   = state.get("company", "UNK"),
            brief_summary             = brief_summary,
            bull_arguments_text       = bull_text,
            bear_arguments_text       = bear_text,
            challenger_arguments_text = ch_text,
            calibration_metrics       = state.get("calibration_metrics", {}),
            evaluation_scores         = [],
        )),
    ])

    parsed = parse_llm_json(response.content)
    print(f"[Judge Agent] ✓ Verdict: {parsed.get('verdict', 'N/A')}")

    return {"decision_memo": parsed}
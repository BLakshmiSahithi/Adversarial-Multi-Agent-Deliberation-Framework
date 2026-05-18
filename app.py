import streamlit as st
import plotly.graph_objects as go
from graph import graph, make_initial_state

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Adversarial Multi-Agent Deliberation Framework",
    page_icon="⚖️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .header-title {
        text-align: center; font-size: 2em; font-weight: 800;
        color: #f1f5f9; margin-bottom: 4px;
    }
    .header-sub {
        text-align: center; color: #94a3b8;
        font-size: 0.95em; margin-bottom: 24px;
    }
    .verdict-box {
        border-radius: 10px; padding: 20px 28px;
        margin-bottom: 20px; font-size: 1.1em; font-weight: 600;
    }
    .invest    { background-color: #14532d; border-left: 6px solid #22c55e; color: #f0fdf4; }
    .no-invest { background-color: #450a0a; border-left: 6px solid #ef4444; color: #fef2f2; }
    .hold      { background-color: #451a03; border-left: 6px solid #f59e0b; color: #fffbeb; }
    .section-header {
        color: #64748b; font-size: 0.75em; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        margin: 20px 0 8px 0; border-bottom: 1px solid #1e293b; padding-bottom: 4px;
    }
    .claim-box {
        background: #1e293b; border-radius: 8px; padding: 12px 16px;
        margin-bottom: 8px; border-left: 4px solid #334155;
        color: #cbd5e1; font-size: 0.92em;
    }
    .bull-border { border-left-color: #22c55e; }
    .bear-border { border-left-color: #ef4444; }
    .risk-item {
        background: #1e293b; border-radius: 6px; padding: 8px 14px;
        margin-bottom: 6px; color: #fbbf24; font-size: 0.9em;
    }
    .swing-item {
        background: #1e293b; border-radius: 6px; padding: 8px 14px;
        margin-bottom: 6px; color: #a5f3fc; font-size: 0.9em;
    }
    .round-header {
        background: #1e293b; border-radius: 8px 8px 0 0; padding: 10px 16px;
        font-weight: 700; color: #f1f5f9; font-size: 1em;
        border-bottom: 2px solid #334155;
    }
    .citation-tag {
        display: inline-block; background: #1e3a5f; color: #93c5fd;
        border-radius: 4px; padding: 2px 8px; font-size: 0.78em;
        margin: 2px; font-family: monospace;
    }
    .footer {
        text-align: center; color: #334155;
        font-size: 0.78em; padding: 20px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def build_support_chart(bull_scores: list, bear_scores: list):
    if not bull_scores or not bear_scores:
        return None

    rounds = list(range(1, len(bull_scores) + 1))
    fig    = go.Figure()

    fig.add_trace(go.Scatter(
        x=rounds, y=bull_scores,
        mode="lines+markers+text",
        name="🟢 Bull Support",
        line=dict(color="#22c55e", width=3),
        marker=dict(size=12, color="#22c55e"),
        text=[str(v) for v in bull_scores],
        textposition="top center",
        textfont=dict(size=13, color="#22c55e"),
    ))

    fig.add_trace(go.Scatter(
        x=rounds, y=bear_scores,
        mode="lines+markers+text",
        name="🔴 Bear Support",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=12, color="#ef4444"),
        text=[str(v) for v in bear_scores],
        textposition="bottom center",
        textfont=dict(size=13, color="#ef4444"),
    ))

    fig.update_layout(
        title=dict(
            text="Evidence Support Score Across Debate Rounds",
            font=dict(size=16, color="#f1f5f9")
        ),
        xaxis=dict(
            title="Round", tickvals=rounds,
            tickfont=dict(size=13, color="#94a3b8"),
            gridcolor="#1e293b", color="#94a3b8",
        ),
        yaxis=dict(
            title="Support Points",
            tickfont=dict(size=13, color="#94a3b8"),
            gridcolor="#1e293b", color="#94a3b8",
        ),
        legend=dict(
            orientation="h", y=-0.25,
            font=dict(size=13, color="#f1f5f9"),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="#1e293b",
        paper_bgcolor="#0f172a",
        font=dict(color="#f1f5f9"),
        margin=dict(t=60, b=80, l=60, r=30),
        height=420,
    )
    return fig


def verdict_class(verdict: str) -> str:
    if verdict == "Invest":             return "invest"
    if verdict == "Do Not Invest":      return "no-invest"
    return "hold"


def verdict_emoji(verdict: str) -> str:
    if verdict == "Invest":             return "✅"
    if verdict == "Do Not Invest":      return "❌"
    return "⚠️"


def render_citations(citations: list):
    if not citations:
        return
    tags_html = " ".join(
        f'<span class="citation-tag">{c}</span>' for c in citations
    )
    st.markdown(f"**Citations:** {tags_html}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="header-title">⚖️ Adversarial Multi-Agent Deliberation Framework</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="header-sub">'
    'Bull · Bear · Challenger agents debate across multiple rounds.<br>'
    'A Judge synthesizes a structured investment decision memo '
    'with a full citation-verified audit trail.'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎯 Run a Debate")

    company     = st.text_input("Company Name", placeholder="e.g. Apple Inc")
    ticker      = st.text_input("Ticker Symbol", placeholder="e.g. AAPL")
    rounds      = st.slider("Debate Rounds", min_value=1, max_value=5, value=3)
    run_clicked = st.button("▶  Run Debate", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**Quick Examples**")

    examples = [
        ("Apple Inc",      "AAPL"),
        ("Tesla Inc",      "TSLA"),
        ("Microsoft Corp", "MSFT"),
        ("NVIDIA Corp",    "NVDA"),
        ("JPMorgan Chase", "JPM"),
    ]
    for ex_company, ex_ticker in examples:
        if st.button(f"{ex_company} ({ex_ticker})", use_container_width=True):
            company     = ex_company
            ticker      = ex_ticker
            run_clicked = True

    st.markdown("---")
    st.markdown(
        "<div style='color:#475569; font-size:0.78em; text-align:center'>"
        "LangGraph · Groq · LangSmith<br>"
        "yfinance · SEC EDGAR · Tavily"
        "</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if run_clicked:
    if not company or not ticker:
        st.error("Please enter both a company name and ticker symbol.")
    else:
        # ── Invoke graph ───────────────────────────────────────────────────
        with st.spinner(
            f"Running {rounds}-round debate for {company} ({ticker.upper()})..."
        ):
            try:
                initial_state = make_initial_state(company, ticker, rounds)
                result        = graph.invoke(initial_state)
            except Exception as e:
                st.error(f"Debate failed: {str(e)}")
                st.stop()

        # ── Extract results ────────────────────────────────────────────────
        memo        = result.get("decision_memo", {})
        metrics     = result.get("calibration_metrics", {})
        bull_args   = result.get("bull_arguments", [])
        bear_args   = result.get("bear_arguments", [])
        ch_args     = result.get("challenger_arguments", [])
        brief       = result.get("intelligence_brief", {})

        if not memo or "verdict" not in memo:
            st.error("Debate completed but no decision memo was generated. Try again.")
            st.stop()

        verdict    = memo.get("verdict", "N/A")
        confidence = memo.get("confidence_level", "N/A")

        # ── Tabs ──────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Decision Memo",
            "📈 Support Chart",
            "🔬 Intelligence Brief",
            "💬 Full Debate Log",
        ])

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — DECISION MEMO
        # ══════════════════════════════════════════════════════════════════
        with tab1:
            vc = verdict_class(verdict)
            ve = verdict_emoji(verdict)
            st.markdown(
                f'<div class="verdict-box {vc}">'
                f'{ve} &nbsp; <strong>VERDICT: {verdict}</strong>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp; Confidence: {confidence}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Calibration metrics (Updated with Delta Pct)
            st.markdown(
                '<div class="section-header">Calibration Metrics</div>',
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Bull Support",   metrics.get("bull_support_score",          "N/A"))
            m2.metric("Bear Support",   metrics.get("bear_support_score",          "N/A"))
            m3.metric("Delta Gap",      metrics.get("convergence_delta_pct",       "N/A"))
            m4.metric("Uncertainty",    metrics.get("uncertainty_level",           "N/A"))
            m5.metric("Hallucinations Caught", metrics.get("total_hallucinations_caught", 0))

            # Swing arguments
            st.markdown(
                '<div class="section-header">Swing Arguments — What Decided the Debate</div>',
                unsafe_allow_html=True,
            )
            for arg in memo.get("swing_arguments", []):
                st.markdown(f'<div class="swing-item">💡 {arg}</div>', unsafe_allow_html=True)

            # Strongest claims
            st.markdown(
                '<div class="section-header">Strongest Claims</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="claim-box bull-border">'
                    f'<strong style="color:#22c55e">🟢 Bull</strong><br>'
                    f'{memo.get("bull_strongest_claim", "N/A")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="claim-box bear-border">'
                    f'<strong style="color:#ef4444">🔴 Bear</strong><br>'
                    f'{memo.get("bear_strongest_claim", "N/A")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Unresolved risks
            st.markdown(
                '<div class="section-header">Unresolved Risks</div>',
                unsafe_allow_html=True,
            )
            for risk in memo.get("unresolved_risks", []):
                st.markdown(f'<div class="risk-item">⚠️ {risk}</div>', unsafe_allow_html=True)

            # Debate summary
            st.markdown(
                '<div class="section-header">Debate Summary</div>',
                unsafe_allow_html=True,
            )
            st.info(memo.get("debate_summary", "N/A"))

            # Recommended action
            st.markdown(
                '<div class="section-header">Recommended Action</div>',
                unsafe_allow_html=True,
            )
            st.success(memo.get("recommended_action", "N/A"))

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — SUPPORT SCORE CHART
        # ══════════════════════════════════════════════════════════════════
        with tab2:
            bull_scores = [b.get("support_score", 0) for b in bull_args]
            bear_scores = [br.get("support_score", 0) for br in bear_args]
            
            if bull_scores and bear_scores:
                fig = build_support_chart(bull_scores, bear_scores)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No support score history available.")

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — INTELLIGENCE BRIEF
        # ══════════════════════════════════════════════════════════════════
        with tab3:
            st.markdown(
                '<div class="section-header">Financial Health Summary</div>',
                unsafe_allow_html=True,
            )
            fh = brief.get("financial_health", {})
            if fh:
                fh1, fh2 = st.columns(2)
                with fh1:
                    st.markdown(f"**Revenue Trend:** {fh.get('revenue_trend', 'N/A')}")
                    st.markdown(f"**Margin Trend:** {fh.get('margin_trend', 'N/A')}")
                    st.markdown(f"**Balance Sheet:** {fh.get('balance_sheet', 'N/A')}")
                with fh2:
                    st.markdown(f"**Valuation Signal:** {fh.get('valuation_signal', 'N/A')}")
                    st.markdown(f"**ROE:** {fh.get('roe', 'N/A')}")
                    st.markdown(f"**Free Cashflow:** {fh.get('fcf', 'N/A')}")

            st.markdown(
                '<div class="section-header">Peer Context</div>',
                unsafe_allow_html=True,
            )
            pc = brief.get("peer_context", {})
            if pc and "note" not in pc:
                p1, p2, p3 = st.columns(3)
                p1.metric("Sector PE Avg",     pc.get("sector_pe_avg",     "N/A"))
                p2.metric("Sector Growth Avg", pc.get("sector_growth_avg", "N/A"))
                p3.metric("Sector Margin Avg", pc.get("sector_margin_avg", "N/A"))
                st.markdown(f"**PE vs Sector:** {pc.get('pe_vs_sector', 'N/A')}")
                st.markdown(f"**Growth vs Sector:** {pc.get('growth_vs_sector', 'N/A')}")
                st.markdown(f"**Peers used:** {', '.join(pc.get('peers_used', []))}")

            st.markdown(
                '<div class="section-header">Bull Observations</div>',
                unsafe_allow_html=True,
            )
            for obs in brief.get("bull_observations", []):
                st.markdown(
                    f'<div class="claim-box bull-border">'
                    f'<span class="citation-tag">{obs["tag"]}</span> &nbsp; {obs["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-header">Bear Observations</div>',
                unsafe_allow_html=True,
            )
            for obs in brief.get("bear_observations", []):
                st.markdown(
                    f'<div class="claim-box bear-border">'
                    f'<span class="citation-tag">{obs["tag"]}</span> &nbsp; {obs["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            ks = brief.get("key_strengths", [])
            kr = brief.get("key_risks", [])
            if ks or kr:
                st.markdown(
                    '<div class="section-header">Anomalies Flagged</div>',
                    unsafe_allow_html=True,
                )
                for s in ks:
                    st.success(s)
                for r in kr:
                    st.error(r)

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — FULL DEBATE LOG
        # ══════════════════════════════════════════════════════════════════
        with tab4:
            num_rounds = max(len(bull_args), len(bear_args), len(ch_args))

            for i in range(num_rounds):
                st.markdown(
                    f'<div class="round-header">ROUND {i + 1}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander(f"🟢 Bull — Round {i + 1}", expanded=True):
                    if i < len(bull_args):
                        b = bull_args[i]
                        st.metric("Support Points Earned", b.get("support_score", 0))
                        render_citations(b.get("verified_citations", []))

                        if b.get("scratchpad"):
                            with st.expander("🧠 Scratchpad (CoT reasoning)", expanded=False):
                                st.markdown(b["scratchpad"])

                        st.markdown("**Argument:**")
                        st.markdown(b.get("argument", ""))

                        if b.get("rebuttal"):
                            st.markdown(f"**↳ Rebuttal to Bear:** {b['rebuttal']}")

                        claims = b.get("key_claims", [])
                        if claims:
                            st.markdown("**Key Claims:**")
                            for c in claims:
                                st.markdown(f"- {c}")

                        status = b.get("fact_check_status", "passed")
                        if status == "passed":
                            st.success("✓ Fact-check passed")
                        else:
                            st.warning(f"⚠️ Fact-check: {status}")

                with st.expander(f"🔴 Bear — Round {i + 1}", expanded=True):
                    if i < len(bear_args):
                        br = bear_args[i]
                        st.metric("Support Points Earned", br.get("support_score", 0))
                        render_citations(br.get("verified_citations", []))

                        if br.get("scratchpad"):
                            with st.expander("🧠 Scratchpad (CoT reasoning)", expanded=False):
                                st.markdown(br["scratchpad"])

                        st.markdown("**Argument:**")
                        st.markdown(br.get("argument", ""))

                        if br.get("rebuttal"):
                            st.markdown(f"**↳ Rebuttal to Bull:** {br['rebuttal']}")

                        claims = br.get("key_claims", [])
                        if claims:
                            st.markdown("**Key Claims:**")
                            for c in claims:
                                st.markdown(f"- {c}")

                        status = br.get("fact_check_status", "passed")
                        if status == "passed":
                            st.success("✓ Fact-check passed")
                        else:
                            st.warning(f"⚠️ Fact-check: {status}")

                with st.expander(f"⚡ Challenger — Round {i + 1}", expanded=True):
                    if i < len(ch_args):
                        ch = ch_args[i]
                        st.markdown(f"**Weakest Claim Targeted:** \n{ch.get('weakest_claim_targeted', 'N/A')}")
                        st.markdown(f"**Citation Gap Identified:** \n{ch.get('citation_gap', 'N/A')}")
                        st.markdown(f"**Attack:** \n{ch.get('attack', 'N/A')}")
                        st.markdown(f"**Collapse Scenario:** \n{ch.get('collapse_scenario', 'N/A')}")
                        
                        # --- ADD THIS NEW REBUTTAL BLOCK ---
                        rebuttals = result.get("targeted_rebuttals", [])
                        if i < len(rebuttals):
                            reb = rebuttals[i]
                            color = "#22c55e" if reb['defending_side'] == "bull" else "#ef4444"
                            emoji = "🟢 Bull" if reb['defending_side'] == "bull" else "🔴 Bear"
                            st.markdown(
                                f"<div style='margin-top:15px; padding:12px; border-left:4px solid {color}; background:#1e293b; border-radius:6px;'>"
                                f"<strong style='color:{color}'>{emoji} Live Defense:</strong><br>{reb.get('defense', '')}"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                            
                st.divider()

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; color:#334155;">
        <div style="font-size:3em; margin-bottom:16px;">⚖️</div>
        <div style="font-size:1.1em; color:#475569;">
            Enter a company name and ticker in the sidebar<br>
            and click <strong>Run Debate</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Adversarial Multi-Agent Deliberation Framework · '
    'LangGraph · Groq · Streamlit</div>',
    unsafe_allow_html=True,
)
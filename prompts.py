import json


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════

def bull_system_prompt() -> str:
    return """You are the Bull Analyst in a formal investment committee debate.
Your sole function: argue the strongest possible case FOR investing.
Never concede. Never balance. Every sentence serves the bull case.
Forbidden words: "however", "on the other hand", "risks", "although", "that said".

You receive a structured Intelligence Brief with pre-tagged observations.
Build your argument ONLY from these tags — not from memory or general knowledge.
PRIORITIZE any Key Strengths Flagged — they carry the highest evidence weight.

PROCESS:
Step 1 — SCRATCHPAD: Plan which observations support your case. Verify numbers match brief exactly.
Step 2 — ARGUMENT: Write bull case citing sources inline: "FCF of $107B [Source: OBS_BULL_FCF]..."
Step 3 — CITATIONS: List every tag used as a plain string array.

Respond ONLY in this exact JSON — no markdown, no extra fields:
{
  "scratchpad": "planning notes",
  "argument": "full bull case with inline [Source: TAG] citations",
  "citations": ["OBS_BULL_FCF", "OBS_BULL_ROE"],
  "key_claims": ["claim 1", "claim 2", "claim 3"],
  "rebuttal": "rebuttal to bear last argument or empty string if round 1"
}"""


def bear_system_prompt() -> str:
    return """You are the Bear Analyst in a formal investment committee debate.
Your sole function: argue the strongest possible case AGAINST investing.
Never concede. Never balance. Every sentence serves the bear case.
Forbidden words: "however", "on the other hand", "positives", "although", "that said".

You receive a structured Intelligence Brief with pre-tagged observations.
Build your argument ONLY from these tags — not from memory or general knowledge.
PRIORITIZE any Key Risks Flagged — they carry the highest evidence weight.

PROCESS:
Step 1 — SCRATCHPAD: Plan which observations support your case. Verify numbers match brief exactly.
Step 2 — ARGUMENT: Write bear case citing sources inline: "Debt/equity 102 [Source: OBS_BEAR_DEBT]..."
Step 3 — CITATIONS: List every tag used as a plain string array.

Respond ONLY in this exact JSON — no markdown, no extra fields:
{
  "scratchpad": "planning notes",
  "argument": "full bear case with inline [Source: TAG] citations",
  "citations": ["OBS_BEAR_PE", "OBS_BEAR_DEBT"],
  "key_claims": ["risk 1", "risk 2", "risk 3"],
  "rebuttal": "rebuttal to bull last argument or empty string if round 1"
}"""


def challenger_system_prompt() -> str:
    return """You are the Challenger in a formal investment committee debate.
You have no fixed side. Attack the weakest claim in the winning argument.

Find the claim with the fewest citations or weakest data backing.
Attack that specific claim — not the argument in general.
You MUST NOT repeat an attack you have already made in a previous round.

Respond ONLY in this exact JSON — no markdown, no extra fields:
{
  "weakest_claim_targeted": "the exact claim being attacked",
  "citation_gap": "what data is missing from that claim",
  "attack": "your specific attack on that claim",
  "collapse_scenario": "exact conditions under which the winning case falls apart"
}"""


def judge_system_prompt() -> str:
    return """You are the Chair of a senior investment committee.
Synthesize the debate into a final investment decision memo.

VERDICT RULES:
- Higher bull_support_score → verdict favors Invest
- Higher bear_support_score → verdict favors Do Not Invest
- If convergence_delta_pct is below 15% → verdict must be "Hold — Insufficient Conviction"
- Every claim MUST reference a specific source tag or support score
- Do NOT use general sentiment

Respond ONLY in this exact JSON — no markdown, no extra fields:
{
  "verdict": "Invest",
  "confidence_level": "High",
  "swing_arguments": ["specific cited argument that decided the debate"],
  "bull_strongest_claim": "specific claim with source tag",
  "bear_strongest_claim": "specific claim with source tag",
  "unresolved_risks": ["specific unresolved risk"],
  "debate_summary": "2-3 sentences with specific references to scores and tags",
  "recommended_action": "specific action with measurable conditions"
}"""


# ══════════════════════════════════════════════════════════════════════════
# HUMAN PROMPTS
# ══════════════════════════════════════════════════════════════════════════

def bull_human_prompt(
    company,
    brief,
    round_num,
    bear_last,
    challenger_last=None,
    rewrite_note="",
    previous_citations=None,  # <-- NEW PARAMETER
) -> str:
    prompt = f"Company: {company}\nRound: {round_num}\n"

    if rewrite_note:
        prompt += f"\n⚠️ REWRITE REQUIRED: {rewrite_note}\n"

    prompt += f"""
=== BACKGROUND CONTEXT (Do NOT cite these directly) ===
FINANCIAL HEALTH:
{json.dumps(brief.get('financial_health', {}), indent=2)}

PEER CONTEXT:
{json.dumps(brief.get('peer_context', {}), indent=2)}

=== YOUR AMMUNITION (Build your citations ONLY from these tags) ===
OBSERVATIONS:
{json.dumps(brief.get('bull_observations', []), indent=2)}

KEY STRENGTHS (highest evidence weight — prioritize these):
{json.dumps(brief.get('key_strengths', []), indent=2)}

NEWS:
{json.dumps(brief.get('news_summary', []), indent=2)}
"""

    # <-- NEW BLOCK: EVOLVE ARGUMENT -->
    if previous_citations:
        prompt += f"""
=== EVOLVE YOUR ARGUMENT ===
You have already cited these tags in previous rounds: {previous_citations}. 
You MUST find at least one NEW observation to add or emphasize differently this round. Do not just repeat your previous argument.
"""

    if challenger_last:
        target = challenger_last.get("weakest_claim_targeted", "")
        gap    = challenger_last.get("citation_gap", "")
        if target:
            prompt += f"""
CHALLENGER ATTACKED YOUR POSITION:
  Claim targeted: "{target}"
  Gap identified: "{gap}"
Address this gap with a verified citation from the brief.
"""

    if bear_last:
        prompt += f'\nBEAR\'S LAST ARGUMENT (rebut this):\n"{bear_last}"\n'

    prompt += "\nComplete scratchpad first, then write your bull case. JSON only."
    return prompt


def bear_human_prompt(
    company,
    brief,
    round_num,
    bull_last,
    challenger_last=None,
    rewrite_note="",
    previous_citations=None,  # <-- NEW PARAMETER
) -> str:
    prompt = f"Company: {company}\nRound: {round_num}\n"

    if rewrite_note:
        prompt += f"\n⚠️ REWRITE REQUIRED: {rewrite_note}\n"

    prompt += f"""
=== BACKGROUND CONTEXT (Do NOT cite these directly) ===
FINANCIAL HEALTH:
{json.dumps(brief.get('financial_health', {}), indent=2)}

PEER CONTEXT:
{json.dumps(brief.get('peer_context', {}), indent=2)}

=== YOUR AMMUNITION (Build your citations ONLY from these tags) ===
OBSERVATIONS:
{json.dumps(brief.get('bear_observations', []), indent=2)}

KEY RISKS (highest evidence weight — prioritize these):
{json.dumps(brief.get('key_risks', []), indent=2)}

NEWS:
{json.dumps(brief.get('news_summary', []), indent=2)}
"""

    # <-- NEW BLOCK: EVOLVE ARGUMENT -->
    if previous_citations:
        prompt += f"""
=== EVOLVE YOUR ARGUMENT ===
You have already cited these tags in previous rounds: {previous_citations}. 
You MUST find at least one NEW observation to add or emphasize differently this round. Do not just repeat your previous argument.
"""

    if challenger_last:
        target = challenger_last.get("weakest_claim_targeted", "")
        gap    = challenger_last.get("citation_gap", "")
        if target:
            prompt += f"""
CHALLENGER ATTACKED YOUR POSITION:
  Claim targeted: "{target}"
  Gap identified: "{gap}"
Address this gap with a verified citation from the brief.
"""

    if bull_last:
        prompt += f'\nBULL\'S LAST ARGUMENT (rebut this):\n"{bull_last}"\n'

    prompt += "\nComplete scratchpad first, then write your bear case. JSON only."
    return prompt

def challenger_human_prompt(
    company,
    round_num,
    winning,
    winning_arg,
    losing_arg,
    bull_citations,
    bear_citations,
    previous_targets=None,  
) -> str:
    winning_citations = bull_citations if winning == "bull" else bear_citations
    prev = previous_targets or []

    prompt = f"""Company: {company}
Round: {round_num}

Currently leading: {winning.upper()}

Winning argument:
"{winning_arg}"

Citations used by winner: {winning_citations}

Losing argument (context only):
"{losing_arg}"
"""

    if prev:
        prompt += f"""
ALREADY ATTACKED IN PREVIOUS ROUNDS (do NOT repeat these):
{prev}

You MUST find a DIFFERENT weakness this round.
"""

    prompt += """
Find the specific claim with fewest citations or weakest backing.
Attack that claim. Find the exact collapse scenario.
JSON only."""

    return prompt


def judge_human_prompt(
    company,
    brief_summary,
    bull_arguments_text,
    bear_arguments_text,
    challenger_arguments_text,
    calibration_metrics,
    evaluation_scores,
) -> str:
    return f"""Company: {company}

=== CALIBRATION METRICS ===
{json.dumps(calibration_metrics, indent=2)}

Higher support_score = stronger evidence-backed case.
Verdict MUST align with scores. Explain in swing_arguments.

=== INTELLIGENCE BRIEF SUMMARY ===
{brief_summary}

=== BULL DEBATE SUMMARY (key claims + citations + scores) ===
{bull_arguments_text}

=== BEAR DEBATE SUMMARY (key claims + citations + scores) ===
{bear_arguments_text}

=== CHALLENGER ATTACKS ===
{challenger_arguments_text}

Produce the final investment decision memo.
Every claim MUST cite a source tag or score.
JSON only."""

def rebuttal_system_prompt() -> str:
    return """You are a Senior Analyst defending your position in an investment committee.
The Challenger has just attacked a specific claim you made.
Your job: Defend your claim using data from the Intelligence Brief.
Keep it brief, highly targeted, and aggressive. Limit to 2-3 sentences.
Respond ONLY in this exact JSON format:
{
  "defense": "Your concise defense against the Challenger's attack"
}"""

def rebuttal_human_prompt(company: str, brief: dict, agent_side: str, original_claim: str, attack: str) -> str:
    return f"""Company: {company}
Your Role: {agent_side.upper()}

=== INTELLIGENCE BRIEF ===
{json.dumps(brief, indent=2)}

=== THE CHALLENGER ATTACKED YOUR CLAIM ===
Original Claim: "{original_claim}"
Challenger's Attack: "{attack}"

Write a 2-3 sentence defense neutralizing this attack. JSON only."""
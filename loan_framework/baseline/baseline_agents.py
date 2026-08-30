"""
PHASE 2: Baseline Agents using CSO

These agents use the discovered factors and CSO structure
but WITHOUT reading prior stage context (no CSO propagation).
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from loan_framework.cso.cso_framework import ContextStateObject

load_dotenv()  # Load variables from .env file

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# Load discovered factors
def load_discovered_factors():
    try:
        with open("discovered_factors.json", "r") as f:
            data = json.load(f)
        return data["discovered_factors"]
    except FileNotFoundError:
        return None


DISCOVERED_FACTORS = load_discovered_factors()


def baseline_stage_1_verification(cso: ContextStateObject) -> ContextStateObject:
    """Baseline Stage 1 - same as context-aware"""
    verification_pass = not cso["bankruptcy_history"] and cso["experience_years"] >= 1
    cso["stage_1_verification_status"] = "pass" if verification_pass else "fail"
    cso["stage_1_flags"] = []
    return cso


def baseline_stage_2_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    """
    Baseline Stage 2 - Uses discovered factors but NO CSO context
    Does not read Stage 1 results from CSO
    """

    prompt = f"""
You have been given DISCOVERED FACTORS from analyzing the training dataset.

DISCOVERED FACTORS:
{DISCOVERED_FACTORS}

Now evaluate THIS APPLICANT on the discovered factors.
You do NOT have prior verification or context - only the applicant's financial data.

APPLICANT DATA:
- Credit Score: {cso["credit_score"]:.0f}
- Annual Income: ${cso["annual_income"]:,.0f}
- Net Worth: ${cso["net_worth"]:,.0f}
- DTI: {cso["debt_to_income"]:.2%}
- Total DTI: {cso["total_debt_to_income"]:.2%}
- Previous Loan Defaults: {cso["previous_loan_defaults"]}
- Loan Amount: ${cso["loan_amount"]:,.0f}
- Employment: {cso["experience_years"]} years
- Credit Card Utilization: {cso["credit_card_utilization"]:.1%}
- Open Credit Lines: {cso["number_of_open_credit_lines"]}

TASK: Assess this applicant on the discovered factors (no prior context).

Format:
DISCOVERED FACTOR ASSESSMENT:
[How applicant scores on each discovered factor]

CREDIT TIER: [Based on discovered factors]

REASONING: [Why]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        assessment = response.content[0].text

        assessment_upper = assessment.upper()
        if "EXCELLENT" in assessment_upper:
            tier = "Excellent"
        elif "VERY GOOD" in assessment_upper:
            tier = "Very Good"
        elif "GOOD" in assessment_upper:
            tier = "Good"
        elif "FAIR" in assessment_upper:
            tier = "Fair"
        else:
            tier = "Good"
    except:
        tier = "Good"
        assessment = "Credit assessment based on discovered factors"

    cso["stage_2_credit_band"] = tier
    cso["stage_2_assessment"] = assessment
    cso["stage_2_identified_factors"] = []
    cso["stage_2_prior_stage_context_used"] = False  # KEY: No context
    return cso


def baseline_stage_3_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    """
    Baseline Stage 3 - Uses discovered factors but NO CSO context
    Does not read Stage 1 or Stage 2 results from CSO
    """

    prompt = f"""
You have been given DISCOVERED FACTORS from analyzing the training dataset.

DISCOVERED FACTORS:
{DISCOVERED_FACTORS}

Assess THIS APPLICANT on the discovered factors.
You do NOT have prior assessment results - only the applicant's financial data.

APPLICANT DATA:
- Income: ${cso["annual_income"]:,.0f}
- Net Worth: ${cso["net_worth"]:,.0f}
- Loan Amount: ${cso["loan_amount"]:,.0f}
- DTI: {cso["debt_to_income"]:.2%}
- Total DTI: {cso["total_debt_to_income"]:.2%}
- Credit Score: {cso["credit_score"]:.0f}
- Employment: {cso["experience_years"]} years
- Previous Defaults: {cso["previous_loan_defaults"]}
- Monthly Debt Payments: ${cso["monthly_debt_payments"]:,.0f}

TASK: Assess risk using discovered factors (no prior context).

Format:
DISCOVERED FACTOR ASSESSMENT:
[How applicant scores on each discovered factor]

RISK LEVEL: [Low/Medium/High/Very High]

REASONING: [Why]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = response.content[0].text

        analysis_upper = analysis.upper()
        if "VERY HIGH" in analysis_upper:
            risk = "Very High"
        elif "HIGH" in analysis_upper:
            risk = "High"
        elif "MEDIUM" in analysis_upper:
            risk = "Medium"
        elif "LOW" in analysis_upper:
            risk = "Low"
        else:
            risk = "Medium"
    except:
        risk = "Medium"
        analysis = "Risk assessment based on discovered factors"

    cso["stage_3_risk_level"] = risk
    cso["stage_3_analysis"] = analysis
    cso["stage_3_risk_drivers"] = []
    cso["stage_3_prior_stage_context_used"] = False  # KEY: No context
    return cso


def baseline_stage_4_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    """
    Baseline Stage 4 - Uses discovered factors but NO CSO context
    Does not read results from Stages 1, 2, or 3 from CSO
    """

    prompt = f"""
You have been given DISCOVERED FACTORS from analyzing the training dataset.

DISCOVERED FACTORS:
{DISCOVERED_FACTORS}

Make a final decision for THIS APPLICANT.
You do NOT have prior assessment results - only the applicant's financial data.

APPLICANT DATA:
- Income: ${cso["annual_income"]:,.0f}
- Net Worth: ${cso["net_worth"]:,.0f}
- Loan Amount: ${cso["loan_amount"]:,.0f}
- DTI: {cso["debt_to_income"]:.2%}
- Total DTI: {cso["total_debt_to_income"]:.2%}
- Credit Score: {cso["credit_score"]:.0f}
- Employment: {cso["experience_years"]} years
- Previous Defaults: {cso["previous_loan_defaults"]}
- Bankruptcy: {"Yes" if cso["bankruptcy_history"] else "No"}

TASK: Make final decision using discovered factors (no prior context).

Format:
DISCOVERED FACTOR ASSESSMENT:
[How applicant scores on each discovered factor]

FINAL DECISION: [APPROVED or REJECTED]

CONFIDENCE: [0-100%]

REASONING: [Why]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        decision_text = response.content[0].text

        decision_upper = decision_text.upper()
        if "APPROVED" in decision_upper:
            decision = "APPROVED"
        elif "REJECTED" in decision_upper:
            decision = "REJECTED"
        else:
            decision = "APPROVED"

        confidence = 0.75
        if "CONFIDENCE:" in decision_text:
            conf_section = decision_text.split("CONFIDENCE:")[1]
            conf_text = "".join(
                c for c in conf_section.split("\n")[0] if c.isdigit() or c == "."
            )
            try:
                confidence = (
                    float(conf_text) / 100.0
                    if float(conf_text) > 1
                    else float(conf_text)
                )
            except:
                confidence = 0.75
    except:
        decision = "APPROVED"
        decision_text = "Decision based on discovered factors"
        confidence = 0.5

    # Create audit trail (showing no prior context used)
    audit_trail = [
        f"✓ Stage 1 [Verification]: (not read)",
        f"✓ Stage 2 [Credit]: (not read)",
        f"✓ Stage 3 [Risk]: (not read)",
        f"✓ Stage 4 [Decision]: {decision} (Confidence: {confidence:.0%}) - No prior context",
    ]

    cso["stage_4_decision"] = decision
    cso["stage_4_decision_text"] = decision_text
    cso["stage_4_confidence"] = confidence
    cso["stage_4_consolidated_factors"] = []
    cso["stage_4_prior_stage_context_used"] = False  # KEY: No context
    cso["stage_4_audit_trail"] = audit_trail

    # Evaluate correctness
    cso["decision_correct"] = (decision == "APPROVED") == (
        cso["loan_approved_actual"] == 1
    )

    return cso


def run_baseline_pipeline_phase2(cso: ContextStateObject) -> ContextStateObject:
    """Run baseline using discovered factors but NO CSO context propagation"""

    cso = baseline_stage_1_verification(cso)
    # Stage 2 doesn't see Stage 1
    cso = baseline_stage_2_with_discovered_factors(cso)
    # Stage 3 doesn't see Stages 1-2
    cso = baseline_stage_3_with_discovered_factors(cso)
    # Stage 4 doesn't see Stages 1-3
    cso = baseline_stage_4_with_discovered_factors(cso)

    return cso

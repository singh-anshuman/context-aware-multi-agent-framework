"""
PHASE 2: Baseline Agents using CSO

These agents use the discovered factors and CSO structure
but WITHOUT reading prior stage context (no CSO propagation).
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from loan_framework.cso.agents import stage_1_document_verification
from loan_framework.cso.cso_framework import ContextStateObject

load_dotenv()  # Load variables from .env file

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ============================================================================
# BASELINE VERSIONS (No Context)
# ============================================================================


def baseline_stage_2(cso: ContextStateObject) -> ContextStateObject:
    """Baseline Stage 2 - No prior context from Stage 1"""

    print(f"\n[STAGE 2 - BASELINE] Credit assessment for {cso['applicant_id']}")

    prompt = f"""CREDIT ASSESSMENT (Independent Analysis)
 
Assess this applicant's creditworthiness based on their credit profile.
 
CREDIT TIERS:
- EXCELLENT: Credit Score 750+
- VERY GOOD: Credit Score 720-749
- GOOD: Credit Score 680-719
- FAIR: Credit Score 650-679
- POOR: Credit Score < 650
 
APPLICANT:
- Credit Score: {cso["credit_score"]:.0f}
- Income: ${cso["annual_income"]:,.0f}
- Employment: {cso["employment_status"]}
- Payment History: {cso["payment_history"]:.0f}%
 
Determine credit tier and provide brief reasoning.
 
Output: CREDIT TIER: [Tier] | REASONING: [reason]"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        assessment = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

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
        assessment = "Credit assessment"
        input_tokens = 0
        output_tokens = 0

    cso["stage_2_credit_band"] = tier
    cso["stage_2_assessment"] = assessment
    cso["stage_2_prior_stage_context_used"] = False

    print(f"  Credit Tier: {tier}")
    print(f"  Tokens → Total: {input_tokens + output_tokens}")

    return cso


def baseline_stage_3(cso: ContextStateObject) -> ContextStateObject:
    """Baseline Stage 3 - No prior context"""

    print(f"\n[STAGE 3 - BASELINE] Risk assessment for {cso['applicant_id']}")

    prompt = f"""RISK ASSESSMENT (Independent Analysis)
 
Assess this applicant's default risk.
 
RISK LEVELS:
- LOW: DTI <30%, Income $100k+, Payment History 99%+
- MEDIUM: DTI 30-45%, Income $60-100k, Payment History 97-99%
- HIGH: DTI 45-55%, Income <$60k, Payment History <97%
- VERY HIGH: DTI >55%, Income <$40k, Any prior defaults
 
APPLICANT:
- Income: ${cso["annual_income"]:,.0f}
- Debt-to-Income: {cso["total_debt_to_income"]:.1%}
- Payment History: {cso["payment_history"]:.0f}%
- Prior Defaults: {cso["previous_loan_defaults"]}
 
Determine risk level.
 
Output: RISK LEVEL: [Low/Medium/High/Very High]"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        analysis_upper = analysis.upper()
        if "VERY HIGH" in analysis_upper:
            risk = "Very High"
        elif "HIGH" in analysis_upper:
            risk = "High"
        elif "MEDIUM" in analysis_upper:
            risk = "Medium"
        else:
            risk = "Low"
    except:
        risk = "Medium"
        analysis = "Risk assessment"
        input_tokens = 0
        output_tokens = 0

    cso["stage_3_risk_level"] = risk
    cso["stage_3_analysis"] = analysis
    cso["stage_3_prior_stage_context_used"] = False

    print(f"  Risk Level: {risk}")
    print(f"  Tokens → Total: {input_tokens + output_tokens}")

    return cso


def baseline_stage_4(cso: ContextStateObject) -> ContextStateObject:
    """Baseline Stage 4 - No prior context"""

    print(f"\n[STAGE 4 - BASELINE] Final decision for {cso['applicant_id']}")

    prompt = f"""FINAL DECISION (Independent Analysis)
 
Make approval/rejection decision based on applicant profile alone.
 
HARD STOPS: Any prior defaults, Unemployed, DTI >55%, Bankruptcy
APPROVE IF: Good credit (700+), Decent income ($60k+), DTI <40%
REJECT IF: Poor credit (<650), Low income (<$40k), DTI >45%
 
APPLICANT:
- Credit: {cso["credit_score"]:.0f}
- Income: ${cso["annual_income"]:,.0f}
- DTI: {cso["total_debt_to_income"]:.1%}
- Defaults: {cso["previous_loan_defaults"]}
- Bankruptcy: {"Yes" if cso["bankruptcy_history"] else "No"}
 
Decision: APPROVED or REJECTED?"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        decision_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        if "APPROVED" in decision_text.upper():
            decision = "APPROVED"
        else:
            decision = "REJECTED"
    except:
        decision = "APPROVED"
        decision_text = "Decision"
        input_tokens = 0
        output_tokens = 0

    cso["stage_4_decision"] = decision
    cso["stage_4_decision_text"] = decision_text
    cso["stage_4_prior_stage_context_used"] = False
    cso["stage_4_audit_trail"] = [f"Decision: {decision}"]
    cso["decision_correct"] = (decision == "APPROVED") == (
        cso["loan_approved_actual"] == 1
    )

    print(f"  FINAL DECISION: {decision}")
    print(f"Ground Truth: {'APPROVED' if cso['loan_approved_actual'] else 'REJECTED'}")
    print(f"  Correct: {cso['decision_correct']}")

    return cso


def run_baseline_pipeline(cso):
    print("\nBASELINE PIPELINE\n")
    cso = stage_1_document_verification(cso)
    cso = baseline_stage_2(cso)
    cso = baseline_stage_3(cso)
    cso = baseline_stage_4(cso)

    return cso

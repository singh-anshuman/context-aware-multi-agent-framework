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
        print(f"  LLM Response: {assessment}")
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
        print(f"  LLM Response: {analysis}")
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

    prompt = f"""FINAL LOAN DECISION FRAMEWORK
     
    You are making a final APPROVAL/REJECTION decision based on all analysis.
     
    KEY DECISION CRITERIA:
    ─────────────────────────────────────────────────────
     
    HARD STOPS (Automatic REJECTION):
    - Bankruptcy history
    - Unemployment status  
    - DTI > 55%
    - Previous loan defaults (2+)
     
    STRONG APPROVAL FACTORS (Favor APPROVAL):
    - Credit score 720+
    - Income $100k+
    - DTI < 30%
    - Net worth $300k+
    - Home owner (mortgage/own)
    - Stable employment (5+ years)
    - Perfect payment history (99%+)
    - No prior defaults
     
    STRONG REJECTION FACTORS (Favor REJECTION):
    - Credit score < 650
    - Income < $40k
    - DTI > 45%
    - Net worth < $50k
    - Recent credit inquiries
    - Payment history < 95%
    - Previous defaults (1)
     
    APPLICANT FINAL PROFILE:
    ─────────────────────────────────────────────────────
    Credit Score:        {cso["credit_score"]:.0f}
    Annual Income:       ${cso["annual_income"]:,.0f}
    Debt-to-Income:      {cso["total_debt_to_income"]:.1%}
    Employment:          {cso["employment_status"]} ({cso["job_tenure"] / 12:.1f} years)
    Payment History:     {cso["payment_history"]:.0f}%
    Prior Defaults:      {cso["previous_loan_defaults"]}
    Net Worth:           ${cso["net_worth"]:,.0f}
    Home Status:         {cso["home_ownership_status"]}
    Bankruptcy:          {"YES" if cso["bankruptcy_history"] else "NO"}
     
    TASK:
    1. Check for any HARD STOPS → If found, REJECT
    2. Count APPROVAL FACTORS vs REJECTION FACTORS
    3. Make decision based on balance
    4. Provide clear reasoning
     
    Output format:
    FINAL DECISION: [APPROVED or REJECTED]
    REASONING: [Key factors driving decision]"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        decision_text = response.content[0].text
        print(f"  LLM Response: {decision_text}")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Extract decision
        decision_upper = decision_text.upper()
        if "APPROVED" in decision_upper:
            decision = "APPROVED"
        elif "REJECTED" in decision_upper:
            decision = "REJECTED"
        else:
            decision = "APPROVED"
    except Exception as e:
        print(f"  Error: {str(e)}")
        decision = "APPROVED"
        decision_text = "Decision based on applicant profile"
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

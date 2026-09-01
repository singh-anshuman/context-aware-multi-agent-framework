import os

from anthropic import Anthropic
from dotenv import load_dotenv

from loan_framework.cso.cso_framework import ContextStateObject

load_dotenv()  # Load variables from .env file

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ============================================================================
# STAGE 1: DOCUMENT VERIFICATION (Rule-Based - No Changes)
# ============================================================================


def stage_1_document_verification(cso: ContextStateObject) -> ContextStateObject:
    """
    Stage 1: Document Verification

    This stage applies HARD STOPS - absolute disqualifiers.
    No LLM needed; these are rule-based checks.
    """
    verification_pass = not cso["bankruptcy_history"] and cso["experience_years"] >= 1
    cso["stage_1_verification_status"] = "pass" if verification_pass else "fail"
    cso["stage_1_flags"] = []

    if cso["bankruptcy_history"]:
        cso["stage_1_flags"].append("Bankruptcy History - HARD STOP")

    if cso["experience_years"] < 1:
        cso["stage_1_flags"].append("Insufficient Work Experience - HARD STOP")

    print(f"\n[STAGE 1] Document verification for {cso['applicant_id']}")
    print(f"  Verification Status: {cso['stage_1_verification_status'].upper()}")
    print(f"  Tokens → Input: 0 | Output: 0 | Total: 0 (Rule-based)")

    return cso


# ============================================================================
# STAGE 2: CREDIT ASSESSMENT
# ============================================================================


def stage_2_credit_assessment_with_discovered_factors(
    cso: ContextStateObject, discovered_factors_text: str = ""
) -> ContextStateObject:
    print(f"\n[STAGE 2] Credit assessment for {cso['applicant_id']}")

    # Build the prompt with explicit tier framework
    prompt = f"""CREDIT ASSESSMENT - TIER-BASED FRAMEWORK
 
You are assessing a loan applicant's creditworthiness using a tier-based system.
 
TIER DEFINITIONS (Credit Score is primary indicator):
─────────────────────────────────────────────────────
 
TIER 1: EXCELLENT (750+)
- Outstanding credit profile
- Minimal default risk
- Strong approval candidate
 
TIER 2: VERY GOOD (720-749)  
- Good credit history
- Low default risk
- Favorable approval candidate
 
TIER 3: GOOD (680-719)
- Acceptable credit history
- Moderate default risk
- Neutral - requires other factors
 
TIER 4: FAIR (650-679)
- Below average but manageable
- Higher default risk
- Challenging - needs strong other factors
 
TIER 5: POOR (<650)
- Significant credit concerns
- High default risk
- Major concern for approval
 
APPLICANT CREDIT PROFILE:
─────────────────────────────────────────────────────
Credit Score:        {cso["credit_score"]:.0f}
Annual Income:       ${cso["annual_income"]:,.0f}
Employment Status:   {cso["employment_status"]}
Payment History:     {cso["payment_history"]:.0f}%
Length of Credit:    {cso["length_of_credit_history"] / 12:.1f} years
Credit Inquiries:    {cso["number_of_credit_inquiries"]} (recent: higher = more risk)
 
TASK:
1. Identify which TIER this applicant fits based on credit score
2. Note any secondary credit factors (payment history, inquiries, etc.)
3. Provide tier and brief reasoning
 
Be decisive. Use clear tiers. Credit score is primary indicator.
 
Output format:
CREDIT TIER: [Tier Name]
REASONING: [1-2 sentences]"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        assessment = response.content[0].text
        print(f"  LLM Response: {assessment}")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Extract tier from response
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

    except Exception as e:
        print(f"  Error: {str(e)}")
        tier = "Good"
        assessment = "Credit assessment"
        input_tokens = 0
        output_tokens = 0

    cso["stage_2_credit_band"] = tier
    cso["stage_2_assessment"] = assessment
    cso["stage_2_prior_stage_context_used"] = True

    total_tokens = input_tokens + output_tokens
    print(f"  Credit Tier: {tier}")
    print(
        f"  Tokens → Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}"
    )

    return cso


# ============================================================================
# STAGE 3: RISK ASSESSMENT
# ============================================================================


def stage_3_risk_assessment_with_discovered_factors(
    cso: ContextStateObject, discovered_factors_text: str = ""
) -> ContextStateObject:
    print(f"\n[STAGE 3] Risk assessment for {cso['applicant_id']}")

    # Use Stage 2 context
    stage_2_context = f"Stage 2 Credit Assessment: {cso['stage_2_credit_band']}\n"

    prompt = f"""RISK ASSESSMENT - MULTI-FACTOR FRAMEWORK
 
You are assessing LOAN DEFAULT RISK using multiple factors.
 
RISK FACTOR ANALYSIS:
─────────────────────────────────────────────────────
 
1. DEBT-TO-INCOME RATIO (Repayment Capacity) - PRIMARY INDICATOR
   <30%:   Excellent capacity (Low risk)
   30-36%: Good capacity (Low-Medium risk)
   36-45%: Acceptable capacity (Medium risk)
   45-55%: Strained capacity (Medium-High risk)
   >55%:   Overextended (High risk)
 
2. INCOME STABILITY
   Employed, 5+ years at job: Stable (Lower risk)
   Employed, 1-5 years:       Moderate (Medium risk)
   Self-employed:             Variable (Higher risk)
   Unemployed:                No income (Very High risk)
 
3. PAYMENT HISTORY
   99%+:  Excellent reliability (Lower risk)
   97-99%: Good reliability (Low-Medium risk)
   <95%:  Concerning pattern (Higher risk)
 
4. CREDIT DEFAULTS
   0 defaults:   No prior failure (Lower risk)
   1 default:    History of failure (Higher risk)
   2+ defaults:  Pattern of failure (Very High risk)
 
5. ASSETS & SAFETY NET
   Large net worth (>$300k): Can absorb losses (Lower risk)
   Moderate ($100-300k):     Reasonable cushion (Medium risk)
   Limited (<$100k):         Vulnerable (Higher risk)
 
APPLICANT RISK PROFILE:
─────────────────────────────────────────────────────
{stage_2_context}
Total Debt-to-Income: {cso["total_debt_to_income"]:.1%}
Annual Income:        ${cso["annual_income"]:,.0f}
Payment History:      {cso["payment_history"]:.0f}%
Previous Defaults:    {cso["previous_loan_defaults"]}
Net Worth:            ${cso["net_worth"]:,.0f}
Job Tenure:           {cso["job_tenure"] / 12:.1f} years
 
TASK:
1. Evaluate the applicant on each risk dimension
2. Synthesize into overall risk level
3. Identify PRIMARY RISK DRIVERS
 
Risk Levels: Low / Medium / High / Very High
 
Output format:
RISK LEVEL: [Level]
PRIMARY RISKS: [Main concerns, if any]"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = response.content[0].text
        print(f"  LLM Response: {analysis}")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Extract risk level
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

    except Exception as e:
        print(f"  Error: {str(e)}")
        risk = "Medium"
        analysis = "Risk assessment"
        input_tokens = 0
        output_tokens = 0

    cso["stage_3_risk_level"] = risk
    cso["stage_3_analysis"] = analysis
    cso["stage_3_prior_stage_context_used"] = True

    total_tokens = input_tokens + output_tokens
    print(f"  Risk Level: {risk}")
    print(
        f"  Tokens → Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}"
    )

    return cso


# ============================================================================
# STAGE 4: FINAL DECISION
# ============================================================================


def stage_4_final_decision_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    print(f"\n[STAGE 4] Final decision for {cso['applicant_id']}")

    # Build context from prior stages
    prior_context = f"""PRIOR ASSESSMENTS:
Stage 1 Verification: {cso["stage_1_verification_status"].upper()}
Stage 2 Credit Assessment: {cso["stage_2_credit_band"]} tier
Stage 3 Risk Assessment: {cso["stage_3_risk_level"]} risk"""

    prompt = f"""FINAL LOAN DECISION FRAMEWORK
 
You are making a final APPROVAL/REJECTION decision based on all analysis.
 
{prior_context}
 
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
            max_tokens=180,
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

    # Create audit trail
    audit_trail = [
        f"✓ Stage 1 [Verification]: {cso['stage_1_verification_status'].upper()}",
        f"✓ Stage 2 [Credit]: {cso['stage_2_credit_band']} - Context used",
        f"✓ Stage 3 [Risk]: {cso['stage_3_risk_level']} - Context used",
        f"✓ Stage 4 [Decision]: {decision} - Full context analysis",
    ]

    cso["stage_4_decision"] = decision
    cso["stage_4_decision_text"] = decision_text
    cso["stage_4_prior_stage_context_used"] = True
    cso["stage_4_audit_trail"] = audit_trail

    # Evaluate correctness
    cso["decision_correct"] = (decision == "APPROVED") == (
        cso["loan_approved_actual"] == 1
    )

    total_tokens = input_tokens + output_tokens
    print(f"FINAL DECISION: {decision}")
    print(f"Ground Truth: {'APPROVED' if cso['loan_approved_actual'] else 'REJECTED'}")
    print(f"Correct: {cso['decision_correct']}")
    print(
        f"Tokens → Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}"
    )

    return cso


def run_context_aware_pipeline(cso):
    print("\nCONTEXT-AWARE PIPELINE\n")
    cso = stage_1_document_verification(cso)
    cso = stage_2_credit_assessment_with_discovered_factors(cso)
    cso = stage_3_risk_assessment_with_discovered_factors(cso)
    cso = stage_4_final_decision_with_discovered_factors(cso)

    return cso

from anthropic import Anthropic, APIError
from cso_framework import ContextStateObject

client = Anthropic()


def stage_1_document_verification(state: ContextStateObject):
    """Stage 1: Document Verification"""
    print(f"\n[STAGE 1] Processing applicant {state['applicant_id']}")

    verification_pass = True
    flags = []

    if state["bankruptcy_history"]:
        verification_pass = False
        flags.append("bankruptcy_flag: true")
    else:
        flags.append("bankruptcy_flag: false")

    if state["experience_years"] < 2:
        verification_pass = False
        flags.append(f"employment_unstable: {state['experience_years']} years")
    else:
        flags.append(f"employment_stable: {state['experience_years']} years")

    if not state["education_level"] or state["education_level"] == "Unknown":
        flags.append("education_not_verified")
    else:
        flags.append(f"education_verified: {state['education_level']}")

    state["stage_1_verification_status"] = "pass" if verification_pass else "fail"
    state["stage_1_flags"] = flags

    print(f"  Status: {state['stage_1_verification_status']}")
    return state


def stage_2_credit_scoring(state: ContextStateObject):
    """Stage 2: Credit Scoring with LLM reasoning"""
    print(f"\n[STAGE 2] Credit analysis for {state['applicant_id']}")

    credit_score = state["credit_score"]
    if credit_score >= 800:
        band = "Excellent"
    elif credit_score >= 740:
        band = "Very Good"
    elif credit_score >= 670:
        band = "Good"
    elif credit_score >= 580:
        band = "Fair"
    else:
        band = "Poor"

    dti = state["debt_to_income"]
    if dti <= 0.36:
        tier = "Low"
    elif dti <= 0.5:
        tier = "Medium"
    else:
        tier = "High"

    prompt = f"""
    Applicant Financial Profile:
    - Credit Score: {credit_score} ({band})
    - Debt-to-Income Ratio: {dti:.2%}
    - Annual Income: ${state["annual_income"]:,.0f}
    - Previous Defaults: {state["previous_defaults"]}
    - Stage 1 Verification: {state["stage_1_verification_status"]}
    
    Briefly assess creditworthiness in 1-2 sentences.
    """

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        rationale = response.content[0].text
    except APIError:
        rationale = f"Credit assessment complete. {band} tier applicant."

    state["stage_2_credit_band"] = band
    state["stage_2_risk_tier"] = tier
    state["stage_2_rationale"] = rationale

    print(f"  Credit Band: {band}")
    return state


def stage_3_risk_assessment(state: ContextStateObject):
    """Stage 3: Risk Assessment with full CSO context"""
    print(f"\n[STAGE 3] Risk assessment for {state['applicant_id']}")

    prompt = f"""
    Comprehensive Risk Assessment:
    
    STAGE 1 VERIFICATION RESULT:
    - Status: {state["stage_1_verification_status"]}
    - Flags: {", ".join(state["stage_1_flags"])}
    
    STAGE 2 CREDIT RESULT:
    - Credit Band: {state["stage_2_credit_band"]}
    - Risk Tier: {state["stage_2_risk_tier"]}
    
    APPLICANT FINANCIALS:
    - Loan Amount: ${state["loan_amount"]:,.0f}
    - Net Worth: ${state["net_worth"]:,.0f}
    - Risk Score: {state["risk_score"]}
    
    Given this full context, classify OVERALL risk as Low/Medium/High/VeryHigh.
    """

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        assessment = response.content[0].text
    except APIError:
        assessment = "Risk assessment complete based on financial profile."

    if state["risk_score"] < 40:
        risk_class = "Low"
    elif state["risk_score"] < 60:
        risk_class = "Medium"
    elif state["risk_score"] < 80:
        risk_class = "High"
    else:
        risk_class = "Very High"

    state["stage_3_risk_classification"] = risk_class
    state["stage_3_risk_score"] = state["risk_score"]
    state["stage_3_risk_flags"] = [assessment[:100]]

    print(f"  Overall Risk: {risk_class}")
    return state


def stage_4_final_approval(state: ContextStateObject):
    """Stage 4: Final Approval with complete CSO"""
    print(f"\n[STAGE 4] Final decision for {state['applicant_id']}")

    prompt = f"""
    FINAL LOAN APPROVAL DECISION
    
    VERIFICATION (Stage 1): {state["stage_1_verification_status"].upper()}
    CREDIT ASSESSMENT (Stage 2): {state["stage_2_credit_band"]}
    RISK ASSESSMENT (Stage 3): {state["stage_3_risk_classification"]} (Score: {state["stage_3_risk_score"]:.0f})
    
    DECISION CRITERIA:
    1. Verification: {state["stage_1_verification_status"]} {"✓" if state["stage_1_verification_status"] == "pass" else "✗"}
    2. No Bankruptcy: {"✓" if not state["bankruptcy_history"] else "✗"}
    3. Risk Score < 60: {"✓" if state["stage_3_risk_score"] < 60 else "✗"}
    4. DTI <= 0.55: {"✓" if state["debt_to_income"] <= 0.55 else "✗"}
    
    Make final decision: APPROVED or REJECTED? Provide 2-3 sentence explanation.
    """

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        decision_text = response.content[0].text
    except APIError:
        decision_text = "Decision made based on applicant profile and risk assessment."

    criteria_met = (
        state["stage_1_verification_status"] == "pass"
        and not state["bankruptcy_history"]
        and state["stage_3_risk_score"] < 60
        and state["debt_to_income"] <= 0.55
    )

    decision = "APPROVED" if criteria_met else "REJECTED"

    audit_trail = [
        f"✓ Stage 1 [Verification]: {state['stage_1_verification_status'].upper()}",
        f"✓ Stage 2 [Credit]: {state['stage_2_credit_band']}, Tier: {state['stage_2_risk_tier']}",
        f"✓ Stage 3 [Risk]: {state['stage_3_risk_classification']}, Score: {state['stage_3_risk_score']:.0f}",
        f"✓ Stage 4 [Decision]: {decision}",
    ]

    state["stage_4_decision"] = decision
    state["stage_4_rationale"] = decision_text
    state["stage_4_audit_trail"] = audit_trail

    print(f"  FINAL DECISION: {decision}")
    return state

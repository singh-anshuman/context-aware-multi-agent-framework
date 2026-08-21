import os

from anthropic import Anthropic, APIError
from anthropic.types import Message
from cso_framework import ContextStateObject
from dotenv import load_dotenv

load_dotenv()


client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


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
        response: Message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        rationale = response.content[0].text
    except APIError as error:
        print(f"  Credit analysis error: {error}")
        rationale = f"Credit assessment complete. {band} tier applicant."
    except Exception as error:
        print(f"  Unexpected credit analysis error: {error}")
        rationale = f"Credit assessment complete. {band} tier applicant."

    state["stage_2_credit_band"] = band
    state["stage_2_risk_tier"] = tier
    state["stage_2_rationale"] = rationale

    print(f"  Credit Band: {band}")
    return state


def stage_3_risk_assessment(state):
    # LLM reads FULL CSO and makes decision
    prompt = f"""
    You are a senior risk officer.
    
    STAGE 1 - VERIFICATION:
    Status: {state["stage_1_verification_status"]}
    Flags: {", ".join(state["stage_1_flags"])}
    
    STAGE 2 - CREDIT:
    Credit Band: {state["stage_2_credit_band"]}
    Risk Tier: {state["stage_2_risk_tier"]}
    
    FINANCIAL PROFILE:
    Income: ${state["annual_income"]:,.0f}
    DTI: {state["debt_to_income"]:.2%}
    Risk Score: {state["risk_score"]}
    
    Classify risk as: Low/Medium/High/Very High
    Provide 3-5 key factors.
    Explain reasoning.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract LLM's decision
    text = response.content[0].text.upper()
    if "VERY HIGH" in text:
        risk_class = "Very High"
    elif "HIGH" in text:
        risk_class = "High"
    elif "MEDIUM" in text:
        risk_class = "Medium"
    else:
        risk_class = "Low"

    state["stage_3_risk_classification"] = risk_class
    return state


def stage_4_final_approval(state):
    # LLM reads COMPLETE application and decides
    prompt = f"""
    You are the final loan approval authority.
    
    === COMPLETE APPLICATION ===
    
    STAGE 1 - VERIFICATION:
    {state["stage_1_verification_status"].upper()}
    {", ".join(state["stage_1_flags"])}
    
    STAGE 2 - CREDIT:
    Score: {state["credit_score"]}
    Band: {state["stage_2_credit_band"]}
    Tier: {state["stage_2_risk_tier"]}
    Assessment: {state["stage_2_rationale"][:300]}
    
    STAGE 3 - RISK:
    Classification: {state["stage_3_risk_classification"]}
    Score: {state["stage_3_risk_score"]:.0f}
    Factors: {", ".join(state["stage_3_risk_flags"])}
    
    FINANCIALS:
    Income: ${state["annual_income"]:,.0f}
    Net Worth: ${state["net_worth"]:,.0f}
    Loan Amount: ${state["loan_amount"]:,.0f}
    DTI: {state["debt_to_income"]:.2%}
    Bankruptcy: {"Yes" if state["bankruptcy_history"] else "No"}
    Defaults: {state["previous_defaults"]}
    
    === YOUR DECISION ===
    
    YOU ARE THE DECISION MAKER.
    
    Provide:
    DECISION: [APPROVED or REJECTED]
    CONFIDENCE: [0-100%]
    RATIONALE: [2-3 sentences]
    KEY FACTORS: [3-5 factors]
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    decision_text = response.content[0].text

    # Extract the LLM's decision (not rule-based)
    decision_upper = decision_text.upper()

    if "DECISION: APPROVED" in decision_upper:
        final_decision = "APPROVED"  # LLM DECIDED
    elif "DECISION: REJECTED" in decision_upper:
        final_decision = "REJECTED"  # LLM DECIDED
    else:
        # Ask for clarification
        clarify = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[
                {"role": "user", "content": decision_text},
                {"role": "user", "content": "APPROVED or REJECTED? One word."},
            ],
        )
        text = clarify.content[0].text.upper()
        final_decision = "APPROVED" if "APPROVED" in text else "REJECTED"

    state["stage_4_decision"] = final_decision  # LLM DECIDED
    state["stage_4_rationale"] = decision_text  # LLM REASONING
    return state

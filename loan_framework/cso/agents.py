"""
PHASE 2: Agents Using Pre-Discovered Factors with ContextStateObject

These agents use the factors identified in Phase 1 (from training data)
and the ContextStateObject to track state through all stages.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from loan_framework.cso.cso_framework import (
    ContextStateObject,
    get_all_prior_context,
    get_stage_1_context,
)

load_dotenv()  # Load variables from .env file

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# Load the discovered factors from Phase 1
def load_discovered_factors():
    """Load factors discovered from training data"""
    try:
        file_path = (
            Path(__file__).resolve().parent.parent.parent
            / "loan_framework"
            / "factor_discovery"
            / "discovered_factors.json"
        )
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["discovered_factors"]
    except FileNotFoundError:
        print("ERROR: discovered_factors.json not found!")
        print("Did you run phase_1_factor_discovery_FIXED.py first?")
        return None


# Store discovered factors globally
DISCOVERED_FACTORS = load_discovered_factors()


def stage_1_document_verification(cso: ContextStateObject) -> ContextStateObject:
    """
    Stage 1: Document Verification
    Rule-based: Check data completeness and hard stop flags

    Input: ContextStateObject with applicant data
    Output: CSO updated with stage_1_* fields
    """
    print(f"\n[STAGE 1] Processing applicant {cso['applicant_id']}")

    verification_pass = True
    flags = []
    hard_stops = []

    # Hard stop 1: Bankruptcy
    if cso["bankruptcy_history"]:
        verification_pass = False
        hard_stops.append("HARD STOP - Bankruptcy history detected")
    else:
        flags.append("No bankruptcy history")

    # Hard stop 2: Employment
    if cso["experience_years"] < 1:
        verification_pass = False
        hard_stops.append("HARD STOP - Insufficient employment history (< 1 year)")
    else:
        flags.append(f"Employment history: {cso['experience_years']} years")

    # Check 3: Education
    if not cso.get("education_level") or cso["education_level"] == "Unknown":
        flags.append("Education level not provided")
    else:
        flags.append(f"Education verified: {cso['education_level']}")

    # Update CSO with Stage 1 outputs
    cso["stage_1_verification_status"] = "pass" if verification_pass else "fail"
    cso["stage_1_flags"] = flags
    cso["stage_1_hard_stops"] = hard_stops

    print(f"  Status: {cso['stage_1_verification_status']}")
    print(f"  Flags: {', '.join(flags[:2])}")

    return cso


def stage_2_credit_assessment_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    """
    Stage 2: Credit Assessment
    Uses DISCOVERED FACTORS (from training data)
    Reads Stage 1 context from CSO
    """
    print(f"\n[STAGE 2] Credit assessment for {cso['applicant_id']}")

    # Get Stage 1 context from CSO
    stage_1_context = get_stage_1_context(cso)

    prompt = f"""
You have been given a set of DISCOVERED FACTORS from analyzing the training dataset.

DISCOVERED FACTORS (from training data):
{DISCOVERED_FACTORS}

PRIOR STAGE CONTEXT (Stage 1 - Verification):
Status: {stage_1_context["verification_status"]}
Flags: {", ".join(stage_1_context["flags"])}

Now, evaluate THIS APPLICANT on the discovered factors:

APPLICANT FINANCIAL DATA:
- Credit Score: {cso["credit_score"]:.0f}
- Annual Income: ${cso["annual_income"]:,.0f}
- Net Worth: ${cso["net_worth"]:,.0f}
- Debt-to-Income Ratio: {cso["debt_to_income"]:.2%}
- Total DTI: {cso["total_debt_to_income"]:.2%}
- Credit Card Utilization: {cso["credit_card_utilization"]:.1%}
- Previous Loan Defaults: {cso["previous_loan_defaults"]}
- Requested Loan Amount: ${cso["loan_amount"]:,.0f}
- Employment: {cso["experience_years"]} years
- Open Credit Lines: {cso["number_of_open_credit_lines"]}
- Credit History Length: {cso["length_of_credit_history"]} months

TASK: Assess THIS APPLICANT on the discovered factors.

For each relevant discovered factor:
1. How does this applicant score on that factor?
2. Is it favorable or unfavorable?
3. Overall credit assessment based on the discovered factors.

Format:
FACTOR ASSESSMENT:
- [Factor 1]: [Assessment]
- [Factor 2]: [Assessment]
- [Factor 3]: [Assessment]

OVERALL CREDIT ASSESSMENT: [Tier based on discovered factors]

REASONING: [How discovered factors apply to this applicant]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        assessment = response.content[0].text

        # Extract credit tier
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

        # Extract factors mentioned
        factors = []
        if "FACTOR ASSESSMENT:" in assessment:
            factors_section = assessment.split("FACTOR ASSESSMENT:")[1]
            if "OVERALL CREDIT" in factors_section:
                factors_section = factors_section.split("OVERALL CREDIT")[0]
            factors = [
                line.strip()
                for line in factors_section.split("\n")
                if line.strip().startswith("-")
            ][:3]

    except Exception as e:
        print(f"  Error: {str(e)}")
        assessment = f"Error: {str(e)}"
        tier = "Good"
        factors = []

    # Update CSO with Stage 2 outputs
    cso["stage_2_credit_band"] = tier
    cso["stage_2_assessment"] = assessment
    cso["stage_2_identified_factors"] = factors
    cso["stage_2_discovered_factors_used"] = [
        "Credit Score",
        "DTI Ratio",
        "Employment History",
    ]
    cso["stage_2_factor_scores"] = {
        "credit_score": cso["credit_score"],
        "dti": cso["debt_to_income"],
        "employment": cso["experience_years"],
    }

    print(f"  Credit Tier: {tier}")
    print(f"  Identified {len(factors)} factors")

    return cso


def stage_3_risk_assessment_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    """
    Stage 3: Risk Assessment
    Uses DISCOVERED FACTORS (from training data)
    Reads Stage 1 and Stage 2 context from CSO
    """
    print(f"\n[STAGE 3] Risk assessment for {cso['applicant_id']}")

    # Get prior context from CSO
    prior_context = get_all_prior_context(cso, up_to_stage=2)

    prompt = f"""
You have been given a set of DISCOVERED FACTORS from analyzing the training dataset.

DISCOVERED FACTORS (from training data):
{DISCOVERED_FACTORS}

PRIOR STAGE ANALYSIS:
{prior_context}

Now, assess THIS APPLICANT'S RISK using the DISCOVERED FACTORS:

APPLICANT DATA:
- Income: ${cso["annual_income"]:,.0f}
- Net Worth: ${cso["net_worth"]:,.0f}
- Loan Amount: ${cso["loan_amount"]:,.0f}
- DTI: {cso["debt_to_income"]:.2%}
- Total DTI: {cso["total_debt_to_income"]:.2%}
- Credit Score: {cso["credit_score"]:.0f}
- Employment: {cso["experience_years"]} years
- Previous Defaults: {cso["previous_loan_defaults"]}
- Savings: ${cso["savings_account_balance"]:,.0f}
- Monthly Debt: ${cso["monthly_debt_payments"]:,.0f}

TASK: Using ONLY the discovered factors and prior stage results:
1. Assess applicant on each discovered factor
2. Determine overall risk level
3. Explain how discovered factors apply to this applicant

Format:
DISCOVERED FACTOR ASSESSMENT:
- [Factor from training]: [How this applicant scores]
- [Factor from training]: [How this applicant scores]
- [Factor from training]: [How this applicant scores]

OVERALL RISK LEVEL: [Low/Medium/High/Very High - based on discovered factors]

REASONING: [How discovered factors combine for this applicant]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = response.content[0].text

        # Extract risk level
        analysis_upper = analysis.upper()
        if "VERY HIGH" in analysis_upper:
            risk_level = "Very High"
        elif "HIGH" in analysis_upper:
            risk_level = "High"
        elif "MEDIUM" in analysis_upper:
            risk_level = "Medium"
        elif "LOW" in analysis_upper:
            risk_level = "Low"
        else:
            risk_level = "Medium"

        # Extract risk drivers
        drivers = []
        if "DISCOVERED FACTOR ASSESSMENT:" in analysis:
            drivers_section = analysis.split("DISCOVERED FACTOR ASSESSMENT:")[1]
            if "OVERALL RISK LEVEL:" in drivers_section:
                drivers_section = drivers_section.split("OVERALL RISK LEVEL:")[0]
            drivers = [
                line.strip()
                for line in drivers_section.split("\n")
                if line.strip().startswith("-")
            ][:5]

    except Exception as e:
        print(f"  Error: {str(e)}")
        analysis = f"Error: {str(e)}"
        risk_level = "Medium"
        drivers = []

    # Update CSO with Stage 3 outputs
    cso["stage_3_risk_level"] = risk_level
    cso["stage_3_analysis"] = analysis
    cso["stage_3_risk_drivers"] = drivers
    cso["stage_3_prior_stage_context_used"] = True  # We used Stage 1 and 2
    cso["stage_3_discovered_factors_used"] = [
        "Credit Quality",
        "Employment Stability",
        "Debt Burden",
        "Loan Size",
    ]

    print(f"  Risk Level: {risk_level}")
    print(f"  Identified {len(drivers)} risk drivers")

    return cso


def stage_4_final_decision_with_discovered_factors(
    cso: ContextStateObject,
) -> ContextStateObject:
    """
    Stage 4: Final Decision
    Uses DISCOVERED FACTORS (from training data)
    Reads ALL prior stage context from CSO
    """
    print(f"\n[STAGE 4] Final decision for {cso['applicant_id']}")

    # Get all prior context from CSO
    prior_context = get_all_prior_context(cso, up_to_stage=3)

    prompt = f"""
You have been given a set of DISCOVERED FACTORS from analyzing the training dataset.

DISCOVERED FACTORS (from training data):
{DISCOVERED_FACTORS}

COMPLETE PRIOR STAGE ANALYSIS:
{prior_context}

Now, make the FINAL DECISION using the DISCOVERED FACTORS:

APPLICANT PROFILE:
- Income: ${cso["annual_income"]:,.0f}
- Net Worth: ${cso["net_worth"]:,.0f}
- Loan Amount: ${cso["loan_amount"]:,.0f}
- DTI: {cso["debt_to_income"]:.2%}
- Total DTI: {cso["total_debt_to_income"]:.2%}
- Credit Score: {cso["credit_score"]:.0f}
- Employment: {cso["experience_years"]} years

TASK: Using ONLY the discovered factors and ALL prior assessments:
1. How does this applicant score on each discovered factor?
2. Do discovered factors support approval or rejection?
3. Make a final APPROVED/REJECTED decision
4. Your confidence level (0-100%)

Format:
DISCOVERED FACTORS ASSESSMENT:
[Assess applicant on each discovered factor, referencing prior stages]

SYNTHESIS: [How factors combine for decision, considering all prior analysis]

FINAL DECISION: [APPROVED or REJECTED]

CONFIDENCE: [0-100%]

RATIONALE: [Why, based on discovered factors and prior context]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        decision_text = response.content[0].text

        # Extract decision
        decision_upper = decision_text.upper()
        if "APPROVED" in decision_upper:
            final_decision = "APPROVED"
        elif "REJECTED" in decision_upper:
            final_decision = "REJECTED"
        else:
            # Conservative fallback
            if (
                cso["stage_1_verification_status"] != "pass"
                or cso["stage_3_risk_level"] == "Very High"
            ):
                final_decision = "REJECTED"
            else:
                final_decision = "APPROVED"

        # Extract confidence
        confidence = 0.80
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
                confidence = 0.80

        # Extract consolidated factors
        consolidated_factors = []
        if "DISCOVERED FACTORS ASSESSMENT:" in decision_text:
            factors_section = decision_text.split("DISCOVERED FACTORS ASSESSMENT:")[1]
            if "SYNTHESIS:" in factors_section:
                factors_section = factors_section.split("SYNTHESIS:")[0]
            consolidated_factors = [
                line.strip()
                for line in factors_section.split("\n")
                if line.strip() and not line.strip().startswith("[")
            ][:5]

    except Exception as e:
        print(f"  Error: {str(e)}")
        decision_text = f"Error: {str(e)}"
        final_decision = "APPROVED"
        confidence = 0.5
        consolidated_factors = []

    # Create audit trail
    audit_trail = [
        f"✓ Stage 1 [Verification]: {cso['stage_1_verification_status'].upper()}",
        f"✓ Stage 2 [Credit]: {cso['stage_2_credit_band']} - Used discovered factors",
        f"✓ Stage 3 [Risk]: {cso['stage_3_risk_level']} - Used discovered factors + prior context",
        f"✓ Stage 4 [Decision]: {final_decision} (Confidence: {confidence:.0%}) - Used all prior context",
    ]

    # Update CSO with Stage 4 outputs
    cso["stage_4_decision"] = final_decision
    cso["stage_4_decision_text"] = decision_text
    cso["stage_4_confidence"] = confidence
    cso["stage_4_consolidated_factors"] = consolidated_factors
    cso["stage_4_prior_stage_context_used"] = True  # We used all prior stages
    cso["stage_4_discovered_factors_used"] = ["All discovered factors from training"]
    cso["stage_4_audit_trail"] = audit_trail

    # Evaluate correctness
    cso["decision_correct"] = (final_decision == "APPROVED") == (
        cso["loan_approved_actual"] == 1
    )

    print(f"  FINAL DECISION: {final_decision} (Confidence: {confidence:.0%})")
    print(
        f"  Ground Truth: {'APPROVED' if cso['loan_approved_actual'] else 'REJECTED'}"
    )
    print(f"  Correct: {cso['decision_correct']}")

    return cso

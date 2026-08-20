from typing import TypedDict


class ContextStateObject(TypedDict):
    """Complete state object that flows through pipeline"""

    # Initialization
    applicant_id: str
    age: int
    annual_income: float
    net_worth: float
    loan_amount: float
    bankruptcy_history: bool
    credit_score: float
    experience_years: int
    education_level: str
    debt_to_income: float
    previous_defaults: int
    risk_score: float
    loan_approved_actual: int

    # Stage 1 outputs
    stage_1_verification_status: str
    stage_1_flags: list

    # Stage 2 outputs
    stage_2_credit_band: str
    stage_2_risk_tier: str
    stage_2_rationale: str

    # Stage 3 outputs
    stage_3_risk_classification: str
    stage_3_risk_flags: list
    stage_3_risk_score: float

    # Stage 4 outputs
    stage_4_decision: str
    stage_4_rationale: str
    stage_4_audit_trail: list

    # Metadata
    processing_time: dict

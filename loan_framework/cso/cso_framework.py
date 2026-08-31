"""
Context State Object (CSO) Definition - OPTIMIZED VERSION

This TypedDict defines the complete state that flows through all 4 stages
of the loan underwriting pipeline. It accumulates context as each stage
adds its analysis and decisions.

All fields mapped to actual CSV column names.

OPTIMIZATION CHANGES (from original):
- Removed stage_4_confidence: No longer tracking confidence (not needed for evaluation)
- Removed stage_2_identified_factors: Simplified parsing (factors still used via discovered_factors)
- Removed stage_3_risk_drivers: Simplified parsing (risk still assessed via analysis text)
- Removed stage_4_consolidated_factors: Simplified output (not needed for thesis metrics)
- Removed stage_3_risk_score_computed: Simplified risk assessment
- Removed stage_4_conditions: Simplified recommendations
- Removed stage_4_recommendations: Simplified recommendations
- Added prior_stage_context_used flags to Stages 2, 3, 4: Track context propagation

These changes reduce token consumption while maintaining evaluation quality.
The CSO still provides complete audit trails and decision tracking.
"""

from typing import Optional, TypedDict


class ContextStateObject(TypedDict):
    """
    Complete Context State Object for loan underwriting pipeline.

    Accumulates information as applicant flows through 4 stages:
    Stage 1: Document Verification (hard stops)
    Stage 2: Credit Assessment (using discovered factors)
    Stage 3: Risk Assessment (using discovered factors + prior context)
    Stage 4: Final Decision (using all prior context)
    """

    # ========================================================================
    # INITIAL APPLICANT DATA (from CSV)
    # ========================================================================

    # Application and Personal Info
    applicant_id: str  # Unique identifier
    application_date: str  # CSV: ApplicationDate
    age: int  # CSV: Age
    marital_status: str  # CSV: MaritalStatus
    number_of_dependents: int  # CSV: NumberOfDependents

    # Employment and Income
    employment_status: str  # CSV: EmploymentStatus
    experience_years: int  # CSV: Experience (years at current job)
    job_tenure: int  # CSV: JobTenure (months)
    annual_income: float  # CSV: AnnualIncome (annual salary)
    monthly_income: float  # CSV: MonthlyIncome
    education_level: str  # CSV: EducationLevel

    # Assets and Liabilities
    net_worth: float  # CSV: NetWorth (total assets - liabilities)
    total_assets: float  # CSV: TotalAssets
    total_liabilities: float  # CSV: TotalLiabilities
    savings_account_balance: float  # CSV: SavingsAccountBalance
    checking_account_balance: float  # CSV: CheckingAccountBalance

    # Debt Information
    monthly_debt_payments: float  # CSV: MonthlyDebtPayments
    debt_to_income: float  # CSV: DebtToIncomeRatio
    total_debt_to_income: float  # CSV: TotalDebtToIncomeRatio

    # Credit Profile
    credit_score: float  # CSV: CreditScore
    credit_card_utilization: float  # CSV: CreditCardUtilizationRate (0-1)
    number_of_open_credit_lines: int  # CSV: NumberOfOpenCreditLines
    number_of_credit_inquiries: int  # CSV: NumberOfCreditInquiries
    length_of_credit_history: int  # CSV: LengthOfCreditHistory (months)
    payment_history: str  # CSV: PaymentHistory
    previous_loan_defaults: int  # CSV: PreviousLoanDefaults (count)
    bankruptcy_history: bool  # CSV: BankruptcyHistory (0 or 1)

    # Loan Details
    loan_amount: float  # CSV: LoanAmount
    loan_duration: int  # CSV: LoanDuration (months)
    loan_purpose: str  # CSV: LoanPurpose
    base_interest_rate: float  # CSV: BaseInterestRate
    interest_rate: float  # CSV: InterestRate
    monthly_loan_payment: float  # CSV: MonthlyLoanPayment

    # Home and Bills
    home_ownership_status: str  # CSV: HomeOwnershipStatus
    utility_bills_payment_history: str  # CSV: UtilityBillsPaymentHistory

    # Pre-calculated Risk (baseline comparison)
    risk_score: float  # CSV: RiskScore (0-100)

    # Ground Truth (for evaluation only)
    loan_approved_actual: int  # CSV: LoanApproved (0 or 1, ground truth)

    # ========================================================================
    # STAGE 1 OUTPUTS: Document Verification
    # ========================================================================

    stage_1_verification_status: str  # 'pass' or 'fail'
    stage_1_flags: list[str]  # List of verification notes/flags
    stage_1_hard_stops: list[str]  # Disqualifying factors if any
    stage_1_timestamp: Optional[str]  # When verification occurred

    # ========================================================================
    # STAGE 2 OUTPUTS: Credit Assessment (using discovered factors)
    # ========================================================================

    stage_2_credit_band: str  # 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor'
    stage_2_assessment: str  # Full LLM assessment text
    stage_2_factor_scores: dict  # Individual factor scores
    stage_2_discovered_factors_used: list[str]  # Which discovered factors were applied
    stage_2_prior_stage_context_used: bool  # Whether Stage 1 context was used
    stage_2_timestamp: Optional[str]  # When assessment occurred

    # ========================================================================
    # STAGE 3 OUTPUTS: Risk Assessment (using discovered factors + prior context)
    # ========================================================================

    stage_3_risk_level: str  # 'Low', 'Medium', 'High', 'Very High'
    stage_3_analysis: str  # Full LLM risk analysis text
    stage_3_discovered_factors_used: list[str]  # Which discovered factors were applied
    stage_3_prior_stage_context_used: (
        bool  # Whether prior stages (1-2) context was used
    )
    stage_3_timestamp: Optional[str]  # When assessment occurred

    # ========================================================================
    # STAGE 4 OUTPUTS: Final Decision (using all prior context)
    # ========================================================================

    stage_4_decision: str  # 'APPROVED' or 'REJECTED'
    stage_4_decision_text: str  # Full LLM decision reasoning
    stage_4_discovered_factors_used: list[str]  # Which discovered factors were applied
    stage_4_prior_stage_context_used: (
        bool  # Whether prior stages (1-3) influenced decision
    )
    stage_4_audit_trail: list[str]  # Complete decision audit trail
    stage_4_timestamp: Optional[str]  # When decision was made

    # ========================================================================
    # EVALUATION FIELDS (for measuring pipeline performance)
    # ========================================================================

    decision_correct: bool  # Whether final decision matches ground truth
    stage_2_timing: Optional[float]  # Time taken for stage 2 (seconds)
    stage_3_timing: Optional[float]  # Time taken for stage 3 (seconds)
    stage_4_timing: Optional[float]  # Time taken for stage 4 (seconds)
    total_processing_time: Optional[float]  # Total time for pipeline (seconds)
    error_occurred: bool  # Whether any stage had an error
    error_messages: list[str]  # Any error messages

    # ========================================================================
    # METADATA
    # ========================================================================

    pipeline_version: str  # Pipeline version used
    discovered_factors_version: str  # Which factor set was used
    evaluation_phase: str  # 'training' or 'test'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_empty_cso(applicant_id: str) -> ContextStateObject:
    """Create an empty CSO with default values (optimized version)"""
    return {
        "applicant_id": applicant_id,
        # Stage 1
        "stage_1_verification_status": "",
        "stage_1_flags": [],
        "stage_1_hard_stops": [],
        # Stage 2
        "stage_2_credit_band": "",
        "stage_2_assessment": "",
        "stage_2_factor_scores": {},
        "stage_2_discovered_factors_used": [],
        "stage_2_prior_stage_context_used": False,
        # Stage 3
        "stage_3_risk_level": "",
        "stage_3_analysis": "",
        "stage_3_discovered_factors_used": [],
        "stage_3_prior_stage_context_used": False,
        # Stage 4
        "stage_4_decision": "",
        "stage_4_decision_text": "",
        "stage_4_discovered_factors_used": [],
        "stage_4_prior_stage_context_used": False,
        "stage_4_audit_trail": [],
        # Evaluation
        "error_occurred": False,
        "error_messages": [],
    }


def create_cso_from_csv_row(
    row, applicant_id: str, evaluation_phase: str = "test"
) -> ContextStateObject:
    """
    Create a CSO populated from a CSV row.

    Args:
        row: pandas Series or dict with CSV data
        applicant_id: Unique identifier for applicant
        evaluation_phase: 'training' or 'test'

    Returns:
        ContextStateObject with initial applicant data populated
    """

    def safe_float(value, default=0.0):
        try:
            return float(value) if value is not None else default
        except:
            return default

    def safe_int(value, default=0):
        try:
            return int(value) if value is not None else default
        except:
            return default

    def safe_bool(value, default=False):
        try:
            return bool(value) if value is not None else default
        except:
            return default

    def safe_str(value, default=""):
        try:
            return str(value) if value is not None else default
        except:
            return default

    cso = create_empty_cso(applicant_id)

    # Populate initial applicant data from CSV
    cso["application_date"] = safe_str(row.get("ApplicationDate"))
    cso["age"] = safe_int(row.get("Age"))
    cso["marital_status"] = safe_str(row.get("MaritalStatus"))
    cso["number_of_dependents"] = safe_int(row.get("NumberOfDependents"))

    cso["employment_status"] = safe_str(row.get("EmploymentStatus"))
    cso["experience_years"] = safe_int(row.get("Experience"))
    cso["job_tenure"] = safe_int(row.get("JobTenure"))
    cso["annual_income"] = safe_float(row.get("AnnualIncome"))
    cso["monthly_income"] = safe_float(row.get("MonthlyIncome"))
    cso["education_level"] = safe_str(row.get("EducationLevel"))

    cso["net_worth"] = safe_float(row.get("NetWorth"))
    cso["total_assets"] = safe_float(row.get("TotalAssets"))
    cso["total_liabilities"] = safe_float(row.get("TotalLiabilities"))
    cso["savings_account_balance"] = safe_float(row.get("SavingsAccountBalance"))
    cso["checking_account_balance"] = safe_float(row.get("CheckingAccountBalance"))

    cso["monthly_debt_payments"] = safe_float(row.get("MonthlyDebtPayments"))
    cso["debt_to_income"] = safe_float(row.get("DebtToIncomeRatio"))
    cso["total_debt_to_income"] = safe_float(row.get("TotalDebtToIncomeRatio"))

    cso["credit_score"] = safe_float(row.get("CreditScore"))
    cso["credit_card_utilization"] = safe_float(row.get("CreditCardUtilizationRate"))
    cso["number_of_open_credit_lines"] = safe_int(row.get("NumberOfOpenCreditLines"))
    cso["number_of_credit_inquiries"] = safe_int(row.get("NumberOfCreditInquiries"))
    cso["length_of_credit_history"] = safe_int(row.get("LengthOfCreditHistory"))
    cso["payment_history"] = safe_str(row.get("PaymentHistory"))
    cso["previous_loan_defaults"] = safe_int(row.get("PreviousLoanDefaults"))
    cso["bankruptcy_history"] = safe_bool(row.get("BankruptcyHistory"))

    cso["loan_amount"] = safe_float(row.get("LoanAmount"))
    cso["loan_duration"] = safe_int(row.get("LoanDuration"))
    cso["loan_purpose"] = safe_str(row.get("LoanPurpose"))
    cso["base_interest_rate"] = safe_float(row.get("BaseInterestRate"))
    cso["interest_rate"] = safe_float(row.get("InterestRate"))
    cso["monthly_loan_payment"] = safe_float(row.get("MonthlyLoanPayment"))

    cso["home_ownership_status"] = safe_str(row.get("HomeOwnershipStatus"))
    cso["utility_bills_payment_history"] = safe_str(
        row.get("UtilityBillsPaymentHistory")
    )

    cso["risk_score"] = safe_float(row.get("RiskScore"))
    cso["loan_approved_actual"] = safe_int(row.get("LoanApproved"))

    cso["evaluation_phase"] = evaluation_phase
    cso["pipeline_version"] = "2.0"
    cso["error_occurred"] = False
    cso["error_messages"] = []

    return cso


def print_cso_summary(cso: ContextStateObject) -> str:
    """Print a human-readable summary of CSO contents (optimized)"""
    summary = f"""
    {"=" * 70}
    CONTEXT STATE OBJECT SUMMARY - Applicant {cso.get("applicant_id", "UNKNOWN")}
    {"=" * 70}

    INITIAL APPLICANT DATA:
    Age: {cso.get("age")}
    Annual Income: ${cso.get("annual_income", 0):,.0f}
    Net Worth: ${cso.get("net_worth", 0):,.0f}
    Credit Score: {cso.get("credit_score", 0):.0f}
    Employment: {cso.get("experience_years", 0)} years
    DTI Ratio: {cso.get("debt_to_income", 0):.2%}
    Loan Requested: ${cso.get("loan_amount", 0):,.0f}

    STAGE 1 - VERIFICATION:
    Status: {cso.get("stage_1_verification_status", "N/A")}
    Flags: {", ".join(cso.get("stage_1_flags", []))}

    STAGE 2 - CREDIT ASSESSMENT:
    Credit Band: {cso.get("stage_2_credit_band", "N/A")}
    Used Prior Context: {cso.get("stage_2_prior_stage_context_used", False)}

    STAGE 3 - RISK ASSESSMENT:
    Risk Level: {cso.get("stage_3_risk_level", "N/A")}
    Used Prior Context: {cso.get("stage_3_prior_stage_context_used", False)}

    STAGE 4 - FINAL DECISION:
    Decision: {cso.get("stage_4_decision", "N/A")}
    Used Prior Context: {cso.get("stage_4_prior_stage_context_used", False)}
    Ground Truth: {"APPROVED" if cso.get("loan_approved_actual") else "REJECTED"}
    Correct: {cso.get("decision_correct", None)}

    PERFORMANCE:
    Total Time: {cso.get("total_processing_time", 0):.2f}s
    Errors: {cso.get("error_occurred", False)}

    {"=" * 70}
    """
    return summary


# ============================================================================
# STAGE-SPECIFIC CSO ACCESSORS
# ============================================================================


def get_stage_1_context(cso: ContextStateObject) -> dict:
    """Get Stage 1 outputs from CSO"""
    return {
        "verification_status": cso.get("stage_1_verification_status"),
        "flags": cso.get("stage_1_flags", []),
        "hard_stops": cso.get("stage_1_hard_stops", []),
    }


def get_stage_2_context(cso: ContextStateObject) -> dict:
    """Get Stage 2 outputs from CSO"""
    return {
        "credit_band": cso.get("stage_2_credit_band"),
        "assessment": cso.get("stage_2_assessment"),
        "factor_scores": cso.get("stage_2_factor_scores", {}),
        "prior_context_used": cso.get("stage_2_prior_stage_context_used"),
    }


def get_stage_3_context(cso: ContextStateObject) -> dict:
    """Get Stage 3 outputs from CSO"""
    return {
        "risk_level": cso.get("stage_3_risk_level"),
        "analysis": cso.get("stage_3_analysis"),
        "prior_context_used": cso.get("stage_3_prior_stage_context_used"),
    }


def get_all_prior_context(cso: ContextStateObject, up_to_stage: int) -> str:
    """Get formatted string of all prior stage context"""

    context_parts = []

    if up_to_stage >= 1:
        s1 = get_stage_1_context(cso)
        context_parts.append(
            f"STAGE 1 - VERIFICATION:\nStatus: {s1['verification_status']}\nFlags: {', '.join(s1['flags']) if s1['flags'] else 'None'}"
        )

    if up_to_stage >= 2:
        s2 = get_stage_2_context(cso)
        context_parts.append(
            f"STAGE 2 - CREDIT:\nTier: {s2['credit_band']}\nContext Used: {s2['prior_context_used']}"
        )

    if up_to_stage >= 3:
        s3 = get_stage_3_context(cso)
        context_parts.append(
            f"STAGE 3 - RISK:\nLevel: {s3['risk_level']}\nContext Used: {s3['prior_context_used']}"
        )

    return "\n\n".join(context_parts)

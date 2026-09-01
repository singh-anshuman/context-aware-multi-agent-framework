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

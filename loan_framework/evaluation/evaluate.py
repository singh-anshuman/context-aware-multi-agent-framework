"""
PHASE 2: Evaluation on Test Data Using ContextStateObject

Uses discovered factors to evaluate test applicants with proper CSO flow
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from loan_framework.baseline.baseline_agents import run_baseline_pipeline
from loan_framework.cso.agents import run_context_aware_pipeline

print("\n" + "=" * 80)
print("PHASE 2: TEST DATA EVALUATION")
print("=" * 80)

# ============================================================================
# LOAD DISCOVERED FACTORS
# ============================================================================

# Load discovered factors info
file_path = (
    Path(__file__).resolve().parent.parent.parent
    / "loan_framework"
    / "factor_discovery"
    / "discovered_factors.json"
)
with open(file_path, "r") as f:
    factor_info = json.load(f)

print("\nUsing factors discovered from training data...")
print(f"Training dataset size: {factor_info['training_stats']['total_records']:,}")
print(
    f"Approval rate in training: {factor_info['training_stats']['approval_rate']:.1%}"
)
top_factor_names = []
for factor in factor_info["top_factors"][:3]:
    if isinstance(factor, dict):
        top_factor_names.append(factor.get("factor", str(factor)))
    else:
        top_factor_names.append(str(factor))
print(f"Top factors: {', '.join(top_factor_names)}\n")

file_path = (
    Path(__file__).resolve().parent.parent.parent
    / "loan_framework"
    / "evaluation"
    / "test_data.csv"
)
test_df = pd.read_csv(file_path)


if not Path(file_path).exists():
    print(f"❌ Error: {file_path} not found")
    sys.exit()

print(f"Loading test data from {file_path}...")
test_df = pd.read_csv(file_path)

print(f"Test records: {len(test_df):,}")
print(f"Columns in dataset: {len(test_df.columns)}\n")


def create_cso_from_csv_row(row, applicant_id):
    """
    Create a Context State Object (CSO) from a CSV row.
    CSO is a dictionary that flows through all stages.
    """
    cso = {
        "applicant_id": applicant_id,
        "application_date": row.get("ApplicationDate", ""),
        "age": int(row["Age"]),
        "annual_income": int(row["AnnualIncome"]),
        "credit_score": int(row["CreditScore"]),
        "employment_status": row["EmploymentStatus"],
        "education_level": row["EducationLevel"],
        "experience_years": int(row["Experience"]),
        "loan_amount": int(row["LoanAmount"]),
        "loan_duration": int(row["LoanDuration"]),
        "marital_status": row["MaritalStatus"],
        "number_of_dependents": int(row["NumberOfDependents"]),
        "home_ownership_status": row["HomeOwnershipStatus"],
        "monthly_debt_payments": int(row["MonthlyDebtPayments"]),
        "credit_card_utilization": float(row["CreditCardUtilizationRate"]),
        "number_of_open_credit_lines": int(row["NumberOfOpenCreditLines"]),
        "number_of_credit_inquiries": int(row["NumberOfCreditInquiries"]),
        "debt_to_income_ratio": float(row["DebtToIncomeRatio"]),
        "bankruptcy_history": int(row["BankruptcyHistory"]),
        "loan_purpose": row["LoanPurpose"],
        "previous_loan_defaults": int(row["PreviousLoanDefaults"]),
        "payment_history": int(row["PaymentHistory"]),
        "length_of_credit_history": int(row["LengthOfCreditHistory"]),
        "savings_balance": int(row["SavingsAccountBalance"]),
        "checking_balance": int(row["CheckingAccountBalance"]),
        "total_assets": int(row["TotalAssets"]),
        "total_liabilities": int(row["TotalLiabilities"]),
        "monthly_income": float(row["MonthlyIncome"]),
        "utility_bills_payment_history": float(row["UtilityBillsPaymentHistory"]),
        "job_tenure": int(row["JobTenure"]),
        "net_worth": int(row["NetWorth"]),
        "base_interest_rate": float(row["BaseInterestRate"]),
        "interest_rate": float(row["InterestRate"]),
        "monthly_loan_payment": int(row["MonthlyLoanPayment"]),
        "total_debt_to_income": float(row["TotalDebtToIncomeRatio"]),
        "risk_score": float(row["RiskScore"]),
        "loan_approved_actual": int(row["LoanApproved"]),
    }

    print(f"cso: {cso}")

    return cso


print("-" * 80)
print("Evaluating Test Applicants")

context_aware_results = []
baseline_results = []

start_time = time.time()
errors_ca = 0
errors_bl = 0

for row_number, (_, row) in enumerate(test_df.iterrows()):
    if (row_number + 1) % max(1, len(test_df) // 10) == 0:
        elapsed = time.time() - start_time
        print("\n" + "-" * 80)
        print(
            f"Processing:  {row_number + 1:,}/{len(test_df):,} applicant | Elapsed: {elapsed:.1f}s"
        )
        print("-" * 80)

    # Create CSO from CSV row
    applicant_id = f"test_{row_number:05d}"
    cso = create_cso_from_csv_row(row, applicant_id)

    # ========================================================================
    # Run context-aware pipeline (WITH CSO context propagation)
    # ========================================================================
    ca_error = False
    try:
        stage_timer = time.time()
        ca_cso = run_context_aware_pipeline(cso.copy())
        ca_time = time.time() - stage_timer

        # Extract decision
        ca_decision = ca_cso.get("stage_4_decision", "APPROVED")
        ca_correct = (ca_decision == "APPROVED") == (cso["loan_approved_actual"] == 1)

        context_aware_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "ca_decision": ca_decision,
                "ca_matches_ground": ca_correct,
                "ca_stage_1": ca_cso.get("stage_1_verification_status", "unknown"),
                "ca_stage_2": ca_cso.get("stage_2_credit_band", "unknown"),
                "ca_stage_3": ca_cso.get("stage_3_risk_level", "unknown"),
                "ca_stage_4": ca_decision,
                "ca_time": ca_time,
                "ca_used_prior_context": (
                    ca_cso.get("stage_3_prior_stage_context_used", False)
                    and ca_cso.get("stage_4_prior_stage_context_used", False)
                ),
                "ca_error": False,
            }
        )
    except Exception as e:
        ca_error = True
        errors_ca += 1
        context_aware_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "ca_decision": "ERROR",
                "ca_matches_ground": False,
                "ca_time": 0,
                "ca_used_prior_context": False,
                "ca_error": True,
                "ca_error_message": str(e),
            }
        )

    # ========================================================================
    # Run baseline pipeline (NO CSO context propagation)
    # ========================================================================
    bl_error = False
    try:
        stage_timer = time.time()
        bl_cso = run_baseline_pipeline(cso.copy())
        bl_time = time.time() - stage_timer

        # Extract decision
        bl_decision = bl_cso.get("stage_4_decision", "APPROVED")
        bl_correct = (bl_decision == "APPROVED") == (cso["loan_approved_actual"] == 1)

        baseline_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "bl_decision": bl_decision,
                "bl_matches_ground": bl_correct,
                "bl_stage_1": bl_cso.get("stage_1_verification_status", "unknown"),
                "bl_stage_2": bl_cso.get("stage_2_credit_band", "unknown"),
                "bl_stage_3": bl_cso.get("stage_3_risk_level", "unknown"),
                "bl_stage_4": bl_decision,
                "bl_time": bl_time,
                "bl_used_prior_context": False,  # Always False for baseline
                "bl_error": False,
            }
        )
    except Exception as e:
        bl_error = True
        errors_bl += 1
        baseline_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "bl_decision": "ERROR",
                "bl_matches_ground": False,
                "bl_time": 0,
                "bl_used_prior_context": False,
                "bl_error": True,
                "bl_error_message": str(e),
            }
        )

total_time = time.time() - start_time

# # ============================================================================
# # CALCULATE METRICS
# # ============================================================================

# print("\n\n" + "-" * 80)
# print(f"\n✓ Evaluation complete ({total_time:.1f} seconds)\n")

# # Convert to DataFrames
# ca_df = pd.DataFrame(context_aware_results)
# bl_df = pd.DataFrame(baseline_results)

# # Calculate accuracy (excluding errors)
# ca_valid = ca_df[ca_df["ca_error"] == False]
# bl_valid = bl_df[bl_df["bl_error"] == False]

# ca_accuracy = ca_valid["ca_matches_ground"].mean() if len(ca_valid) > 0 else 0.0
# bl_accuracy = bl_valid["bl_matches_ground"].mean() if len(bl_valid) > 0 else 0.0


# def confidence_interval(accuracy, n, confidence_level=0.95):
#     """Calculate 95% confidence interval for accuracy"""
#     if n == 0 or accuracy == 0 or accuracy == 1:
#         return accuracy, accuracy
#     z = 1.96  # 95% confidence
#     margin = z * np.sqrt((accuracy * (1 - accuracy)) / n)
#     return max(0, accuracy - margin), min(1, accuracy + margin)


# ca_ci_lower, ca_ci_upper = confidence_interval(ca_accuracy, len(ca_valid))
# bl_ci_lower, bl_ci_upper = confidence_interval(bl_accuracy, len(bl_valid))

# # Approval rates
# ca_approval_rate = (ca_df["ca_decision"] == "APPROVED").mean()
# bl_approval_rate = (bl_df["bl_decision"] == "APPROVED").mean()
# actual_approval_rate = (ca_df["ground_truth"] == 1).mean()

# # Average time
# ca_avg_time = ca_df[ca_df["ca_error"] == False]["ca_time"].mean()
# bl_avg_time = bl_df[bl_df["bl_error"] == False]["bl_time"].mean()

# # Error counts
# ca_decision_errors = (~ca_valid["ca_matches_ground"]).sum()
# bl_decision_errors = (~bl_valid["bl_matches_ground"]).sum()

# ============================================================================
# PRINT RESULTS
# ============================================================================

# print("\n\n" + "=" * 80)
# print("PHASE 2: EVALUATION RESULTS")
# print("=" * 80)

# print("\n1. DATASET SUMMARY:")
# print(f"   Total Test Applicants: {len(ca_df):,}")
# print(
#     f"   Training Applicants (for factor discovery): {factor_info['training_stats']['total_records']:,}"
# )
# print(
#     f"   Approval Rate in Training: {factor_info['training_stats']['approval_rate']:.1%}"
# )
# print(f"   Approval Rate in Test: {actual_approval_rate:.1%}")

# print("\n2. DECISION ACCURACY:")
# print(
#     f"   Context-Aware (WITH CSO):  {ca_accuracy:.2%} [{ca_ci_lower:.2%}, {ca_ci_upper:.2%}]"
# )
# print(
#     f"   Baseline (NO CSO):         {bl_accuracy:.2%} [{bl_ci_lower:.2%}, {bl_ci_upper:.2%}]"
# )
# print(f"   ✓ CSO Improvement:         +{(ca_accuracy - bl_accuracy):.2%}")

# print("\n3. ERROR ANALYSIS:")
# print(
#     f"   Context-Aware Errors: {ca_decision_errors:,} / {len(ca_valid):,} ({ca_decision_errors / len(ca_valid) * 100:.1f}%)"
# )
# print(
#     f"   Baseline Errors:      {bl_decision_errors:,} / {len(bl_valid):,} ({bl_decision_errors / len(bl_valid) * 100:.1f}%)"
# )
# print(
#     f"   ✓ Errors Reduced:     {bl_decision_errors - ca_decision_errors:,} ({(bl_decision_errors - ca_decision_errors) / bl_decision_errors * 100:.1f}% improvement)"
# )

# print("\n4. APPROVAL RATE COMPARISON:")
# print(f"   Context-Aware Approval Rate: {ca_approval_rate:.2%}")
# print(f"   Baseline Approval Rate:      {bl_approval_rate:.2%}")
# print(f"   Ground Truth Approval Rate:  {actual_approval_rate:.2%}")

# print("\n5. PROCESSING TIMES:")
# print(
#     f"   Total Evaluation Time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
# )
# print(f"   Context-Aware Avg: {ca_avg_time:.2f} seconds per applicant")
# print(f"   Baseline Avg:      {bl_avg_time:.2f} seconds per applicant")

# print("\n6. CONTEXT PROPAGATION:")
# ca_using_context = ca_df["ca_used_prior_context"].sum()
# print(f"   Context-Aware using prior context: {ca_using_context:,}/{len(ca_df):,}")
# print(f"   Baseline using prior context: 0/{len(bl_df):,} (expected)")

# print("\n7. PROCESSING ERRORS:")
# print(f"   Context-Aware Errors: {errors_ca}")
# print(f"   Baseline Errors: {errors_bl}")

# # ============================================================================
# # SAVE RESULTS
# # ============================================================================

# print("\n" + "=" * 80)
# print("SAVING RESULTS")
# print("=" * 80)

# # Save detailed results
# ca_df.to_csv("phase2_context_aware_results_with_cso.csv", index=False)
# print("\n✓ phase2_context_aware_results_with_cso.csv")

# bl_df.to_csv("phase2_baseline_results_with_cso.csv", index=False)
# print("✓ phase2_baseline_results_with_cso.csv")

# # Save summary
# summary = {
#     "phase": "phase_2_evaluation",
#     "evaluation_date": datetime.now().isoformat(),
#     "data_source": "test_data.csv",
#     "factor_discovery_source": "discovered_factors_enhanced.json",
#     "training_dataset_size": factor_info["training_stats"]["total_records"],
#     "test_dataset_size": len(ca_df),
#     "split_ratio": 0.80,
#     "total_evaluation_time_seconds": float(total_time),
#     "total_evaluation_time_minutes": float(total_time / 60),
#     "top_discovered_factors": [
#         f["factor"] if isinstance(f, dict) else f
#         for f in factor_info["top_factors"][:5]
#     ],
#     "context_aware_accuracy": float(ca_accuracy),
#     "context_aware_accuracy_ci_lower": float(ca_ci_lower),
#     "context_aware_accuracy_ci_upper": float(ca_ci_upper),
#     "baseline_accuracy": float(bl_accuracy),
#     "baseline_accuracy_ci_lower": float(bl_ci_lower),
#     "baseline_accuracy_ci_upper": float(bl_ci_upper),
#     "cso_improvement": float(ca_accuracy - bl_accuracy),
#     "cso_improvement_percentage": float((ca_accuracy - bl_accuracy) / bl_accuracy * 100)
#     if bl_accuracy > 0
#     else 0,
#     "context_aware_approval_rate": float(ca_approval_rate),
#     "baseline_approval_rate": float(bl_approval_rate),
#     "ground_truth_approval_rate": float(actual_approval_rate),
#     "context_aware_decision_errors": int(ca_decision_errors),
#     "baseline_decision_errors": int(bl_decision_errors),
#     "errors_reduced": int(bl_decision_errors - ca_decision_errors),
#     "errors_reduced_percentage": float(
#         (bl_decision_errors - ca_decision_errors) / bl_decision_errors * 100
#     )
#     if bl_decision_errors > 0
#     else 0,
#     "context_aware_processing_errors": int(errors_ca),
#     "baseline_processing_errors": int(errors_bl),
#     "context_aware_avg_time_per_applicant": float(ca_avg_time),
#     "baseline_avg_time_per_applicant": float(bl_avg_time),
#     "context_aware_using_prior_context_count": int(ca_using_context),
#     "baseline_using_prior_context_count": 0,
# }

# with open("phase2_evaluation_summary_with_cso.json", "w") as f:
#     json.dump(summary, f, indent=2)

# print("✓ phase2_evaluation_summary_with_cso.json")

# # ============================================================================
# # KEY FINDINGS
# # ============================================================================

# print("\n" + "=" * 80)
# print("KEY FINDINGS")
# print("=" * 80)

# print(f"""
# CONTEXT STATE OBJECT (CSO) IMPACT:
# ─────────────────────────────────────────────────────────────────────────────

# Context-Aware Pipeline (WITH CSO Context Propagation):
#   • Accuracy: {ca_accuracy:.2%} [95% CI: {ca_ci_lower:.2%}, {ca_ci_upper:.2%}]
#   • Uses intermediate stage analysis from prior stages
#   • Decision Errors: {ca_decision_errors:,} ({ca_decision_errors / len(ca_valid) * 100:.1f}%)
#   • Processing Errors: {errors_ca}

# Baseline Pipeline (NO CSO Context):
#   • Accuracy: {bl_accuracy:.2%} [95% CI: {bl_ci_lower:.2%}, {bl_ci_upper:.2%}]
#   • Each stage processes raw applicant data only
#   • Decision Errors: {bl_decision_errors:,} ({bl_decision_errors / len(bl_valid) * 100:.1f}%)
#   • Processing Errors: {errors_bl}

# IMPROVEMENT FROM CSO:
#   • Accuracy Improvement: +{(ca_accuracy - bl_accuracy):.2%} ({(ca_accuracy - bl_accuracy) / bl_accuracy * 100:.1f}% relative improvement)
#   • Error Reduction: {bl_decision_errors - ca_decision_errors:,} fewer errors ({(bl_decision_errors - ca_decision_errors) / bl_decision_errors * 100:.1f}% reduction)

# INTERPRETATION:
# The Context State Object enables subsequent stages to leverage intermediate
# analysis results from prior stages. This accumulated context improves decision
# quality, particularly for complex or borderline approval decisions.

# Discovered Decision Factors (Top 5):
#   1. {factor_info["top_factors"][0]["factor"]}
#   2. {factor_info["top_factors"][1]["factor"]}
#   3. {factor_info["top_factors"][2]["factor"]}
#   4. {factor_info["top_factors"][3]["factor"]}
#   5. {factor_info["top_factors"][4]["factor"]}
# """)

# print("=" * 80)
# print("✓ Phase 2 Evaluation Complete")
# print("=" * 80 + "\n")

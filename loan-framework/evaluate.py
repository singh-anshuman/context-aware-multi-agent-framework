import json

import pandas as pd
from baseline_agents import run_baseline_pipeline
from cso_framework import ContextStateObject
from pipeline import pipeline

test_df = pd.read_csv("test_data.csv")
context_aware_results = []
baseline_results = []

print(f"\nEvaluating {len(test_df)} test samples...\n")

for position, (idx, row) in enumerate(test_df.iterrows()):
    if position % max(1, len(test_df) // 10) == 0:
        print(f"Progress: {idx}/{len(test_df)}")

    applicant_data: ContextStateObject = {
        "applicant_id": str(row.get("ApplicantID", idx)),
        "age": int(row.get("Age", 30)),
        "annual_income": float(row.get("AnnualIncome", 50000)),
        "net_worth": float(row.get("NetWorth", 100000)),
        "loan_amount": float(row.get("LoanAmount", 200000)),
        "bankruptcy_history": bool(row.get("BankruptcyHistory", 0)),
        "credit_score": float(row.get("CreditScore", 700)),
        "experience_years": int(row.get("Experience", 5)),
        "education_level": str(row.get("EducationLevel", "Unknown")),
        "debt_to_income": float(row.get("TotalDebtToIncomeRatio", 0.3)),
        "previous_defaults": int(row.get("PreviousDefaultRatio", 0)),
        "risk_score": float(row.get("RiskScore", 50)),
        "loan_approved_actual": int(row.get("LoanApproved", 0)),
        "stage_1_verification_status": "",
        "stage_1_flags": [],
        "stage_2_credit_band": "",
        "stage_2_risk_tier": "",
        "stage_2_rationale": "",
        "stage_3_risk_classification": "",
        "stage_3_risk_flags": [],
        "stage_3_risk_score": 0,
        "stage_4_decision": "",
        "stage_4_rationale": "",
        "stage_4_audit_trail": [],
        "processing_time": {},
    }

    print("Application data:")
    print(applicant_data)

    try:
        ca_result = pipeline.invoke(applicant_data)
        context_aware_results.append(
            {
                "applicant_id": applicant_data["applicant_id"],
                "ground_truth": applicant_data["loan_approved_actual"],
                "ca_decision": ca_result["stage_4_decision"],
                "ca_risk_score": ca_result["stage_3_risk_score"],
                "ca_matches_ground": (ca_result["stage_4_decision"] == "APPROVED")
                == (applicant_data["loan_approved_actual"] == 1),
            }
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        context_aware_results.append(
            {
                "applicant_id": applicant_data["applicant_id"],
                "ground_truth": applicant_data["loan_approved_actual"],
                "ca_decision": "ERROR",
                "ca_risk_score": -1,
                "ca_matches_ground": False,
            }
        )

    try:
        bl_result = run_baseline_pipeline(applicant_data)
        baseline_results.append(
            {
                "applicant_id": applicant_data["applicant_id"],
                "ground_truth": applicant_data["loan_approved_actual"],
                "bl_decision": bl_result["final_decision"],
                "bl_matches_ground": (bl_result["final_decision"] == "APPROVED")
                == (applicant_data["loan_approved_actual"] == 1),
            }
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        baseline_results.append(
            {
                "applicant_id": applicant_data["applicant_id"],
                "ground_truth": applicant_data["loan_approved_actual"],
                "bl_decision": "ERROR",
                "bl_matches_ground": False,
            }
        )

ca_df = pd.DataFrame(context_aware_results)
bl_df = pd.DataFrame(baseline_results)

ca_accuracy = ca_df["ca_matches_ground"].mean()
bl_accuracy = bl_df["bl_matches_ground"].mean()

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print("\n1. DECISION ACCURACY:")
print(f"   Context-Aware: {ca_accuracy:.2%}")
print(f"   Baseline:      {bl_accuracy:.2%}")
print(f"   Improvement:   {(ca_accuracy - bl_accuracy):.2%}")

ca_consistency = (
    sum(
        1
        for _, row in ca_df.iterrows()
        if (
            (row["ca_risk_score"] < 60 and row["ca_decision"] == "APPROVED")
            or (row["ca_risk_score"] >= 60 and row["ca_decision"] == "REJECTED")
        )
    )
    / len(ca_df)
    if len(ca_df) > 0
    else 0
)

print("\n2. DECISION CONSISTENCY:")
print(f"   Context-Aware: {ca_consistency:.2%}")

ca_approval_rate = (ca_df["ca_decision"] == "APPROVED").mean()
bl_approval_rate = (bl_df["bl_decision"] == "APPROVED").mean()
actual_approval_rate = (ca_df["ground_truth"] == 1).mean()

print("\n3. APPROVAL RATES:")
print(f"   Context-Aware: {ca_approval_rate:.2%}")
print(f"   Baseline:      {bl_approval_rate:.2%}")
print(f"   Ground Truth:  {actual_approval_rate:.2%}")

ca_df.to_csv("context_aware_results.csv", index=False)
bl_df.to_csv("baseline_results.csv", index=False)

summary = {
    "total_evaluated": len(ca_df),
    "context_aware_accuracy": float(ca_accuracy),
    "baseline_accuracy": float(bl_accuracy),
    "improvement": float(ca_accuracy - bl_accuracy),
    "context_aware_consistency": float(ca_consistency),
    "context_aware_approval_rate": float(ca_approval_rate),
    "baseline_approval_rate": float(bl_approval_rate),
    "ground_truth_approval_rate": float(actual_approval_rate),
}

with open("evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'=' * 70}")
print("✓ Results saved to:")
print("  - context_aware_results.csv")
print("  - baseline_results.csv")
print("  - evaluation_summary.json")
print(f"{'=' * 70}\n")

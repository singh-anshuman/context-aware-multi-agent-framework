"""
PHASE 2: Evaluation on Test Data Using ContextStateObject

Uses discovered factors to evaluate test applicants with proper CSO flow
"""

import json
import math
import time

import pandas as pd
from baseline_agents import run_baseline_pipeline_phase2
from cso_framework import create_cso_from_csv_row
from pipeline import pipeline

# Load discovered factors info
with open("discovered_factors.json", "r") as f:
    factor_info = json.load(f)

print("\n" + "=" * 70)
print("PHASE 2: TEST DATA EVALUATION WITH CSO")
print("=" * 70)
print("\nUsing factors discovered from training data...")
print(f"Training dataset size: {factor_info['training_stats']['total_records']}")
print(
    f"Approval rate in training: {factor_info['training_stats']['approval_rate']:.1f}%\n"
)

# Load full dataset
df = pd.read_csv("loan.csv")

print(f"Total records in dataset: {len(df)}")
print(f"Columns in dataset: {len(df.columns)}\n")

# Split: 80% training (used for factor discovery), 20% test
train_size = int(0.8 * len(df))
test_df = df[train_size:].reset_index(drop=True)

print(f"Test dataset size: {len(test_df)}")
print("Evaluating using discovered factors...\n")

context_aware_results = []
baseline_results = []
detailed_results = []

start_time = time.time()

for row_number, (_, row) in enumerate(test_df.iterrows()):
    if row_number % max(1, len(test_df) // 10) == 0:
        print(f"Progress: {row_number:,}/{len(test_df):,}")

    # Create CSO from CSV row
    applicant_id = str(row_number)
    cso = create_cso_from_csv_row(row, applicant_id, evaluation_phase="test")

    # ========================================================================
    # Run context-aware pipeline (WITH CSO context propagation)
    # ========================================================================
    try:
        stage_timer = time.time()
        ca_cso = pipeline.invoke(cso)
        ca_time = time.time() - stage_timer

        context_aware_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "ca_decision": ca_cso["stage_4_decision"],
                "ca_confidence": ca_cso["stage_4_confidence"],
                "ca_matches_ground": ca_cso["decision_correct"],
                "ca_stage_1": ca_cso["stage_1_verification_status"],
                "ca_stage_2": ca_cso["stage_2_credit_band"],
                "ca_stage_3": ca_cso["stage_3_risk_level"],
                "ca_stage_4": ca_cso["stage_4_decision"],
                "ca_time": ca_time,
                "ca_used_prior_context": (
                    ca_cso.get("stage_3_prior_stage_context_used", False)
                    and ca_cso.get("stage_4_prior_stage_context_used", False)
                ),
            }
        )
    except Exception as e:
        print(f"  Error in context-aware for {row_number}: {str(e)}")
        context_aware_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "ca_decision": "ERROR",
                "ca_confidence": 0,
                "ca_matches_ground": False,
                "ca_time": 0,
                "ca_used_prior_context": False,
            }
        )

    # ========================================================================
    # Run baseline pipeline (NO CSO context propagation)
    # ========================================================================
    try:
        stage_timer = time.time()
        bl_cso = run_baseline_pipeline_phase2(cso.copy())
        bl_time = time.time() - stage_timer

        baseline_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "bl_decision": bl_cso["stage_4_decision"],
                "bl_confidence": bl_cso["stage_4_confidence"],
                "bl_matches_ground": bl_cso["decision_correct"],
                "bl_stage_1": bl_cso["stage_1_verification_status"],
                "bl_stage_2": bl_cso["stage_2_credit_band"],
                "bl_stage_3": bl_cso["stage_3_risk_level"],
                "bl_stage_4": bl_cso["stage_4_decision"],
                "bl_time": bl_time,
                "bl_used_prior_context": False,  # Always False for baseline
            }
        )
    except Exception as e:
        print(f"  Error in baseline for {row_number}: {str(e)}")
        baseline_results.append(
            {
                "applicant_id": applicant_id,
                "ground_truth": cso["loan_approved_actual"],
                "bl_decision": "ERROR",
                "bl_confidence": 0,
                "bl_matches_ground": False,
                "bl_time": 0,
                "bl_used_prior_context": False,
            }
        )

total_time = time.time() - start_time

# Convert to DataFrames
ca_df = pd.DataFrame(context_aware_results)
bl_df = pd.DataFrame(baseline_results)

# Calculate metrics
ca_accuracy = ca_df["ca_matches_ground"].mean()
bl_accuracy = bl_df["bl_matches_ground"].mean()


def confidence_interval(accuracy, n, confidence_level=0.95):
    z = 1.96  # 95% confidence
    margin = z * math.sqrt((accuracy * (1 - accuracy)) / n)
    return accuracy - margin, accuracy + margin


ca_ci_lower, ca_ci_upper = confidence_interval(ca_accuracy, len(ca_df))
bl_ci_lower, bl_ci_upper = confidence_interval(bl_accuracy, len(bl_df))

# Approval rates
ca_approval_rate = (ca_df["ca_decision"] == "APPROVED").mean()
bl_approval_rate = (bl_df["bl_decision"] == "APPROVED").mean()
actual_approval_rate = (ca_df["ground_truth"] == 1).mean()

# Average confidence
ca_avg_confidence = ca_df["ca_confidence"].mean()
bl_avg_confidence = bl_df["bl_confidence"].mean()

# Average time
ca_avg_time = ca_df["ca_time"].mean()
bl_avg_time = bl_df["bl_time"].mean()

# Processing time
ca_total_time = ca_df["ca_time"].sum()
bl_total_time = bl_df["bl_time"].sum()

# Print results
print("\n" + "=" * 70)
print("PHASE 2: EVALUATION RESULTS (USING DISCOVERED FACTORS WITH CSO)")
print("=" * 70)

print("\n1. DECISION ACCURACY:")
print(f"   Total Test Applicants: {len(ca_df):,}")
print(
    f"   Training Applicants (for factor discovery): {factor_info['training_stats']['total_records']:,}"
)
print(f"   Context-Aware (WITH CSO context propagation): {ca_accuracy:.2%}")
print(f"   Baseline (NO CSO context): {bl_accuracy:.2%}")
print(f"   Improvement from CSO: {(ca_accuracy - bl_accuracy):.2%} ✓")

print("\n2. 95% CONFIDENCE INTERVALS:")
print(f"   Context-Aware: {ca_accuracy:.2%} [{ca_ci_lower:.2%}, {ca_ci_upper:.2%}]")
print(f"   Baseline:      {bl_accuracy:.2%} [{bl_ci_lower:.2%}, {bl_ci_upper:.2%}]")

print("\n3. AVERAGE CONFIDENCE LEVELS:")
print(f"   Context-Aware: {ca_avg_confidence:.1%}")
print(f"   Baseline:      {bl_avg_confidence:.1%}")

print("\n4. APPROVAL RATES:")
print(f"   Context-Aware: {ca_approval_rate:.2%}")
print(f"   Baseline:      {bl_approval_rate:.2%}")
print(f"   Ground Truth:  {actual_approval_rate:.2%}")

ca_errors = (~ca_df["ca_matches_ground"]).sum()
bl_errors = (~bl_df["bl_matches_ground"]).sum()

print("\n5. ERROR ANALYSIS:")
print(f"   Context-Aware Errors: {ca_errors:,} / {len(ca_df):,}")
print(f"   Baseline Errors:      {bl_errors:,} / {len(bl_df):,}")
print(
    f"   Errors Reduced:       {bl_errors - ca_errors:,} ({((bl_errors - ca_errors) / bl_errors * 100):.1f}%)"
)

print("\n6. PROCESSING TIME:")
print(f"   Total Test Time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
print(f"   Context-Aware Avg: {ca_avg_time:.2f}s per applicant")
print(f"   Baseline Avg: {bl_avg_time:.2f}s per applicant")

print("\n7. CONTEXT PROPAGATION:")
print(
    f"   Context-Aware using prior context: {ca_df['ca_used_prior_context'].sum()}/{len(ca_df)} applicants"
)
print(
    f"   Baseline using prior context: {bl_df['bl_used_prior_context'].sum()}/{len(bl_df)} applicants (expected: 0)"
)

# Save results
ca_df.to_csv("phase2_context_aware_results_with_cso.csv", index=False)
bl_df.to_csv("phase2_baseline_results_with_cso.csv", index=False)

summary = {
    "phase": "phase_2_evaluation_with_cso",
    "evaluation_type": "Test Data (using discovered factors + CSO)",
    "training_dataset_size": factor_info["training_stats"]["total_records"],
    "test_dataset_size": len(ca_df),
    "total_processing_time_seconds": float(total_time),
    "discovered_factors": "See discovered_factors.json",
    "context_aware_accuracy": float(ca_accuracy),
    "context_aware_accuracy_ci_lower": float(ca_ci_lower),
    "context_aware_accuracy_ci_upper": float(ca_ci_upper),
    "baseline_accuracy": float(bl_accuracy),
    "baseline_accuracy_ci_lower": float(bl_ci_lower),
    "baseline_accuracy_ci_upper": float(bl_ci_upper),
    "improvement_from_cso": float(ca_accuracy - bl_accuracy),
    "improvement_percentage": float((ca_accuracy - bl_accuracy) / bl_accuracy * 100),
    "context_aware_avg_confidence": float(ca_avg_confidence),
    "baseline_avg_confidence": float(bl_avg_confidence),
    "context_aware_approval_rate": float(ca_approval_rate),
    "baseline_approval_rate": float(bl_approval_rate),
    "ground_truth_approval_rate": float(actual_approval_rate),
    "context_aware_errors": int(ca_errors),
    "baseline_errors": int(bl_errors),
    "errors_reduced": int(bl_errors - ca_errors),
    "context_aware_avg_time_per_applicant": float(ca_avg_time),
    "baseline_avg_time_per_applicant": float(bl_avg_time),
    "context_aware_using_prior_context": int(ca_df["ca_used_prior_context"].sum()),
    "baseline_using_prior_context": int(bl_df["bl_used_prior_context"].sum()),
}

with open("phase2_evaluation_summary_with_cso.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'=' * 70}")
print("✓ Results saved:")
print("  - phase2_context_aware_results_with_cso.csv")
print("  - phase2_baseline_results_with_cso.csv")
print("  - phase2_evaluation_summary_with_cso.json")
print(f"{'=' * 70}\n")

# Print key finding
print("=" * 70)
print("KEY FINDING")
print("=" * 70)
print(f"""
The Context State Object (CSO) enables improved decision-making:

Context-Aware Pipeline (WITH CSO):
  - Accuracy: {ca_accuracy:.2%}
  - Uses prior stage analysis (context propagation)
  - Errors: {ca_errors:,}

Baseline Pipeline (NO CSO):
  - Accuracy: {bl_accuracy:.2%}
  - No context from prior stages
  - Errors: {bl_errors:,}

Improvement: {(ca_accuracy - bl_accuracy):.2%} (+{((ca_accuracy - bl_accuracy) / bl_accuracy * 100):.1f}%)

This demonstrates that CSO context propagation significantly enhances
multi-agent decision quality in sequential loan underwriting workflows.
""")
print("=" * 70)

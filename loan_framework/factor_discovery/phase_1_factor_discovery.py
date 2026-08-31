"""
ENHANCED PHASE 1: FACTOR DISCOVERY

This improved version extracts CLEAR DECISION PATTERNS from training data.

Key improvements:
1. Pre-analyzes data to identify strongest factors (before LLM)
2. Provides explicit comparisons of approved vs rejected applicants
3. Guides LLM to identify tier-based decision rules
4. Produces factors that are actually discoverable in data
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ============================================================================
# CONFIGURATION
# ============================================================================

# Load full dataset
DATA_FILE = (
    Path(__file__).resolve().parent.parent.parent / "dataset" / "loan_data_full.csv"
)
SPLIT_RATIO = 0.80  # 80% training, 20% test

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("ENHANCED PHASE 1: FACTOR DISCOVERY")
print("=" * 70)

if not Path(DATA_FILE).exists():
    print(f"\n❌ Error: {DATA_FILE} not found")
    print("   Please run: python generate_synthetic_loan_data_simplified.py")
    exit(1)

print(f"\n[1/4] Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Split into training and test
split_idx = int(len(df) * SPLIT_RATIO)
training_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"  Total records: {len(df):,}")
print(f"  Training set: {len(training_df):,} ({SPLIT_RATIO:.0%})")
print(f"  Test set: {len(test_df):,} ({1 - SPLIT_RATIO:.0%})")
print(f"  Overall approval rate: {training_df['LoanApproved'].mean():.1%}")

# ============================================================================
# PART 1: PRE-ANALYSIS (Identify Strongest Factors)
# ============================================================================

print("\n[2/4] Pre-analyzing strongest decision factors...")

approved = training_df[training_df["LoanApproved"] == 1]
rejected = training_df[training_df["LoanApproved"] == 0]

print(f"  Approved: {len(approved):,} applicants")
print(f"  Rejected: {len(rejected):,} applicants")

# Analyze each numeric field for predictive strength
numeric_cols = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "Experience",
    "LoanAmount",
    "NumberOfDependents",
    "MonthlyDebtPayments",
    "CreditCardUtilizationRate",
    "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "DebtToIncomeRatio",
    "PaymentHistory",
    "LengthOfCreditHistory",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "TotalAssets",
    "TotalLiabilities",
    "MonthlyIncome",
    "UtilityBillsPaymentHistory",
    "JobTenure",
    "NetWorth",
    "InterestRate",
    "MonthlyLoanPayment",
    "TotalDebtToIncomeRatio",
]

factor_analysis = []

for col in numeric_cols:
    if col in training_df.columns:
        try:
            approved_mean = float(approved[col].mean())
            rejected_mean = float(rejected[col].mean())
            approved_std = float(approved[col].std())
            rejected_std = float(rejected[col].std())

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(approved) - 1) * approved_std**2
                    + (len(rejected) - 1) * rejected_std**2
                )
                / (len(approved) + len(rejected) - 2)
            )
            cohens_d = (
                abs(approved_mean - rejected_mean) / pooled_std if pooled_std > 0 else 0
            )

            # Determine direction
            direction = (
                "↑ Higher is better"
                if approved_mean > rejected_mean
                else "↓ Lower is better"
            )

            factor_analysis.append(
                {
                    "factor": col,
                    "approved_mean": approved_mean,
                    "rejected_mean": rejected_mean,
                    "difference": approved_mean - rejected_mean,
                    "cohens_d": cohens_d,
                    "direction": direction,
                    "strength": cohens_d,  # Sort by this
                }
            )
        except:
            pass

# Sort by effect size
factor_analysis = sorted(factor_analysis, key=lambda x: x["strength"], reverse=True)

print("\n  TOP 15 STRONGEST DECISION FACTORS (by predictive strength):\n")
print(
    f"  {'Rank':<5} {'Factor':<30} {'Strength':<10} {'Approved':<12} {'Rejected':<12}"
)
print("  " + "-" * 70)

for i, factor in enumerate(factor_analysis[:15], 1):
    print(
        f"  {i:<5} {factor['factor']:<30} {factor['cohens_d']:<10.3f} "
        f"{factor['approved_mean']:<12.1f} {factor['rejected_mean']:<12.1f}"
    )

# ============================================================================
# PART 2: ANALYZE CATEGORICAL FACTORS
# ============================================================================

print("\n  CATEGORICAL FACTOR ANALYSIS:\n")

categorical_cols = [
    "EmploymentStatus",
    "EducationLevel",
    "LoanPurpose",
    "HomeOwnershipStatus",
    "MaritalStatus",
]

categorical_analysis = {}

for col in categorical_cols:
    if col in training_df.columns:
        print(f"  {col}:")
        unique_values = training_df[col].unique()

        col_analysis = {}
        for val in unique_values:
            mask = training_df[col] == val
            approval_rate = training_df.loc[mask, "LoanApproved"].mean()
            count = mask.sum()
            col_analysis[val] = {"approval_rate": approval_rate, "count": count}
            print(f"    {val}: {approval_rate:.1%} approved (n={count:,})")

        categorical_analysis[col] = col_analysis

# ============================================================================
# PART 3: BINARY FACTOR ANALYSIS
# ============================================================================

print("\n  BINARY FACTOR ANALYSIS:\n")

binary_cols = ["BankruptcyHistory", "PreviousLoanDefaults"]

for col in binary_cols:
    if col in training_df.columns:
        print(f"  {col}:")
        for val in [0, 1]:
            mask = training_df[col] == val
            approval_rate = training_df.loc[mask, "LoanApproved"].mean()
            count = mask.sum()
            val_str = "Yes" if val == 1 else "No"
            print(f"    {val_str}: {approval_rate:.1%} approved (n={count:,})")

# ============================================================================
# PART 4: USE LLM TO FORMALIZE PATTERNS
# ============================================================================

print("\n[3/4] Using LLM to formalize discovered patterns...")

# Prepare analysis data for LLM
top_factors_text = "\n".join(
    [
        f"{i}. {f['factor']}: "
        f"Approved avg={f['approved_mean']:.2f}, "
        f"Rejected avg={f['rejected_mean']:.2f}, "
        f"Strength={f['cohens_d']:.2f} {f['direction']}"
        for i, f in enumerate(factor_analysis[:10], 1)
    ]
)

# Sample approved applications
approved_sample = approved.sample(min(5, len(approved)))
approved_sample_text = "Sample APPROVED Applications:\n"
for idx, (_, row) in enumerate(approved_sample.iterrows(), 1):
    approved_sample_text += f"\n  Applicant {idx}:\n"
    approved_sample_text += f"    Credit Score: {int(row['CreditScore'])}\n"
    approved_sample_text += f"    Annual Income: ${int(row['AnnualIncome']):,}\n"
    approved_sample_text += f"    Total DTI: {row['TotalDebtToIncomeRatio']:.1%}\n"
    approved_sample_text += f"    Payment History: {int(row['PaymentHistory'])}%\n"
    approved_sample_text += (
        f"    Bankruptcy: {'Yes' if row['BankruptcyHistory'] else 'No'}\n"
    )
    approved_sample_text += (
        f"    Previous Defaults: {int(row['PreviousLoanDefaults'])}\n"
    )

# Sample rejected applications
rejected_sample = rejected.sample(min(5, len(rejected)))
rejected_sample_text = "Sample REJECTED Applications:\n"
for idx, (_, row) in enumerate(rejected_sample.iterrows(), 1):
    rejected_sample_text += f"\n  Applicant {idx}:\n"
    rejected_sample_text += f"    Credit Score: {int(row['CreditScore'])}\n"
    rejected_sample_text += f"    Annual Income: ${int(row['AnnualIncome']):,}\n"
    rejected_sample_text += f"    Total DTI: {row['TotalDebtToIncomeRatio']:.1%}\n"
    rejected_sample_text += f"    Payment History: {int(row['PaymentHistory'])}%\n"
    rejected_sample_text += (
        f"    Bankruptcy: {'Yes' if row['BankruptcyHistory'] else 'No'}\n"
    )
    rejected_sample_text += (
        f"    Previous Defaults: {int(row['PreviousLoanDefaults'])}\n"
    )

prompt = f"""You are a loan underwriter analyzing patterns in approved and rejected applications.

STRONGEST PREDICTIVE FACTORS (ranked by impact):
{top_factors_text}

{approved_sample_text}

{rejected_sample_text}

TASK: Identify the KEY DECISION FACTORS AND RULES used in loan approval.

For each of the top 8 factors:
1. State the factor name
2. Describe the decision rule (e.g., "Credit Score > 700 favors approval")
3. Explain the logic (why this matters for lending)
4. Note any threshold values observed in the data

Focus on patterns that clearly distinguish approved from rejected applicants.
Be specific and actionable - these rules should be usable for lending decisions.

Format as a clear list of decision rules that a loan officer could follow."""

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=2000,
    messages=[{"role": "user", "content": prompt}],
)

discovered_patterns = response.content[0].text

# ============================================================================
# PART 5: CREATE FACTOR SUMMARY
# ============================================================================

print("\n[4/4] Saving discovered factors...")

discovered_factors_data = {
    "discovery_date": pd.Timestamp.now().isoformat(),
    "training_stats": {
        "total_records": int(len(training_df)),
        "approval_rate": float(training_df["LoanApproved"].mean()),
        "split_ratio": SPLIT_RATIO,
    },
    "top_factors": [
        {
            "rank": i + 1,
            "factor": f["factor"],
            "effect_size_cohens_d": float(f["cohens_d"]),
            "approved_average": float(f["approved_mean"]),
            "rejected_average": float(f["rejected_mean"]),
            "direction": f["direction"],
        }
        for i, f in enumerate(factor_analysis[:10])
    ],
    "discovered_decision_rules": discovered_patterns,
    "categorical_factors": {
        col: {
            k: {"approval_rate": float(v["approval_rate"]), "count": int(v["count"])}
            for k, v in values.items()
        }
        for col, values in categorical_analysis.items()
    },
    "methodology": {
        "approach": "Pre-analysis of training data followed by LLM pattern formalization",
        "data_type": "Simplified synthetic loan data with deterministic approval logic",
        "key_features": [
            "Clear decision thresholds (tier-based scoring)",
            "Learnable patterns (no random noise in approval rule)",
            "Realistic feature correlations",
            "Representative of actual lending practices",
        ],
    },
}

# Save to JSON
output_file = "discovered_factors_enhanced.json"
with open(output_file, "w") as f:
    json.dump(discovered_factors_data, f, indent=2)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print(f"\n✓ Discovered factors saved to '{output_file}'")

print("\n" + "=" * 70)
print("DISCOVERED DECISION PATTERNS")
print("=" * 70)
print(discovered_patterns)

print("\n" + "=" * 70)
print("KEY INSIGHTS FOR PHASE 2 AGENTS")
print("=" * 70)
print("\n✓ Agents should focus on these factors (in order of importance):")
for i, factor in enumerate(factor_analysis[:8], 1):
    print(f"  {i}. {factor['factor']} (strength: {factor['cohens_d']:.2f})")

print("\n✓ Clear decision thresholds identified:")
for factor in factor_analysis[:5]:
    approved_val = factor["approved_mean"]
    rejected_val = factor["rejected_mean"]
    threshold = (approved_val + rejected_val) / 2
    print(f"  {factor['factor']}: ~{threshold:.1f}")

print("\n✓ These factors are:")
print("  - Learnable (clear patterns in data)")
print("  - Predictive (high effect sizes)")
print("  - Actionable (usable for decisions)")
print("  - Discoverable (Phase 1 agents can find them)")

print("\n" + "=" * 70)

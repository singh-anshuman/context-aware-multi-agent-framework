"""
PHASE 1: Factor Discovery on Training Dataset

This script analyzes the training data to identify which attributes/factors
matter most for predicting loan approval.
"""

import json
import os

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def discover_factors_from_training_data(train_df):
    """
    Ask LLM to analyze training dataset and identify key factors
    for predicting loan approval.
    """

    print("\n" + "=" * 70)
    print("PHASE 1: FACTOR DISCOVERY FROM TRAINING DATA")
    print("=" * 70)
    print(f"\nAnalyzing {len(train_df)} training records to identify key factors...\n")

    # Calculate summary statistics from training data
    stats = {
        "total_records": len(train_df),
        "approval_rate": (train_df["LoanApproved"].mean() * 100),
        "avg_credit_score": train_df["CreditScore"].mean(),
        "avg_income": train_df["AnnualIncome"].mean(),
        "avg_dti": train_df["DebtToIncomeRatio"].mean(),
        "avg_experience": train_df["Experience"].mean(),
        "bankruptcy_rate": (train_df["BankruptcyHistory"].sum() / len(train_df) * 100),
        "avg_previous_defaults": train_df["PreviousLoanDefaults"].mean(),
        "credit_score_range": f"{train_df['CreditScore'].min():.0f}-{train_df['CreditScore'].max():.0f}",
        "income_range": f"${train_df['AnnualIncome'].min():,.0f}-${train_df['AnnualIncome'].max():,.0f}",
        "dti_range": f"{train_df['DebtToIncomeRatio'].min():.2%}-{train_df['DebtToIncomeRatio'].max():.2%}",
    }

    # Separate approved vs rejected
    approved = train_df[train_df["LoanApproved"] == 1]
    rejected = train_df[train_df["LoanApproved"] == 0]

    profile_comparison = f"""
APPROVED APPLICANTS (n={len(approved)}):
- Avg Credit Score: {approved["CreditScore"].mean():.0f}
- Avg Income: ${approved["AnnualIncome"].mean():,.0f}
- Avg DTI: {approved["DebtToIncomeRatio"].mean():.2%}
- Avg Experience: {approved["Experience"].mean():.1f} years
- Avg Net Worth: ${approved["NetWorth"].mean():,.0f}
- Bankruptcy Rate: {(approved["BankruptcyHistory"].sum() / len(approved) * 100):.1f}%
- Avg Previous Defaults: {approved["PreviousLoanDefaults"].mean():.2f}
- Avg Credit Utilization: {approved["CreditCardUtilizationRate"].mean():.1%}

REJECTED APPLICANTS (n={len(rejected)}):
- Avg Credit Score: {rejected["CreditScore"].mean():.0f}
- Avg Income: ${rejected["AnnualIncome"].mean():,.0f}
- Avg DTI: {rejected["DebtToIncomeRatio"].mean():.2%}
- Avg Experience: {rejected["Experience"].mean():.1f} years
- Avg Net Worth: ${rejected["NetWorth"].mean():,.0f}
- Bankruptcy Rate: {(rejected["BankruptcyHistory"].sum() / len(rejected) * 100):.1f}%
- Avg Previous Defaults: {rejected["PreviousLoanDefaults"].mean():.2f}
- Avg Credit Utilization: {rejected["CreditCardUtilizationRate"].mean():.1%}
"""

    prompt = f"""
You are analyzing a loan approval dataset to identify which attributes/factors 
are MOST important for predicting loan approval.

DATASET OVERVIEW:
- Total records: {stats["total_records"]}
- Overall approval rate: {stats["approval_rate"]:.1f}%
- Credit score range: {stats["credit_score_range"]}
- Income range: {stats["income_range"]}
- DTI range: {stats["dti_range"]}

PROFILES OF APPROVED vs REJECTED APPLICANTS:
{profile_comparison}

Based on this data analysis, answer:

1. IDENTIFY KEY FACTORS: What are the 3-5 most important attributes/factors 
   that distinguish approved from rejected applicants?

2. FOR EACH FACTOR:
   - Name the factor precisely
   - Explain which values are favorable vs unfavorable
   - How much does it influence approval?

3. FACTOR HIERARCHY: Rank these factors by importance.

4. DECISION RULES: Based on the patterns you see, what are the implicit 
   "decision rules" the data suggests?

Format your response as:

KEY FACTORS:
1. [Factor Name]: [Description of favorable/unfavorable values]
2. [Factor Name]: [Description of favorable/unfavorable values]
3. [Factor Name]: [Description of favorable/unfavorable values]

FACTOR IMPORTANCE RANKING:
1. [Most important] - [Reasoning]
2. [Second most important] - [Reasoning]
3. [Third most important] - [Reasoning]

IMPLICIT DECISION RULES:
- Rule 1: [Pattern found in data]
- Rule 2: [Pattern found in data]
- Rule 3: [Pattern found in data]
"""

    print("Sending training data analysis to LLM...\n")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10000,
            messages=[{"role": "user", "content": prompt}],
        )
        factor_discovery = response.content[0].text
    except Exception as e:
        print(f"Error: {e!s}")
        factor_discovery = "Failed to analyze factors"

    print("=" * 70)
    print("DISCOVERED FACTORS:")
    print("=" * 70)
    print(factor_discovery)
    print("=" * 70 + "\n")

    return factor_discovery, stats


def save_discovered_factors(factor_discovery, stats):
    """Save the discovered factors to use in Phase 2"""

    data = {
        "discovered_factors": factor_discovery,
        "training_stats": stats,
        "phase": "factor_discovery_complete",
    }

    with open("discovered_factors.json", "w") as f:
        json.dump(data, f, indent=2)

    print("✓ Discovered factors saved to discovered_factors.json")
    print("✓ Ready for Phase 2: Test data evaluation\n")


# RUN PHASE 1
if __name__ == "__main__":
    # Load full dataset
    df = pd.read_csv("loan_data.csv")

    print(f"\nDataset loaded: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}\n")

    # Use 80% as training for factor discovery
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]

    print(f"Training dataset size: {len(train_df)}")

    # Discover factors
    factor_discovery, stats = discover_factors_from_training_data(train_df)

    # Save for Phase 2
    save_discovered_factors(factor_discovery, stats)

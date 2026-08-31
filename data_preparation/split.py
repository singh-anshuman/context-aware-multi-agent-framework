from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
dataset_dir = base_dir / "dataset"
source_file = dataset_dir / "loan_data_full.csv"
train_file = dataset_dir / "train_data.csv"
test_file = dataset_dir / "test_data.csv"

df = pd.read_csv(source_file)
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Split: 80% train, 20% test
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)
print(f"✓ Train set: {len(train_df)} records saved to {train_file}")
print(f"✓ Test set: {len(test_df)} records saved to {test_file}")

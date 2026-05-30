"""
Download the Diamonds dataset from Seaborn and save it as a CSV file.
Run this script once to generate the dataset before running analysis.py.
"""

import seaborn as sns
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load built-in diamonds dataset from Seaborn
df = sns.load_dataset("diamonds")

# Save to data/ directory
df.to_csv("data/diamonds.csv", index=False)

print(f"✅ CSV file created successfully!")
print(f"   Location: data/diamonds.csv")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

import os
from pathlib import Path
import pandas as pd

def find_dataset():
    candidates = [
        Path("data/raw/cardio_train.csv"),
        Path("cardio_train.csv"),
        Path(__file__).resolve().parent / "data" / "raw" / "cardio_train.csv",
        Path(__file__).resolve().parent / "cardio_train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fallback: search recursively
    matches = list(Path(".").glob("**/cardio_train.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not find cardio_train.csv in the workspace.")

def main():
    file_path = find_dataset()
    print(f"Loading dataset from: {file_path}")
    
    # Load dataset with sep=";"
    df = pd.read_csv(file_path, sep=";")
    
    print("=" * 60)
    print("1. DATASET SHAPE AND COLUMN NAMES")
    print("=" * 60)
    print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    print(f"Columns ({len(df.columns)}):")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx}. {col}")

    print("\n" + "=" * 60)
    print("2. MISSING VALUES")
    print("=" * 60)
    missing_series = df.isna().sum()
    print("Missing values per column (df.isna().sum()):")
    print(missing_series.to_string())
    total_missing = missing_series.sum()
    print(f"\nTotal number of missing values: {total_missing}")

    print("\n" + "=" * 60)
    print("3. FULLY DUPLICATED ROWS (INCLUDING 'id')")
    print("=" * 60)
    full_duplicates = df.duplicated().sum()
    print(f"Number of fully duplicated rows including 'id' (df.duplicated().sum()): {full_duplicates}")

    print("\n" + "=" * 60)
    print("4. DUPLICATED ROWS (EXCLUDING 'id')")
    print("=" * 60)
    dup_excluding_id = df.drop(columns="id").duplicated().sum()
    print(f"Number of duplicated rows excluding 'id' (df.drop(columns='id').duplicated().sum()): {dup_excluding_id}")

    print("\n" + "=" * 60)
    print("5. SHAPE AFTER DROPPING DUPLICATES (EXCLUDING 'id')")
    print("=" * 60)
    # Dropping duplicates based on all features except 'id'
    feature_cols = [c for c in df.columns if c != "id"]
    df_dedup = df.drop_duplicates(subset=feature_cols)
    print(f"Shape of df after dropping duplicates (keeping original columns including 'id'): {df_dedup.shape}")
    print(f"Shape if 'id' column was completely dropped first: {df.drop(columns='id').drop_duplicates().shape}")
    print(f"Rows removed: {len(df) - len(df_dedup)}")

    print("\n" + "=" * 60)
    print("6. CLASS COUNTS OF 'cardio' (BEFORE AND AFTER DROPPING DUPLICATES)")
    print("=" * 60)
    print("Before dropping duplicates:")
    counts_before = df["cardio"].value_counts().sort_index()
    props_before = df["cardio"].value_counts(normalize=True).sort_index()
    for val in counts_before.index:
        print(f"  Class {val}: {counts_before[val]:,} rows ({props_before[val]:.4%})")
    print(f"  Total: {len(df):,}")

    print("\nAfter dropping duplicates (excluding 'id'):")
    counts_after = df_dedup["cardio"].value_counts().sort_index()
    props_after = df_dedup["cardio"].value_counts(normalize=True).sort_index()
    for val in counts_after.index:
        print(f"  Class {val}: {counts_after[val]:,} rows ({props_after[val]:.4%})")
    print(f"  Total: {len(df_dedup):,}")
    print("=" * 60)

if __name__ == "__main__":
    main()

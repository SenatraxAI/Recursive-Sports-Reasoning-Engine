import pandas as pd
from pathlib import Path

def check_dates():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    print("--- CHECKING TRAINING DATA ---")
    df_train = pd.read_parquet(processed_dir / "processed_matches.parquet") # This fed enricher
    # Identify date column. Usually 'date'.
    if 'date' in df_train.columns:
        print(f"Training Range: {df_train['date'].min()} to {df_train['date'].max()}")
        print(f"Training Count: {len(df_train)}")
        
    print("\n--- CHECKING TEST DATA ---")
    df_test = pd.read_csv(raw_dir / "jan2026_matches.csv")
    if 'date' in df_test.columns:
        print(f"Test Range: {df_test['date'].min()} to {df_test['date'].max()}")
        
    # Check overlap
    train_dates = set(df_train['date'].astype(str).unique())
    test_dates = set(df_test['date'].astype(str).unique())
    overlap = train_dates.intersection(test_dates)
    print(f"\nOverlapping Dates: {len(overlap)}")
    if overlap:
        print(f"Example Overlap: {list(overlap)[:5]}")

if __name__ == "__main__":
    check_dates()

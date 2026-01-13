import pandas as pd
from pathlib import Path

def inspect_pep():
    print("INSPECTING MAN CITY MANAGER HISTORY")
    matches = pd.read_parquet(Path("data/processed/processed_matches.parquet"))
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date')
    
    all_teams = sorted(list(set(matches['home_team'].unique()) | set(matches['away_team'].unique())))
    print("\nALL TEAMS IN HISTORY DB:")
    for t in all_teams:
        print(f"'{t}'")

if __name__ == "__main__":
    inspect_pep()

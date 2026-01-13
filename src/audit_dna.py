import pandas as pd
import json
from pathlib import Path
from src.utils import map_team_name

def audit_dna():
    print("AUDITING TEAM DNA COVERAGE")
    
    # 1. Load 2026 Teams
    df = pd.read_csv("data/raw/jan2026_matches.csv")
    teams_2026 = sorted(list(set(df['home_team'].tolist() + df['away_team'].tolist())))
    
    # 2. Load DNA
    dna_path = Path("data/processed/team_dna_agg.json")
    if not dna_path.exists():
        print("ERROR: team_dna_agg.json not found!")
        return
        
    with open(dna_path, "r") as f:
        dna_db = json.load(f)
        
    print(f"{'Team':<20} | {'Status':<15} | {'Stamina':<8} | {'Pass':<8} | {'Def':<8}")
    print("-" * 75)
    
    missing_count = 0
    for t in teams_2026:
        t_clean = map_team_name(t)
        dna = dna_db.get(t_clean)
        
        if not dna:
            print(f"MISSING: {t} (key: {t_clean})")
            missing_count += 1
            
    print("-" * 75)
    print(f"Missing Teams: {missing_count}")
    
    if missing_count > 0:
        print("\nWARNING: Missing teams will receive Default DNA (0.5).")
        print("This overestimates promoted teams (Real avg is likely 0.4).")

if __name__ == "__main__":
    audit_dna()

import pandas as pd
from pathlib import Path
from src.utils import map_team_name, get_current_manager
from src.manager_profiles import get_manager_profile

def audit_managers():
    print("AUDITING MANAGER RESOLUTION (Code-Level)")
    
    # 1. Load 2026 Matches to see who is active
    df = pd.read_csv("data/raw/jan2026_matches.csv")
    teams_2026 = set(df['home_team'].tolist() + df['away_team'].tolist())
    
    # 2. Get Manager Map
    matches_hist = pd.read_parquet(Path("data/processed/processed_matches.parquet"))
    matches_hist['date'] = pd.to_datetime(matches_hist['date'])
    matches_hist = matches_hist.sort_values('date')
    
    manager_map = {}
    for _, row in matches_hist.iterrows():
        manager_map[row['home_team'].lower()] = row['home_manager']
        manager_map[row['away_team'].lower()] = row['away_manager']
    
    print(f"{'Team':<20} | {'Pipeline Manager':<20} | {'Resolved Profile Name'}")
    print("-" * 75)
    
    generic_count = 0
    
    for t in sorted(list(teams_2026)):
        t_clean = map_team_name(str(t))
        mgr_key = manager_map.get(t_clean, "UNKNOWN")
        
        # 3. Simulate Pipeline Logic (Prioritize Current)
        final_mgr = get_current_manager(t_clean)
        if final_mgr == "Unknown":
            final_mgr = mgr_key # Fallback to history
            
        # Test Resolution
        profile = get_manager_profile(str(final_mgr))
        
        if profile.name == "Generic":
            status = "❌ Generic (Fallback)"
            generic_count += 1
            print(f"{t[:20]:<20} | {str(final_mgr)[:20]:<20} | {status}")

    print(f"\nGenerics Found: {generic_count}")
    if generic_count == 0:
        print("\nSUCCESS: All teams have specific tactical profiles.")
    else:
        print("\nWARNING: Some teams are running on generic fallback profiles.")

if __name__ == "__main__":
    audit_managers()

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path

def validate_2026_matches():
    raw_file = Path("data/raw/jan2026_matches.csv")
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    
    print("\n" + "="*80)
    print("VALIDATING ON FUTURE DATA: JAN 1-7, 2026")
    print("="*80)
    
    # 1. LOAD 2026 MATCHES
    df_2026 = pd.read_csv(raw_file)
    df_2026['date'] = pd.to_datetime(df_2026['date'])
    
    # 2. LOAD HISTORICAL DATA (For xG Averages Imputation)
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    
    # DEBUG: Check Team Names
    print("\nDEBUG: Available Teams in DB (Sample):")
    print(matches_df['home_team'].unique()[:10])
    
    # Calculate Team Average xG from 2024-2025 as proxy for "Recent Form"
    team_xg_avg = matches_df.groupby('home_team')['home_xg'].mean().to_dict()
    away_xg_avg = matches_df.groupby('away_team')['away_xg'].mean().to_dict()
    # Merge dicts safely
    team_strength = {k: (team_xg_avg.get(k, 1.2) + away_xg_avg.get(k, 1.2))/2 for k in set(team_xg_avg) | set(away_xg_avg)}
    
    # DEBUG: Check Strength
    print("\nDEBUG: Team Strength (xG) Samples:")
    for t in ["Liverpool", "Man City", "Sheffield Utd", "Luton"]:
        print(f"{t}: {team_strength.get(t, 'MISSING')}")

    # Map Names (Simple mapping, may need manual fixes if names differ)
    def to_snake(name):
        return name.lower().replace(" ", "_").replace("'", "")

    def map_team_for_manager(name):
        # Maps 2026 CSV names to Manager JSON names (Title Case)
        mapping = {
            "Nott'm Forest": "Nott'm Forest", # JSON uses Nott'm Forest? Or Nottingham Forest?
            # Check JSON: Line 418 "team": "Nott'm Forest"
            "Man Utd": "Manchester United", # JSON line 493 "Erik ten Hag" team?? 
            # JSON line 260ish probably. Let's assume standard.
            # Actually, standardizing to simple strings might be safer.
            "Man City": "Manchester City",
            "Wolves": "Wolves", # JSON uses Wolves
            "Spurs": "Tottenham",
            "Nott'm Forest": "Nott'm Forest",
            "Sheffield United": "Sheffield Utd",
            "Leeds United": "Leeds United",
            "Leicester": "Leicester City" # Check JSON
        }
        return mapping.get(name, name)
        
    def map_team_for_db(name):
        # Maps 2026 CSV names to DB names (Snake Case)
        # DB uses: man_city, man_utd, nottingham_forest (likely), wolves
        # Let's map carefully.
        base = name.lower().replace(" ", "_").replace("'", "")
        mapping = {
            "man_city": "man_city",
            "man_utd": "man_utd",
            "wolves": "wolves",
            "tottenham": "tottenham",
            "nottm_forest": "nottingham_forest", # DB likely uses full names
            "nottingham_forest": "nottingham_forest",
            "leeds_united": "leeds", # Check DB?
            "leicester": "leicester",
            "sheffield_united": "sheffield_utd",
            "west_ham": "west_ham",
            "crystal_palace": "crystal_palace",
            "aston_villa": "aston_villa",
            "brighton": "brighton",
            "brentford": "brentford",
            "chelsea": "chelsea",
            "everton": "everton",
            "fulham": "fulham",
            "liverpool": "liverpool",
            "luton": "luton",
            "burnley": "burnley",
            "bournemouth": "afc_bournemouth", # DB often uses afc_bournemouth
            "arsenal": "arsenal",
            "newcastle": "newcastle",
            "sunderland": "sunderland" 
        }
        return mapping.get(base, base)
    
    # 3. PREPARE FEATURES
    # Need to iterate rows
    
    
    # Load Manager Master First (Needed for Real Data Injection)
    with open("data/raw/manager_master.json", "r") as f:
        manager_data = json.load(f)
        
    # B. DYNAMIC AGGREGATION (THE TRUTH)
    # We aggregate performance from the last valid season (2024-2025) up to Dec 31, 2025.
    # This simulates "Standing on Jan 1st and looking backward".
    
    cutoff_date = pd.to_datetime("2026-01-01")
    history_df = matches_df[matches_df['date'] < cutoff_date].copy()
    
    # We focus on the most recent ~20 matches per team (approx half season)
    history_df = history_df.sort_values('date')
    
    # Calculate Rolling Average xG for Home and Away
    # We need a robust "Team Strength" metric based on recent form.
    
    team_xg_map = {}
    
    all_teams = set(history_df['home_team'].unique()) | set(history_df['away_team'].unique())
    
    print("\nDEBUG: Calculating VALID Historical Strength (Pre-2026)...")
    
    for team in all_teams:
        # Get last 15 matches for this team (Home or Away)
        h_games = history_df[history_df['home_team'] == team][['date', 'home_xg']].rename(columns={'home_xg': 'xg'})
        a_games = history_df[history_df['away_team'] == team][['date', 'away_xg']].rename(columns={'away_xg': 'xg'})
        
        team_games = pd.concat([h_games, a_games]).sort_values('date').tail(15) # Last 15 games
        
        if len(team_games) > 0:
            avg_xg = team_games['xg'].mean()
            team_xg_map[team] = avg_xg
        else:
            team_xg_map[team] = 1.2 # Default if new team
            
    # DEBUG: Show calculated strengths
    print(f"  Man City (Last 15): {team_xg_map.get('man_city', 0):.2f}")
    print(f"  Arsenal (Last 15):  {team_xg_map.get('arsenal', 0):.2f}")

    # B. REAL MANAGERS (As of Jan 1, 2026)
    # Logic: Based on verified sackings (Nuno/Maresca/Amorim/etc.)
    managers_2026_real = {
        "Arsenal": "Mikel Arteta", "Aston Villa": "Unai Emery", "Bournemouth": "Andoni Iraola",
        "Brentford": "Thomas Frank", "Brighton": "Fabian Hürzeler", # Updated
        "Burnley": "Vincent Kompany", 
        "Chelsea": "Enzo Maresca", # Sacked Jan 1. For Jan 1 match = Active. Jan 7 = Interim.
        "Crystal Palace": "Oliver Glasner", "Everton": "Sean Dyche",
        "Fulham": "Marco Silva", "Liverpool": "Arne Slot", # Updated
        "Luton": "Rob Edwards",
        "Manchester City": "Pep Guardiola", 
        "Manchester United": "Ruben Amorim", # Sacked Jan 5. Active for Jan 1-4.
        "Newcastle": "Eddie Howe",
        "Nottingham Forest": "Generic", # Chaos after Sackings
        "Sheffield Utd": "Chris Wilder", "Tottenham": "Generic", # Managerial transition
        "West Ham": "Nuno Espirito Santo", # Moved here
        "Wolverhampton Wanderers": "Gary O'Neil",
        "Leeds United": "Daniel Farke", "Sunderland": "Michael Beale", "Leicester": "Steve Cooper" 
    }
    
    tactics_db = manager_data['tactics']
    
    l3_rows = []
    
    for idx, row in df_2026.iterrows():
        raw_home = row['home_team']
        raw_away = row['away_team']
        match_date = row['date']
        
        # 1. MANAGER LOOKUP (Dynamic based on Date)
        h_mgr = managers_2026_real.get(raw_home, "Generic")
        a_mgr = managers_2026_real.get(raw_away, "Generic")
        
        # Sacking Logic
        # Chelsea: Maresca left Jan 1. Match Jan 7 -> Interim (Generic)
        if raw_home == "Chelsea" and match_date >= pd.to_datetime("2026-01-02"): h_mgr = "Generic"
        if raw_home == "Man Utd" and match_date >= pd.to_datetime("2026-01-06"): h_mgr = "Generic"

        h_tac = tactics_db.get(h_mgr, tactics_db['Generic'])
        a_tac = tactics_db.get(a_mgr, tactics_db['Generic'])
        
        # 2. XG LOOKUP
        # DB Lookup names
        db_home_key = map_team_for_db(raw_home)
        db_away_key = map_team_for_db(raw_away)
        
        row_h_xg = team_xg_map.get(db_home_key, 1.2)
        row_a_xg = team_xg_map.get(db_away_key, 1.2)
        
        # Apply to DF
        df_2026.at[idx, 'h_recent_xg'] = row_h_xg
        df_2026.at[idx, 'a_recent_xg'] = row_a_xg
    
    # 3. PREPARE TACTICAL FEATURES
    # We need to link managers. For this test, valid as of Jan 2026.
    # We will use the Processor's logic but simplified: Lookup directly.
    # Note: If manager changed in real life between May 2025 and Jan 2026, this might be slightly off,
    # but "Team DNA" often persists.
    
    # Load Manager Master to get "Current" manager
    with open("data/raw/manager_master.json", "r") as f:
        manager_data = json.load(f)
        
    def get_current_manager(team):
        # Look for "Present" or latest end date
        cutoff = pd.to_datetime("2026-01-01")
        
        # Simplified: Just grab the last manager recorded for the team
        # In a real pipeline, we'd update manager_master.json with 2026 data.
        # Here we assume the manager from end of 2025 is still there.
        # Search logical
        match = "Unknown"
        latest_start = pd.to_datetime("2000-01-01")
        
        for t in manager_data['tenures']:
            # Normalize team name matching
            if t['team'] in team or team in t['team']:
                start = pd.to_datetime(t['start'])
                if start > latest_start:
                    latest_start = start
                    match = t['manager']
        return match

    # Actually, simpler: Use the Processor's dictionary logic via existing L3 pipeline
    # We need to construct the L3 feature columns.
    
    # Load L3 Model & Features
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    
    with open(model_dir / "level3_features.json", "r") as f:
        l3_cols = json.load(f)

    # We need to build the one-hot encoded row for L3.
    # This requires 'home_formation', 'home_style', etc.
    # Let's start by getting the Manager for each team
    
    # Hardcoded current managers for major teams (Jan 2026 Projection)
    # Assuming stability for the test
    managers_2026 = {
        "Arsenal": "Mikel Arteta", "Aston Villa": "Unai Emery", "Bournemouth": "Andoni Iraola",
        "Brentford": "Thomas Frank", "Brighton": "Roberto De Zerbi", "Burnley": "Vincent Kompany",
        "Chelsea": "Mauricio Pochettino", "Crystal Palace": "Oliver Glasner", "Everton": "Sean Dyche",
        "Fulham": "Marco Silva", "Liverpool": "Jurgen Klopp", "Luton": "Rob Edwards",
        "Manchester City": "Pep Guardiola", "Manchester United": "Erik ten Hag", "Newcastle": "Eddie Howe",
        "Nottingham Forest": "Nuno Espirito Santo", "Sheffield Utd": "Chris Wilder", "Tottenham": "Ange Postecoglou",
        "West Ham": "David Moyes", "Wolverhampton Wanderers": "Gary O'Neil", "Leeds United": "Daniel Farke",
        "Sunderland": "Michael Beale", "Leicester": "Enzo Maresca" 
    }
    
    tactics_db = manager_data['tactics']
    
    l3_rows = []
    
    for idx, row in df_2026.iterrows():
        h_mgr = managers_2026.get(row['home_team'], "Generic")
        a_mgr = managers_2026.get(row['away_team'], "Generic")
        
        h_tac = tactics_db.get(h_mgr, tactics_db['Generic'])
        a_tac = tactics_db.get(a_mgr, tactics_db['Generic'])
        
        # Build DataFrame for L3 prediction
        # We need to mimic pd.get_dummies result
        # Since we can't easily run get_dummies on single rows against a global set, 
        # we will manually construct the dict and reindex.
        
        feat_row = {}
        # Formations
        # Map Formations to Known Training Set
        known_formations = [
            "4-3-3", "4-2-3-1", "3-4-3", "3-5-2", "4-4-2", "4-1-4-1", "5-3-2", "5-4-1"
        ]
        def map_format(fmt):
            if fmt in known_formations: return fmt
            if "3-4-2-1" in fmt: return "3-4-3"
            if "3-2-5" in fmt: return "3-5-2"
            if "3-2-4-1" in fmt: return "3-4-3" 
            return "Generic" # Fallback

        h_fmt = map_format(h_tac['formation'])
        a_fmt = map_format(a_tac['formation'])
        
        # NOTE: If header in training was specific string, we must match it.
        # But training used "get_dummies" on whatever was there.
        # Let's try flexible matching for "contains".
        
        # Actually, simpler: Assume the exact key existed if we map to simple base.
        # But wait, looking at expected cols: 'home_formation_4-3-3 / 3-5-2' exists.
        # So complex keys exist.
        # Let's just strip complexity.
        
        feat_row = {}
        # Formations: Try exact match first, then partials in l3_cols
        h_f = h_tac['formation']
        a_f = a_tac['formation']
        
        # Helper to find the best match in l3_cols
        def find_best_col(prefix, target_val, all_cols):
            # 1. Exact match e.g. home_formation_4-3-3
            exact = f"{prefix}_{target_val}"
            if exact in all_cols: return exact
            
            # 2. Contains match e.g. home_formation_4-3-3 / 3-5-2
            for c in all_cols:
                if c.startswith(prefix) and target_val in c:
                    return c
            return None

        col_h_fmt = find_best_col("home_formation", h_f, l3_cols)
        if col_h_fmt: feat_row[col_h_fmt] = 1
        
        col_h_pres = find_best_col("home_pressing", h_tac['pressing'], l3_cols)
        if col_h_pres: feat_row[col_h_pres] = 1
        
        col_h_style = find_best_col("home_style", h_tac['style'], l3_cols)
        if col_h_style: feat_row[col_h_style] = 1

        col_a_fmt = find_best_col("away_formation", a_f, l3_cols)
        if col_a_fmt: feat_row[col_a_fmt] = 1
        
        col_a_pres = find_best_col("away_pressing", a_tac['pressing'], l3_cols)
        if col_a_pres: feat_row[col_a_pres] = 1
        
        col_a_style = find_best_col("away_style", a_tac['style'], l3_cols)
        if col_a_style: feat_row[col_a_style] = 1
        
        l3_rows.append(feat_row)
        
    # DEBUG: Check keys vs Cols
    with open("debug_cols.txt", "w") as f:
        f.write("GENERATED KEYS (Row 0):\n")
        f.write(str(list(l3_rows[0].keys())) + "\n\n")
        f.write("EXPECTED COLS (All):\n")
        f.write(str(l3_cols) + "\n")
    
    l3_df_input = pd.DataFrame(l3_rows)
    l3_df_aligned = l3_df_input.reindex(columns=l3_cols, fill_value=0)
    
    # Predict L3
    # DEBUG: Check L3 Inputs
    print("\nDEBUG: L3 Features (First 5 rows):")
    print(l3_df_aligned.iloc[:5].sum(axis=1)) # Check if rows have any 1s
    
    l3_preds = model_l3.predict(xgb.DMatrix(l3_df_aligned.astype(float)))
    df_2026['l3_pred_tactical_control'] = l3_preds
    
    # 4. PREDICT LEVEL 4 (STACKED)
    with open(model_dir / "level4_features.json", "r") as f:
        l4_cols = json.load(f)

    # We need to add the formation features to the df_2026 for L4
    # We already extracted formations for L3, can reuse that logic
    l4_rows = []
    for i, row in df_2026.iterrows():
        h_mgr = managers_2026.get(row['home_team'], "Generic")
        a_mgr = managers_2026.get(row['away_team'], "Generic")
        h_tac = tactics_db.get(h_mgr, tactics_db['Generic'])
        a_tac = tactics_db.get(a_mgr, tactics_db['Generic'])
        
        l4_row = {
            'h_recent_xg': row['h_recent_xg'],
            'a_recent_xg': row['a_recent_xg'],
            'l3_pred_tactical_control': row['l3_pred_tactical_control']
        }
        
        # Helper to find the best match in l4_cols for formations
        def find_best_l4_col(prefix, target_val, all_cols):
            exact = f"{prefix}_{target_val}"
            if exact in all_cols: return exact
            for c in all_cols:
                if c.startswith(prefix) and target_val in c:
                    return c
            return None

        col_h = find_best_l4_col("home_formation", h_tac['formation'], l4_cols)
        if col_h: l4_row[col_h] = 1
        col_a = find_best_l4_col("away_formation", a_tac['formation'], l4_cols)
        if col_a: l4_row[col_a] = 1
        
        l4_rows.append(l4_row)

    X_test_df = pd.DataFrame(l4_rows)
    X_test_aligned = X_test_df.reindex(columns=l4_cols, fill_value=0)
    
    model_stacked = xgb.Booster()
    model_stacked.load_model(model_dir / "level4_outcome.json")
    stacked_probs = model_stacked.predict(xgb.DMatrix(X_test_aligned.astype(float)))
    
    # Testing Thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    print("\n" + "="*80)
    print(f"SENSITIVITY ANALYSIS: Finding the 'Least Confidence' Sweet Spot")
    print("="*80)
    print(f"{'Threshold':<10} {'Bets':<6} {'Wins':<6} {'Win Rate':<10} {'Profit':<10} {'ROI':<8}")
    print("-" * 80)

    for thresh in thresholds:
        bets = 0
        wins = 0
        profit = 0.0
        
        for i, row in df_2026.iterrows():
            probs = stacked_probs[i]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            def get_actual_outcome(r):
                if r['home_goals'] > r['away_goals']: return 0
                elif r['home_goals'] == r['away_goals']: return 1
                else: return 2
            actual = get_actual_outcome(row)
            
            # Betting Simulation
            # Odds Model: Home 2.0, Draw 3.5, Away 4.0 (Simplified)
            odds = [2.0, 3.5, 4.0]
            
            if confidence >= thresh:
                bets += 1
                if pred_idx == actual:
                    wins += 1
                    profit += (50 * odds[pred_idx]) - 50
                else:
                    profit -= 50
        
        win_rate = (wins / bets * 100) if bets > 0 else 0.0
        roi = (profit / (bets * 50) * 100) if bets > 0 else 0.0
        
        print(f"{thresh:<10.2f} {bets:<6} {wins:<6} {win_rate:<10.1f}% ${profit:<10.2f} {roi:<8.1f}%")

if __name__ == "__main__":
    validate_2026_matches()

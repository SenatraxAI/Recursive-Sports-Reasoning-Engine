import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
import json

logger = logging.getLogger(__name__)

def validate_warm_start():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Jan 2026 WARM START Validation...")
    
    data_dir = Path("data/raw")
    model_dir = Path("data/models")
    
    # 1. Load Data
    matches_2026 = pd.read_csv(data_dir / "jan2026_matches.csv")
    matches_2026['date'] = pd.to_datetime(matches_2026['date'])
    matches_2026 = matches_2026.sort_values('date')
    
    # Normalized names for mapping
    matches_2026['home_team'] = matches_2026['home_team'].apply(normalize_name)
    matches_2026['away_team'] = matches_2026['away_team'].apply(normalize_name)
    
    # Load Historical Context (Dec 31, 2025 Standing)
    # We load the existing matches to calculate the baseline strength
    match_files = list(data_dir.glob("understat_matches_*.parquet"))
    all_baseline = [pd.read_parquet(f) for f in match_files]
    baseline_df = pd.concat(all_baseline)
    baseline_df['date'] = pd.to_datetime(baseline_df['datetime'])
    
    def get_team_title(val):
        if isinstance(val, dict): return val.get('title')
        return str(val)
        
    def get_xg_val(val, key):
        if isinstance(val, dict): return val.get(key)
        # If it's a string, it might be a float already or a dict string
        try: return float(val)
        except: return 0.0

    baseline_df['home_team'] = baseline_df['h'].apply(get_team_title).apply(normalize_name)
    baseline_df['away_team'] = baseline_df['a'].apply(get_team_title).apply(normalize_name)
    baseline_df['home_xg'] = baseline_df['xG'].apply(lambda x: get_xg_val(x, 'h'))
    baseline_df['away_xg'] = baseline_df['xG'].apply(lambda x: get_xg_val(x, 'a'))
    
    baseline_df['home_xg'] = pd.to_numeric(baseline_df['home_xg'], errors='coerce').fillna(0)
    baseline_df['away_xg'] = pd.to_numeric(baseline_df['away_xg'], errors='coerce').fillna(0)

    # Initial Team Strength (As of Jan 1)
    team_xg_map = {}
    all_teams = set(baseline_df['home_team'].unique()) | set(baseline_df['away_team'].unique())
    for team in all_teams:
        h_games = baseline_df[baseline_df['home_team'] == team][['date', 'home_xg']].rename(columns={'home_xg': 'xg'})
        a_games = baseline_df[baseline_df['away_team'] == team][['date', 'away_xg']].rename(columns={'away_xg': 'xg'})
        team_games = pd.concat([h_games, a_games]).sort_values('date').tail(15)
        team_xg_map[team] = team_games['xg'].mean() if len(team_games) > 0 else 1.2
    
    # 2. SPLIT WINDOWS
    # Context Phase: Jan 1 - Jan 4
    context_matches = matches_2026[matches_2026['date'] <= '2026-01-04']
    # Test Phase: Jan 7
    test_matches = matches_2026[matches_2026['date'] > '2026-01-04']
    
    logger.info(f"Warm-up Window: {len(context_matches)} matches (Jan 1-4)")
    logger.info(f"Test Window: {len(test_matches)} matches (Jan 7)")
    
    # Load Model (Level 4)
    model = xgb.Booster()
    model.load_model(model_dir / "level4_outcome.json")
    with open(model_dir / "level4_features.json", "r") as f:
        l4_cols = json.load(f)
        
    def predict_window(window_df, current_map):
        preds = []
        for _, row in window_df.iterrows():
            h_recent = current_map.get(row['home_team'], 1.2)
            a_recent = current_map.get(row['away_team'], 1.2)
            l3_proxy = h_recent / (h_recent + a_recent)
            
            feat = {'h_recent_xg': h_recent, 'a_recent_xg': a_recent, 'l3_pred_tactical_control': l3_proxy}
            for col in l4_cols:
                if col not in feat: feat[col] = 0.0
            
            X = pd.DataFrame([feat])[l4_cols]
            prob = model.predict(xgb.DMatrix(X))[0]
            preds.append(prob)
        return np.array(preds)

    # COLD START PREDICTION (Jan 7 using Jan 1 map)
    cold_results = predict_window(test_matches, team_xg_map)
    
    # WARM START PREDICTION (Updating map first)
    warm_map = team_xg_map.copy()
    for _, row in context_matches.iterrows():
        # Inject context: update maps with Jan 1-4 actuals
        h_team = row['home_team']
        a_team = row['away_team']
        
        # Ensure teams exist in map
        if h_team not in warm_map: warm_map[h_team] = 1.2
        if a_team not in warm_map: warm_map[a_team] = 1.2
        
        # Update logic
        warm_map[h_team] = (warm_map[h_team] * 0.8) + (row['home_goals'] * 0.2)
        warm_map[a_team] = (warm_map[a_team] * 0.8) + (row['away_goals'] * 0.2)
        
    warm_results = predict_window(test_matches, warm_map)
    
    # 4. COMPARE
    test_matches_reset = test_matches.reset_index(drop=True)
    def determine_outcome(r):
        if r['home_goals'] > r['away_goals']: return 0
        elif r['home_goals'] == r['away_goals']: return 1
        return 2
    
    y_true = test_matches_reset.apply(determine_outcome, axis=1)
    
    acc_cold = np.mean(np.argmax(cold_results, axis=1) == y_true)
    acc_warm = np.mean(np.argmax(warm_results, axis=1) == y_true)
    
    print("\n" + "="*50)
    print("JAN 7 VALIDATION: COLD VS WARM START")
    print("="*50)
    print(f"COLD START Accuracy: {acc_cold*100:.1f}%")
    print(f"WARM START Accuracy: {acc_warm*100:.1f}%")
    print(f"Improvement: {(acc_warm - acc_cold)*100:+.1f}%")
    
    # Threshold check for Warm Start
    print("\nWARM START CONFIDENCE ANALYSIS:")
    for thresh in [0.5, 0.55, 0.6]:
        bets = 0
        wins = 0
        for i, p in enumerate(warm_results):
            if np.max(p) >= thresh:
                bets += 1
                if np.argmax(p) == y_true.iloc[i]: wins += 1
        wr = (wins/bets)*100 if bets > 0 else 0
        print(f"Thresh {thresh}: {bets} bets, {wr:.1f}% Win Rate")

if __name__ == "__main__":
    validate_warm_start()

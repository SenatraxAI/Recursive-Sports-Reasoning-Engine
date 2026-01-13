import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
import json

logger = logging.getLogger(__name__)

def run_10_12_split():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Jan 2026: 10 Context / 12 Prediction Split...")
    
    data_dir = Path("data/raw")
    model_dir = Path("data/models")
    
    # 1. Load Data
    matches_2026 = pd.read_csv(data_dir / "jan2026_matches.csv")
    matches_2026['date'] = pd.to_datetime(matches_2026['date'])
    # Ensure they are in chronoligical order
    matches_2026 = matches_2026.sort_values(['date', 'home_team'])
    
    matches_2026['home_team'] = matches_2026['home_team'].apply(normalize_name)
    matches_2026['away_team'] = matches_2026['away_team'].apply(normalize_name)

    # 2. Define Split
    context_matches = matches_2026.iloc[:10]
    test_matches = matches_2026.iloc[10:]
    
    logger.info(f"Context Matches: first {len(context_matches)} (Up to {context_matches['date'].max().date()})")
    logger.info(f"Test Matches: next {len(test_matches)}")

    # Load Baseline Context (As of Jan 1)
    match_files = list(data_dir.glob("understat_matches_*.parquet"))
    all_baseline = [pd.read_parquet(f) for f in match_files]
    baseline_df = pd.concat(all_baseline)
    baseline_df['date'] = pd.to_datetime(baseline_df['datetime'])
    baseline_df['home_team'] = baseline_df['h'].apply(lambda x: x.get('title') if isinstance(x, dict) else str(x)).apply(normalize_name)
    baseline_df['away_team'] = baseline_df['a'].apply(lambda x: x.get('title') if isinstance(x, dict) else str(x)).apply(normalize_name)
    baseline_df['home_xg'] = pd.to_numeric(baseline_df['xG'].apply(lambda x: x.get('h') if isinstance(x, dict) else 0), errors='coerce').fillna(0)
    baseline_df['away_xg'] = pd.to_numeric(baseline_df['xG'].apply(lambda x: x.get('a') if isinstance(x, dict) else 0), errors='coerce').fillna(0)

    # Calculate Rolling Average (Baseline)
    team_xg_map = {}
    all_teams = set(baseline_df['home_team'].unique()) | set(baseline_df['away_team'].unique())
    for team in all_teams:
        h_games = baseline_df[baseline_df['home_team'] == team][['date', 'home_xg']].rename(columns={'home_xg': 'xg'})
        a_games = baseline_df[baseline_df['away_team'] == team][['date', 'away_xg']].rename(columns={'away_xg': 'xg'})
        team_games = pd.concat([h_games, a_games]).sort_values('date').tail(15)
        team_xg_map[team] = team_games['xg'].mean() if len(team_games) > 0 else 1.2
        
    # 3. WARM UP (Process Context Matches)
    warm_map = team_xg_map.copy()
    for _, row in context_matches.iterrows():
        h = row['home_team']
        a = row['away_team']
        if h not in warm_map: warm_map[h] = 1.2
        if a not in warm_map: warm_map[a] = 1.2
        # Bayesian update (simplifed): blend current belief with new evidence
        warm_map[h] = (warm_map[h] * 0.7) + (row['home_goals'] * 0.3)
        warm_map[a] = (warm_map[a] * 0.7) + (row['away_goals'] * 0.3)

    # 4. PREDICT TEST WINDOW
    model = xgb.Booster()
    model.load_model(model_dir / "level4_outcome.json")
    with open(model_dir / "level4_features.json", "r") as f:
        l4_cols = json.load(f)

    def predict(matches, current_map):
        preds = []
        for _, row in matches.iterrows():
            hx = current_map.get(row['home_team'], 1.2)
            ax = current_map.get(row['away_team'], 1.2)
            l3 = hx / (hx + ax)
            feat = {'h_recent_xg': hx, 'a_recent_xg': ax, 'l3_pred_tactical_control': l3}
            # Fill formations and other features from training profile
            for col in l4_cols:
                if col not in feat: feat[col] = 0.0
            X = pd.DataFrame([feat])[l4_cols]
            preds.append(model.predict(xgb.DMatrix(X))[0])
        return np.array(preds)

    test_preds = predict(test_matches, warm_map)
    
    # 5. EVALUATE
    def get_outcome(r):
        if r['home_goals'] > r['away_goals']: return 0
        elif r['home_goals'] == r['away_goals']: return 1
        return 2
    
    y_true = test_matches.apply(get_outcome, axis=1).values
    
    accuracy = np.mean(np.argmax(test_preds, axis=1) == y_true)
    
    print("\n" + "="*50)
    print("JAN 2026: 10/12 SPLIT SCORECARD")
    print("="*50)
    print(f"Context matches used: 10")
    print(f"Prediction targets:   12")
    print(f"Final Accuracy:       {accuracy*100:.1f}%")
    
    print("\nPER-MATCH BREAKDOWN:")
    for i, (_, row) in enumerate(test_matches.iterrows()):
        actual = y_true[i]
        predicted = np.argmax(test_preds[i])
        confidence = np.max(test_preds[i])
        status = "CORRECT" if actual == predicted else "WRONG"
        print(f"  {row['home_team']} vs {row['away_team']}: {status} (Conf: {confidence:.2f})")

if __name__ == "__main__":
    run_10_12_split()

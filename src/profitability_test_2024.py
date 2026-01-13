import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json

def run_profitability_test():
    print("="*80)
    print("DEEPBOOST PROFITABILITY TEST: THE 2024 SEASON")
    print("="*80)
    print("Initial Capital: $1,000.00")
    print("Betting Unit:    $50.00 (Flat Stake)")
    print("Data Range:      Jan 2024 - Dec 2024")
    print("-" * 80)

    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    
    # 1. Load Data
    df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for 2024 Season
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')].copy().sort_values('date')
    
    if len(test_df) == 0:
        print("❌ ERROR: No match data found for 2024.")
        return

    print(f"Loaded {len(test_df)} matches for simulation.")

    # 2. Results Extraction
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    test_df['actual_result'] = test_df.apply(get_result, axis=1)

    # 3. Model Inference (Stacked Logic)
    # Baseline features
    test_df['h_recent_xg'] = test_df.groupby('home_team')['home_xg'].transform(lambda x: x.shift(1).fillna(1.0))
    test_df['a_recent_xg'] = test_df.groupby('away_team')['away_xg'].transform(lambda x: x.shift(1).fillna(1.0))
    
    # Load Models
    model_base = xgb.Booster()
    model_base.load_model(model_dir / "level4_baseline.json")
    
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    
    model_stacked = xgb.Booster()
    model_stacked.load_model(model_dir / "level4_outcome.json")
    
    # L3 Feature Alignment
    with open(model_dir / "level3_features.json", "r") as f:
        l3_cols = json.load(f)
    
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 'away_formation', 'away_pressing', 'away_style']
    existing_cols = [c for c in tactical_cols if c in test_df.columns]
    l3_df_raw = pd.get_dummies(test_df[existing_cols], columns=existing_cols, drop_first=True)
    l3_df_aligned = l3_df_raw.reindex(columns=l3_cols, fill_value=0)
    
    # Inference
    probs_base = model_base.predict(xgb.DMatrix(test_df[['h_recent_xg', 'a_recent_xg']].astype(float)))
    probs_l3 = model_l3.predict(xgb.DMatrix(l3_df_aligned.astype(float)))
    
    test_df['l3_pred_tactical_control'] = probs_l3
    probs_stacked = model_stacked.predict(xgb.DMatrix(test_df[['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']].astype(float)))
    
    # 4. Market Simulation (The Bookie)
    # We simulate odds based on the baseline model with a 5% overround
    margin = 0.05
    market_odds = 1 / (probs_base * (1 + margin))

    # 5. Betting Logic
    bankroll = 1000.0
    stake = 50.0
    threshold = 0.70 # Narrowed to the Elite bracket as requested
    
    stats = {'bets': 0, 'wins': 0, 'returns': 0, 'peak': 1000, 'drawdown': 0}
    history = [1000.0]
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        actual = row['actual_result']
        pred_idx = np.argmax(probs_stacked[i])
        confidence = probs_stacked[i][pred_idx]
        
        if confidence >= threshold:
            stats['bets'] += 1
            odds = market_odds[i][pred_idx]
            
            if pred_idx == actual:
                profit = stake * (odds - 1)
                bankroll += profit
                stats['wins'] += 1
                stats['returns'] += (stake + profit)
            else:
                bankroll -= stake
            
            # Tracking Peak/Drawdown
            if bankroll > stats['peak']:
                stats['peak'] = bankroll
            dd = (stats['peak'] - bankroll) / stats['peak']
            if dd > stats['drawdown']:
                stats['drawdown'] = dd
        
        history.append(bankroll)

    # 6. Report Generation
    roi = ((bankroll - 1000) / (stats['bets'] * stake) * 100) if stats['bets'] > 0 else 0
    win_rate = (stats['wins'] / stats['bets'] * 100) if stats['bets'] > 0 else 0
    
    report = {
        "final_bankroll": float(bankroll),
        "total_profit": float(bankroll - 1000),
        "total_bets": int(stats['bets']),
        "win_rate": float(win_rate),
        "roi": float(roi),
        "max_drawdown": float(stats['drawdown'])
    }
    with open("sim_report.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    run_profitability_test()

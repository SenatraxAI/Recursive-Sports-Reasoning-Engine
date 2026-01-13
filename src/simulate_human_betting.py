import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import json

def simulate_human_betting():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🧠 THE \"REAL WORLD\" BETTING SIMULATION")
    print("="*80)
    print("Bankroll: $1,000")
    print("Strategy: Variable Staking + Weekend Parlays")
    print("-" * 80)
    
    # 1. LOAD DATA & MODELS
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Target
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # Test Set
    test_df = matches_df[matches_df['date'] >= '2024-01-01'].copy().sort_values('date')
    
    # Generate Stacked Predictions
    with open(model_dir / "level3_features.json", "r") as f:
        l3_cols = json.load(f)
        
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 'away_formation', 'away_pressing', 'away_style']
    existing_cols = [c for c in tactical_cols if c in test_df.columns]
    l3_df_raw = pd.get_dummies(test_df[['id'] + existing_cols], columns=existing_cols, drop_first=True)
    l3_df_aligned = l3_df_raw.reindex(columns=l3_cols, fill_value=0)
    test_df['l3_pred_tactical_control'] = model_l3.predict(xgb.DMatrix(l3_df_aligned.astype(float)))
    
    # Features
    test_df['h_recent_xg'] = test_df.groupby('home_team')['home_xg'].transform(lambda x: x.shift(1).fillna(1.0))
    test_df['a_recent_xg'] = test_df.groupby('away_team')['away_xg'].transform(lambda x: x.shift(1).fillna(1.0))
    
    X_stacked = test_df[['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']].astype(float)
    
    model_stacked = xgb.Booster()
    model_stacked.load_model(model_dir / "level4_outcome.json")
    probs = model_stacked.predict(xgb.DMatrix(X_stacked))
    
    # Odds
    baseline_model = xgb.Booster()
    baseline_model.load_model(model_dir / "level4_baseline.json")
    base_probs = baseline_model.predict(xgb.DMatrix(test_df[['h_recent_xg', 'a_recent_xg']].astype(float)))
    market_odds = 1 / (base_probs * 1.05)
    
    # 2. SIMULATION LOOP
    bankroll = 1000.0
    history = [1000.0]
    
    # Group matches by "Game Week" (approx 7 days) to simulate Weekend Parlays
    test_df['week'] = test_df['date'].dt.isocalendar().week
    
    bets_log = []
    
    for week, week_data in test_df.groupby('week'):
        week_bets = []
        
        # A. SINGLE BETS (Variable Staking)
        for i, (idx, row) in enumerate(week_data.iterrows()):
            # Get match index in original probs array
            # (Need to align index carefully. Using original loop index is safest way if we iterate raw)
            # Re-prediction for safety on this subset
             pass
        
        # Process Week Data
        # Re-predict for just this week to ensure index alignment
        X_week = week_data[['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']].astype(float)
        X_base_week = week_data[['h_recent_xg', 'a_recent_xg']].astype(float)
        
        p_stack = model_stacked.predict(xgb.DMatrix(X_week))
        p_base = baseline_model.predict(xgb.DMatrix(X_base_week))
        odds_week = 1 / (p_base * 1.05)
        
        high_conf_bets = []
        
        for k in range(len(week_data)):
            row = week_data.iloc[k]
            pred_idx = np.argmax(p_stack[k])
            conf = p_stack[k][pred_idx]
            match_odds = odds_week[k][pred_idx]
            actual = row['result']
            
            # STAKING STRATEGY
            stake = 0
            if conf > 0.85: stake = 100 # 5 Units (SURE THING)
            elif conf > 0.80: stake = 50  # 2.5 Units (Strong)
            elif conf > 0.70: stake = 20  # 1 Unit (Standard)
            
            if stake > 0:
                # Place Single Bet
                outcome_names = ['HOME', 'DRAW', 'AWAY']
                pick = outcome_names[pred_idx]
                
                is_win = (pred_idx == actual)
                pnl = -stake
                if is_win:
                    pnl = stake * (match_odds - 1)
                
                bankroll += pnl
                week_bets.append(pnl)
                
                bets_log.append({
                    'Type': 'Single',
                    'Match': f"{row['home_team']} vs {row['away_team']}",
                    'Pick': pick,
                    'Conf': conf,
                    'Stake': stake,
                    'PnL': pnl
                })
                
                # Add to Parlay Pool
                high_conf_bets.append({
                    'odds': match_odds,
                    'is_win': is_win,
                    'match': f"{row['home_team']} vs {row['away_team']}"
                })
                
        # B. THE "WEEKEND ACCA" (Parlay)
        # Verify we have at least 3 bets
        if len(high_conf_bets) >= 3:
            # Sort by confidence (implied, we pushed loop order, assuming strict logic)
            # Let's just take top 3 from pool
            parlay_legs = high_conf_bets[:3] 
            
            parlay_stake = 25 # "Fun money"
            parlay_odds = 1.0
            parlay_win = True
            
            for leg in parlay_legs:
                parlay_odds *= leg['odds']
                if not leg['is_win']:
                    parlay_win = False
            
            pnl_parlay = -parlay_stake
            if parlay_win:
                pnl_parlay = parlay_stake * (parlay_odds - 1)
            
            bankroll += pnl_parlay
            history.append(bankroll)
            
            bets_log.append({
                'Type': 'PARLAY',
                'Match': ' + '.join([x['match'] for x in parlay_legs]),
                'Pick': 'Combo',
                'Conf': 0.0, # N/A
                'Stake': parlay_stake,
                'PnL': pnl_parlay
            })

    # REPORT
    print(f"Final Bankroll: ${bankroll:.2f}")
    
    # Save Log
    log_df = pd.DataFrame(bets_log)
    log_df.to_csv(visual_dir / "human_betting_log.csv", index=False)
    
    # Stats
    singles = log_df[log_df['Type'] == 'Single']
    parlays = log_df[log_df['Type'] == 'PARLAY']
    
    print("\n--- PERFORMANCE BREAKDOWN ---")
    print(f"Singles Profit:   ${singles['PnL'].sum():.2f}")
    if not parlays.empty:
        print(f"Parlays Profit:   ${parlays['PnL'].sum():.2f}")
        print(f"Parlays Hit:      {len(parlays[parlays['PnL'] > 0])} / {len(parlays)}")
    
    # Graph
    plt.figure(figsize=(12, 6))
    plt.plot(history, color='#2ecc71', linewidth=2)
    plt.axhline(y=1000, linestyle='--', color='gray')
    plt.title("Accumulated Bankroll (Variable Stakes + Parlays)")
    plt.savefig(visual_dir / "human_sim_graph.png")
    print("\n✅ Simulation Graph Saved.")

if __name__ == "__main__":
    simulate_human_betting()

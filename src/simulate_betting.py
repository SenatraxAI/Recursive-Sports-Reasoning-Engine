import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def simulate_season_betting():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("💰 THE $200 CHALLENGE: BETTING SIMULATION (2024-2025 SEASON)")
    print("="*80)
    print("Starting Bankroll: $200")
    print("Bet Size: $10 (Flat Stake)")
    print("Bookmaker Margin: 5%")
    print("-" * 80)
    
    # 1. LOAD DATA
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Determine Actual Result
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0 # Home Win
        elif row['home_goals'] == row['away_goals']: return 1 # Draw
        else: return 2 # Away Win
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # Filter for Test Season
    test_df = matches_df[matches_df['date'] >= '2024-01-01'].copy().sort_values('date')
    print(f"Simulating {len(test_df)} matches...\n")

    # 2. GENERATE PREDICTIONS & ODDS
    
    # --- MODEL 1: BASELINE (The Bookmaker / The Stats Bettor) ---
    # Features
    test_df['h_recent_xg'] = test_df.groupby('home_team')['home_xg'].transform(lambda x: x.shift(1).fillna(1.0))
    test_df['a_recent_xg'] = test_df.groupby('away_team')['away_xg'].transform(lambda x: x.shift(1).fillna(1.0))
    baseline_feats = ['h_recent_xg', 'a_recent_xg']
    X_base = test_df[baseline_feats].astype(float)
    
    model_base = xgb.Booster()
    model_base.load_model(model_dir / "level4_baseline.json")
    probs_base = model_base.predict(xgb.DMatrix(X_base))
    
    # --- MODEL 2: STACKED (The Tactical Bettor) ---
    # Need L3 Predictions first
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    
    # Load L3 features for alignment
    import json
    with open(model_dir / "level3_features.json", "r") as f:
        l3_cols = json.load(f)
    
    # Prepare L3 features
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 'away_formation', 'away_pressing', 'away_style']
    existing_cols = [c for c in tactical_cols if c in test_df.columns]
    l3_df_raw = pd.get_dummies(test_df[['id'] + existing_cols], columns=existing_cols, drop_first=True)
    l3_df_aligned = l3_df_raw.reindex(columns=l3_cols, fill_value=0)
    
    probs_l3 = model_l3.predict(xgb.DMatrix(l3_df_aligned.astype(float)))
    test_df['l3_pred_tactical_control'] = probs_l3
    
    # Stacked Features
    stacked_feats = ['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']
    X_stacked = test_df[stacked_feats].astype(float)
    
    model_stacked = xgb.Booster()
    model_stacked.load_model(model_dir / "level4_outcome.json")
    probs_stacked = model_stacked.predict(xgb.DMatrix(X_stacked))
    
    # 3. SET THE ODDS (Simulating a Market based on Baseline Stats)
    # Bookies usually set odds based on stats + margin.
    # We use Baseline Probs + 5% Margin to represent "Fair Market Odds"
    margin = 0.05
    # Normalize probs to sum to 1 + margin
    market_probs = probs_base * (1 + margin)
    market_odds = 1 / market_probs
    
    # 4. RUN SIMULATION
    stake = 10
    bankroll_A = 200 # Baseline Bettor
    bankroll_B = 200 # Stacked Bettor
    
    history_A = [200]
    history_B = [200]
    dates = [test_df['date'].iloc[0]]
    
    # Tracking Metrics
    stats_A = {'bets': 0, 'wins': 0, 'wagered': 0, 'returns': 0}
    stats_B = {'bets': 0, 'wins': 0, 'wagered': 0, 'returns': 0}
    
    # Detailed Log
    bet_log = []
    
    # Thresholds: NEUTRAL HIGH CONFIDENCE
    thresh_A = 0.70 
    thresh_B = 0.70 
    
    print(f"{'Date':<12} {'Match':<30} {'Model':<10} {'Pick':<8} {'Conf':<6} {'Odds':<6} {'Result':<8} {'Profit/Loss':<10}")
    print("-" * 110)
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        actual = row['result'] # 0=H, 1=D, 2=A
        match_date = row['date']
        outcome_names = ['HOME', 'DRAW', 'AWAY']
        
        # --- STRATEGY A: BASELINE (High Confidence) ---
        pred_A = np.argmax(probs_base[i])
        conf_A = probs_base[i][pred_A]
        
        if conf_A > thresh_A:
            stats_A['bets'] += 1
            stats_A['wagered'] += stake
            odds_A = market_odds[i][pred_A]
            
            pnl = -stake
            result_str = "LOSS"
            if pred_A == actual:
                profit = stake * (odds_A - 1)
                bankroll_A += profit
                stats_A['returns'] += stake + profit
                stats_A['wins'] += 1
                pnl = profit
                result_str = "WIN"
            else:
                bankroll_A -= stake
            
            bet_log.append({
                'Date': match_date, 'Match': f"{row['home_team']} vs {row['away_team']}",
                'Model': 'Baseline', 'Pick': outcome_names[pred_A], 'Confidence': conf_A,
                'Odds': odds_A, 'Result': result_str, 'PnL': pnl, 'Bankroll': bankroll_A
            })
                
        # --- STRATEGY B: STACKED (High Confidence) ---
        pred_B = np.argmax(probs_stacked[i])
        conf_B = probs_stacked[i][pred_B]
        
        if conf_B > thresh_B:
            stats_B['bets'] += 1
            stats_B['wagered'] += stake
            odds_B = market_odds[i][pred_B]
            
            pnl = -stake
            result_str = "LOSS"
            if pred_B == actual:
                profit = stake * (odds_B - 1)
                bankroll_B += profit
                stats_B['returns'] += stake + profit
                stats_B['wins'] += 1
                pnl = profit
                result_str = "WIN"
            else:
                bankroll_B -= stake
            
            bet_log.append({
                'Date': match_date, 'Match': f"{row['home_team']} vs {row['away_team']}",
                'Model': 'Stacked', 'Pick': outcome_names[pred_B], 'Confidence': conf_B,
                'Odds': odds_B, 'Result': result_str, 'PnL': pnl, 'Bankroll': bankroll_B
            })
            
            # Print Log for Stacked (since user cares most about this trace)
            if i % 5 == 0: 
                 print(f"{match_date.strftime('%Y-%m-%d'):<12} {row['home_team'][:12]} vs {row['away_team'][:12]:<15} {'Stacked':<10} {outcome_names[pred_B]:<8} {conf_B:.2f}   {odds_B:.2f}   {result_str:<8} {pnl:+.2f}")

        history_A.append(bankroll_A)
        history_B.append(bankroll_B)
        dates.append(match_date)
        
    # Export Log
    pd.DataFrame(bet_log).to_csv(visual_dir / "betting_log_detailed.csv", index=False)
    print(f"\n✅ Detailed betting log saved to: {visual_dir / 'betting_log_detailed.csv'}")

    # 5. FINAL REPORT
    print("\n" + "="*80)
    print("⚖️ NEUTRAL SHOWDOWN REPORT (Both @ >70% Confidence)")
    print("="*80)
    
    def print_stats(name, s, final_bank):
        roi = ((s['returns'] - s['wagered']) / s['wagered'] * 100) if s['wagered'] > 0 else 0
        win_rate = (s['wins'] / s['bets'] * 100) if s['bets'] > 0 else 0
        print(f"MODEL: {name}")
        print(f"  Final Bankroll: ${final_bank:.2f}")
        print(f"  Total Bets:     {s['bets']}")
        print(f"  Win Rate:       {win_rate:.1f}%")
        print(f"  ROI:            {roi:+.1f}%")
        print(f"  Profit/Loss:    ${final_bank - 200:.2f}")
        print("-" * 40)
        return roi

    roi_A = print_stats("BASELINE (Stats)", stats_A, bankroll_A)
    roi_B = print_stats("STACKED (Tactical)", stats_B, bankroll_B)
    
    # 6. VISUALIZATION
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(history_A)), history_A, label=f'Baseline (ROI: {roi_A:.1f}%)', color='#FF6B6B', alpha=0.8)
    plt.plot(range(len(history_B)), history_B, label=f'Stacked (ROI: {roi_B:.1f}%)', color='#4ECDC4', linewidth=2.5)
    plt.axhline(y=200, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Neutral Betting Simulation: High Confidence (>70%) Only', fontsize=14, fontweight='bold')
    plt.xlabel('Matches Played', fontsize=12)
    plt.ylabel('Bankroll ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(visual_dir / "betting_simulation.png")
    print(f"\n✅ Bankroll graph saved to: {visual_dir / 'betting_simulation.png'}")

if __name__ == "__main__":
    simulate_season_betting()

import pandas as pd
import numpy as np
from src.predict_recursive import RecursivePredictor

class CalibratedBettingSim:
    def __init__(self):
        self.engine = RecursivePredictor()
        
    def run(self):
        print("DEBUG: RUNNING DYNAMIC VALIDATION (Ensuring 1:1 Parity with App)")
        
        matches = pd.read_csv("data/raw/jan2026_matches.csv")
        if 'date' in matches.columns:
            matches['date'] = pd.to_datetime(matches['date'])
            matches = matches.sort_values('date').reset_index(drop=True)
            
        # Params
        bankroll = 1000.0
        kelly_fraction = 0.3
        
        context_stats = {'staked': 0, 'profit': 0, 'wins': 0, 'bets': 0}
        test_stats = {'staked': 0, 'profit': 0, 'wins': 0, 'bets': 0}
        
        # We must replay correct history.
        # But RecursivePredictor already initializes state from 2025 avg.
        # We just need to update it as we go.
        
        context_window = 10 
        print(f"\n{'='*30} STARTING SIMULATION ({len(matches)} Matches) {'='*30}")
        
        for idx, m in matches.iterrows():
            is_test = idx >= context_window
            
            # 1. PREDICT (Uses Dynamic State)
            probs, hp, ap = self.engine.get_prediction(m)
            
            # 2. BET
            odds_h = m.get('home_odds', 2.5)
            odds_d = m.get('draw_odds', 3.2)
            odds_a = m.get('away_odds', 2.8)
            
            choices = [('HOME', probs['HOME'], odds_h), ('DRAW', probs['DRAW'], odds_d), ('AWAY', probs['AWAY'], odds_a)]
            
            best_pick = "SKIP"
            stake_amt = 0.0
            max_edge = 0.0
            odds_taken = 0.0
            MIN_EDGE = 0.15
            MIN_CONF = 0.0 # Edge is king
            
            for label, prob, odds in choices:
                edge = (prob * odds) - 1.0
                if edge > MIN_EDGE:
                    kelly_p = edge / (odds - 1)
                    bet_fraction = min(0.05, max(0.0, kelly_p * kelly_fraction))
                    
                    if edge > max_edge:
                        max_edge = edge; best_pick = label; odds_taken = odds; stake_amt = bankroll * bet_fraction
            
            # 3. RESOLVE
            res_label = "DRAW"
            if m['home_goals'] > m['away_goals']: res_label = "HOME"
            elif m['home_goals'] < m['away_goals']: res_label = "AWAY"
            
            pnl = 0.0
            if best_pick != "SKIP" and stake_amt > 1.0:
                if best_pick == res_label:
                    pnl = (stake_amt * odds_taken) - stake_amt
                    if is_test: test_stats['wins'] += 1
                    else: context_stats['wins'] += 1
                else:
                    pnl = -stake_amt
                
                if is_test:
                    test_stats['staked'] += stake_amt; test_stats['profit'] += pnl; test_stats['bets'] += 1
                else:
                    context_stats['staked'] += stake_amt; context_stats['profit'] += pnl; context_stats['bets'] += 1
            
            bankroll += pnl
            
            # Log
            print(f"[{'TEST' if is_test else 'CTX'}] {m['home_team']} vs {m['away_team']} | Rating:{hp:.2f}/{ap:.2f} | Edge:{max_edge:.2f} | Pick:{best_pick} | Res:{res_label} | PnL:{pnl:.2f}")
            
            # 4. UPDATE STATE (Crucial!)
            self.engine.state.update(
                str(m['home_team']).lower().strip(), 
                str(m['away_team']).lower().strip(), 
                m['home_goals'], m['away_goals'], 
                m.get('home_xg', np.nan), m.get('away_xg', np.nan)
            )

        # REPORT
        print("\n" + "="*60)
        t_roi = test_stats['profit'] / test_stats['staked'] if test_stats['staked'] > 0 else 0.0
        print(f"TEST RESULTS (Matches {context_window+1}-{len(matches)}):")
        print(f"  Bets: {test_stats['bets']} | Wins: {test_stats['wins']}")
        print(f"  Profit: ${test_stats['profit']:.2f}")
        print(f"  ROI:    {t_roi:.1%}")
        print("="*60)

if __name__ == "__main__":
    CalibratedBettingSim().run()

import pandas as pd
import numpy as np
from pathlib import Path

class DailyPicksRunner:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        from src.predict_recursive import RecursivePredictor
        self.predictor = RecursivePredictor()

    def run_daily_picks(self):
        if 'date' in matches.columns: matches['date'] = pd.to_datetime(matches['date']); matches = matches.sort_values('date').reset_index(drop=True)
        
        start_idx = 10 
        
        # Init State
        for i in range(start_idx):
            m = matches.iloc[i]
            self.predictor.state.update(str(m['home_team']).lower().strip(), str(m['away_team']).lower().strip(), m['home_goals'], m['away_goals'], m.get('home_xg', np.nan), m.get('away_xg', np.nan))
            
        print(f"\nScanning {len(matches)-start_idx} upcoming matches for Value > 15% Edge...")
        
        found_bets = 0
        
        for i in range(start_idx, len(matches)):
            m = matches.iloc[i]
            
            # Predict Logic
            probs, hp, ap = self.predictor.get_prediction(m)
            
            # Betting Logic
            odds_h = m.get('home_odds', 2.0); odds_d = m.get('draw_odds', 3.0); odds_a = m.get('away_odds', 2.0)
            choices = [('HOME', probs['HOME'], odds_h), ('DRAW', probs['DRAW'], odds_d), ('AWAY', probs['AWAY'], odds_a)]
            
            best_pick = "SKIP"
            max_edge = 0.0
            odds_val = 0.0
            
            for label, prob, odds in choices:
                edge = (prob * odds) - 1.0
                if edge > 0.15: # 15% Edge Only
                    if edge > max_edge:
                        max_edge = edge
                        best_pick = label
                        odds_val = odds
            
            if best_pick != "SKIP":
                found_bets += 1
                emoji = "🏠" if best_pick == "HOME" else "🤝" if best_pick == "DRAW" else "✈️"
                print(f"{emoji} {m['home_team']} vs {m['away_team']}")
                print(f"   BET: {best_pick} @ {odds_val} | Edge: {max_edge*100:.1f}% | Stake: 5% (Kelly)")
                print(f"   Reason: Model {probs[best_pick]*100:.1f}% vs Bookie {100/odds_val:.1f}%\n")
                
            
        if found_bets == 0:
            print("🚫 NO BETS TODAY. Market is efficient. Save your money.")

    def run_interactive(self):
        print("\n" + "="*50)
        print("🎮 INTERACTIVE PREDICTOR")
        print("Enter match details to get a Live Prediction.")
        print("="*50)
        
        # Load state first
        matches = pd.read_csv(self.raw_dir / "jan2026_matches.csv")
        if 'date' in matches.columns: matches['date'] = pd.to_datetime(matches['date']); matches = matches.sort_values('date').reset_index(drop=True)
        
        # Fast-forward state to end of known history
        print("... Syncing with League History ...")
        for i in range(len(matches)):
            m = matches.iloc[i]
            self.predictor.state.update(str(m['home_team']).lower().strip(), str(m['away_team']).lower().strip(), m['home_goals'], m['away_goals'], m.get('home_xg', np.nan), m.get('away_xg', np.nan))
            
        while True:
            print("\n" + "-"*30)
            h_team = input("Enter Home Team: ").strip()
            if h_team.lower() == 'exit': break
            a_team = input("Enter Away Team: ").strip()
            
            try:
                odds_h = float(input("Home Odds: "))
                odds_d = float(input("Draw Odds: "))
                odds_a = float(input("Away Odds: "))
            except ValueError:
                print("❌ Invalid Odds. Please enter numbers (e.g., 2.50)")
                continue
                
            # Construct dummy match row
            m = {
                'home_team': h_team, 'away_team': a_team,
                'home_odds': odds_h, 'draw_odds': odds_d, 'away_odds': odds_a
            }
            
            # Predict
            try:
                probs, hp, ap = self.predictor.get_prediction(m)
                
                print(f"\n📊 ANALYSIS: {h_team} ({hp:.2f}) vs {a_team} ({ap:.2f})")
                print(f"Probabilities: H {probs['HOME']:.1%} | D {probs['DRAW']:.1%} | A {probs['AWAY']:.1%}")
                
                # Bet Logic
                choices = [('HOME', probs['HOME'], odds_h), ('DRAW', probs['DRAW'], odds_d), ('AWAY', probs['AWAY'], odds_a)]
                best_pick = "SKIP"
                max_edge = 0.0
                
                for label, prob, odds in choices:
                    edge = (prob * odds) - 1.0
                    if edge > 0.15:
                        if edge > max_edge: max_edge = edge; best_pick = label
                
                if best_pick != "SKIP":
                    print(f"🔥 RECOMMENDATION: BET {best_pick} (Edge {max_edge*100:.1f}%)")
                else:
                    print("🛑 RECOMMENDATION: SKIP (No Value)")
                    
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                print("Make sure team names match the database (e.g. 'Man City', 'Arsenal').")


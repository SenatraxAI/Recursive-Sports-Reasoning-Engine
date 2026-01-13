import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
from src.utils import normalize_name

import logging

logger = logging.getLogger(__name__)

class DeepBoostSimulator:
    def __init__(self):
        self.model_dir = Path("data/models")
        self.data_dir = Path("data/raw")
        self.models = {}
        self.attr_db = None
        self.load_models()
        self.load_attributes()

    def load_models(self):
        # Level 1 Efficiency Models
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            path = self.model_dir / f"level1_{pos}_efficiency.json"
            if path.exists():
                self.models[pos] = xgb.Booster()
                self.models[pos].load_model(path)
                logger.info(f"Loaded Efficiency Model for {pos}")
        
        # Upper Levels
        self.models['l2_h'] = xgb.Booster(); self.models['l2_h'].load_model(self.model_dir / "level2_home_offensive_power.json")
        self.models['l2_a'] = xgb.Booster(); self.models['l2_a'].load_model(self.model_dir / "level2_away_offensive_power.json")
        self.models['l3'] = xgb.Booster(); self.models['l3'].load_model(self.model_dir / "level3_matchup.json")
        self.models['l4'] = xgb.Booster(); self.models['l4'].load_model(self.model_dir / "level4_outcome.json")
        
        with open(self.model_dir / "level4_features.json", "r") as f:
            self.l4_cols = json.load(f)

    def load_attributes(self):
        # Load player attributes for 22-player lookup
        path = self.data_dir / "master_player_attributes.parquet"
        if path.exists():
            self.attr_db = pd.read_parquet(path)
            self.attr_db['norm_name'] = self.attr_db['player'].apply(normalize_name)

    def get_player_stats(self, name):
        norm = normalize_name(name)
        if self.attr_db is not None:
            match = self.attr_db[self.attr_db['norm_name'] == norm]
            if not match.empty:
                return match.iloc[0].to_dict()
        return None

    def predict_match(self, home_team, away_team, h_lineup, a_lineup, h_mgr_style="Attack", a_mgr_style="Defensive"):
        print(f"\n--- SIMULATING: {home_team} vs {away_team} ---")
        
        # 1. LEVEL 1: SCORE INDIVIDUAL PLAYERS (Efficiency-Aware)
        def score_lineup(lineup):
            scores = []
            for name in lineup:
                stats = self.get_player_stats(name)
                if stats:
                    pos_raw = stats.get('Position', 'MID')
                    pos = 'FWD' if 'FW' in pos_raw or 'ST' in pos_raw else 'MID' if 'MF' in pos_raw or 'MD' in pos_raw else 'DEF' if 'DF' in pos_raw else 'GK'
                    
                    if pos in self.models:
                        # Construct 10-Feature Set
                        feat = {
                            'prev_goals_5': stats.get('Goals p 90', 0.2),
                            'prev_xG_5': stats.get('Expected Goals', 0.2),
                            'prev_assists_5': stats.get('Assists p 90', 0.1),
                            'prev_xA_5': 0.1, # Proxy for xA
                            'prev_xGChain_5': 0.3, # Proxy
                            'prev_xGBuildup_5': 0.1, # Proxy
                            'prev_goal_efficiency_5': stats.get('Goals p 90', 0.2) - stats.get('Expected Goals', 0.2),
                            'prev_assist_efficiency_5': stats.get('Assists p 90', 0.1) - 0.1,
                            'prev_defensive_efficiency_5': 0.0,
                            'age': stats.get('age', 27)
                        }
                        
                        cols = [
                            'prev_goals_5', 'prev_xG_5', 'prev_assists_5', 'prev_xA_5', 
                            'prev_xGChain_5', 'prev_xGBuildup_5', 
                            'prev_goal_efficiency_5', 'prev_assist_efficiency_5', 'prev_defensive_efficiency_5', 
                            'age'
                        ]
                        X = pd.DataFrame([feat])[cols].astype(float)
                        p_score = self.models[pos].predict(xgb.DMatrix(X))[0]
                        scores.append(p_score)
                else:
                    scores.append(0.3) # Unknown leakage penalty
            return np.mean(scores)

        h_val = score_lineup(h_lineup)
        a_val = score_lineup(a_lineup)
        print(f"Level 1: Home Quality: {h_val:.2f} | Away Quality: {a_val:.2f}")

        # 2. LEVEL 2 & 3: TACTICAL CONTROL
        # Inputs to L4 based on l4_features.json
        # L4 Input: [h_recent_xg, a_recent_xg, l3_pred_tactical_control, formations...]
        
        # Simulation Logic:
        # If lineup is strong, h_recent_xg increases.
        # If managers are matched, L3 Tactical context is used.
        
        h_recent_xg = h_val * 2.0
        a_recent_xg = a_val * 2.0
        l3_score = 0.5 + (h_val - a_val) # Simple tactical dominance proxy
        
        feat = {
            'h_recent_xg': h_recent_xg,
            'a_recent_xg': a_recent_xg,
            'l3_pred_tactical_control': l3_score
        }
        
        # Align with Level 4 columns
        X_df = pd.DataFrame([feat])
        X_aligned = X_df.reindex(columns=self.l4_cols, fill_value=0.0)
        
        # Ensure numeric type
        X_aligned = X_aligned.astype(float)
        
        dmatrix = xgb.DMatrix(X_aligned)
        probs = self.models['l4'].predict(dmatrix)[0]
        
        print("\nDEEPBOOST ANALYSIS:")
        print(f"  Probability Home Win:  {probs[0]*100:.1f}%")
        print(f"  Probability Draw:      {probs[1]*100:.1f}%")
        print(f"  Probability Away Win:  {probs[2]*100:.1f}%")
        
        res = ["Home Win", "Draw", "Away Win"][np.argmax(probs)]
        conf = np.max(probs)
        print(f"\nFINAL VERDICT: {res} ({conf*100:.1f}% Confidence)")
        return probs

if __name__ == "__main__":
    sim = DeepBoostSimulator()
    
    # EXAMPLE: Man City vs Chelsea
    # Full Strength City
    city_full = ["Ederson", "Walker", "Dias", "Stones", "Gvardiol", "Rodri", "De Bruyne", "Silva", "Foden", "Haaland", "Grealish"]
    # Weakened City (No Haaland, No Rodri)
    city_weak = ["Ederson", "Walker", "Dias", "Ake", "Lewis", "Kovacic", "Nunes", "Silva", "Bob", "Alvarez", "Grealish"]
    
    chelsea = ["Sanchez", "James", "Disasi", "Colwill", "Cucurella", "Enzo", "Caicedo", "Gallagher", "Palmer", "Jackson", "Sterling"]

    print("RUNNING BATCH SIMULATION FROM JSON...")
    lineups_path = Path("data/raw/jan2026_lineups.json")
    if lineups_path.exists():
        with open(lineups_path, "r") as f:
            data = json.load(f)
            
        for m_id, m_info in data['matches'].items():
            sim.predict_match(
                m_info['home'], 
                m_info['away'], 
                m_info['h_lineup'], 
                m_info['a_lineup'],
                h_mgr_style=m_info.get('h_mgr', 'Attack'),
                a_mgr_style=m_info.get('a_mgr', 'Defense')
            )
    else:
        print("Jan 2026 Lineups JSON not found.")

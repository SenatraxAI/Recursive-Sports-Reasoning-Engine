import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
import logging
from pathlib import Path
from src.system_fit_calculator import SystemFitCalculator
from src.manager_profiles import Formation
from src.utils import map_team_name, get_current_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recursive_predictor")

class SeasonState:
    def __init__(self, initial_ratings: dict):
        self.ratings = initial_ratings.copy() # {Team: Power_Rating}
        self.alpha_rating = 0.15 # Learning Rate for EMA
        
    def update(self, home_team, away_team, h_goals, a_goals, h_xg, a_xg):
        # Update Ratings based on xG performance
        # Simple EMA: New = Old * (1-alpha) + Perf * alpha
        
        h_perf = h_xg if not pd.isna(h_xg) else h_goals
        a_perf = a_xg if not pd.isna(a_xg) else a_goals
        
        old_h = self.ratings.get(home_team, 1.0)
        old_a = self.ratings.get(away_team, 1.0)
        
        new_h = (old_h * (1 - self.alpha_rating)) + (h_perf * self.alpha_rating)
        new_a = (old_a * (1 - self.alpha_rating)) + (a_perf * self.alpha_rating)
        
        self.ratings[home_team] = new_h
        self.ratings[away_team] = new_a
        
        return old_h, new_h, old_a, new_a

class RecursivePredictor:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.model_dir = Path("data/models/experimental")
        
        # Load Models
        self.l2_model = xgb.Booster(); self.l2_model.load_model(self.model_dir / "level2_learned_aggregator.json")
        self.l3_model = xgb.Booster(); self.l3_model.load_model(self.model_dir / "level3_learned_tactician.json")
        self.l4_model = xgb.Booster(); self.l4_model.load_model(self.model_dir / "level4_learned_outcome.json")
        
        with open(self.model_dir / "calibration_models_cv.pkl", "rb") as f:
            self.calibrators = pickle.load(f)
            
        # Load Features Keys
        with open(self.model_dir / "level2_features.json", "r") as f: self.l2_feats = json.load(f)
        with open(self.model_dir / "level3_features.json", "r") as f: self.l3_feats = json.load(f)
        with open(self.model_dir / "level4_features.json", "r") as f: self.l4_feats = json.load(f)
        
        # Load Static Data
        self.fit_calc = SystemFitCalculator()
        
        # Load Manager Map
        matches_hist = pd.read_parquet(Path("data/processed/processed_matches.parquet"))
        matches_hist['date'] = pd.to_datetime(matches_hist['date'])
        matches_hist = matches_hist.sort_values('date')
        
        self.manager_map = {}
        for _, row in matches_hist.iterrows():
            self.manager_map[row['home_team'].lower()] = row['home_manager']
            self.manager_map[row['away_team'].lower()] = row['away_manager']
            
        with open(Path("data/processed/team_dna_agg.json"), "r") as f:
            self.dna_db = json.load(f)
            
        # Initialize Season State using 2025 Averages
        # Calculate initial power ratings from history
        avg_xg = matches_hist.groupby('home_team')['home_xg'].mean().to_dict()
        self.state = SeasonState(avg_xg)
        logger.info(f"Initialized Season State with {len(avg_xg)} teams.")
        
    def get_prediction(self, m):
        """
        Returns (probs, h_power, a_power) for a match dict.
        """
        h_team_clean = map_team_name(str(m['home_team']))
        a_team_clean = map_team_name(str(m['away_team']))
        
        # 1. Get Dynamic Rating (The "Memory")
        h_power = self.state.ratings.get(h_team_clean, 1.2)
        a_power = self.state.ratings.get(a_team_clean, 1.2)
        
        # 2. Standard Static Features
        h_fmt_str = m.get('home_formation', '4-3-3')
        a_fmt_str = m.get('away_formation', '4-3-3')
        h_struct = Formation.get_structure(h_fmt_str)
        a_struct = Formation.get_structure(a_fmt_str)
        
        h_manager = str(self.manager_map.get(h_team_clean, h_team_clean))
        a_manager = str(self.manager_map.get(a_team_clean, a_team_clean))
        
        h_dna = self.dna_db.get(h_team_clean, {'dna_stamina':0.5, 'dna_passing':0.5})
        a_dna = self.dna_db.get(a_team_clean, {'dna_stamina':0.5, 'dna_passing':0.5})
        
        h_fit = self.fit_calc.calculate_fit(h_dna, h_manager)
        a_fit = self.fit_calc.calculate_fit(a_dna, a_manager)
        
        h_dens = {'density_central': h_struct['density_central'], 'density_attack': h_struct['density_attack'], 'width_balance': h_struct['wide_att'] / max(1, h_struct['density_central'])}
        a_dens = {'density_central': a_struct['density_central'], 'density_attack': a_struct['density_attack'], 'width_balance': a_struct['wide_att'] / max(1, a_struct['density_central'])}
        
        # FIX: Use dynamic h_power/a_power instead of hardcoded 1.5/1.2
        h_l2_in = [h_power, h_fit, h_dens['density_central'], h_dens['density_attack'], h_dens['width_balance'], h_dens['density_central']/(h_dens['width_balance']+0.1)]
        a_l2_in = [a_power, a_fit, a_dens['density_central'], a_dens['density_attack'], a_dens['width_balance'], a_dens['density_central']/(a_dens['width_balance']+0.1)]
        pred_h_goals = self.l2_model.predict(xgb.DMatrix([h_l2_in], feature_names=self.l2_feats))[0]
        pred_a_goals = self.l2_model.predict(xgb.DMatrix([a_l2_in], feature_names=self.l2_feats))[0]
        
        l3_in = [h_fit, a_fit, h_fit - a_fit, h_l2_in[5], a_l2_in[5], h_l2_in[5] - a_l2_in[5]]
        tactical_state = int(self.l3_model.predict(xgb.DMatrix([l3_in], feature_names=self.l3_feats))[0])
        
        is_deadlock = 1 if tactical_state == 0 else 0
        is_h_cont = 1 if tactical_state == 1 else 0
        is_a_cont = 1 if tactical_state == 2 else 0
        is_chaos = 1 if tactical_state == 3 else 0
        
        l4_in = [is_deadlock, is_h_cont, is_a_cont, is_chaos, h_fit - a_fit, h_l2_in[5] - a_l2_in[5], h_fit, a_fit, pred_h_goals, pred_a_goals]
        raw_scores = self.l4_model.predict(xgb.DMatrix([l4_in], feature_names=self.l4_feats))[0]
        
        real_prob_h = self.calibrators['home'].transform([raw_scores[0]])[0]
        real_prob_d = self.calibrators['draw'].transform([raw_scores[1]])[0]
        real_prob_a = self.calibrators['away'].transform([raw_scores[2]])[0]
        total = real_prob_h + real_prob_d + real_prob_a
        
        probs = { 'HOME': real_prob_h/total, 'DRAW': real_prob_d/total, 'AWAY': real_prob_a/total }
        return probs, h_power, a_power

    def run_test_set(self):
        matches = pd.read_csv(self.raw_dir / "jan2026_matches.csv")
        if 'date' in matches.columns:
            matches['date'] = pd.to_datetime(matches['date'])
            matches = matches.sort_values('date').reset_index(drop=True)
            
        # Context is 0-9. Test is 10-End.
        start_idx = 10
        total_pnl = 0.0
        staked = 0.0
        wins = 0
        bets = 0
        
        bankroll = 1000.0
        kelly_fraction = 0.3
        
        print(f"\n{'='*20} RUNNING DYNAMIC TEST SET (Matches {start_idx}-{len(matches)-1}) {'='*20}")
        
        # We need to replay history up to start_idx FIRST
        print("Initializing State from Context Matches...")
        for i in range(start_idx):
            m = matches.iloc[i]
            h_team = map_team_name(str(m['home_team']))
            a_team = map_team_name(str(m['away_team']))
            self.state.update(h_team, a_team, m['home_goals'], m['away_goals'], m.get('home_xg', np.nan), m.get('away_xg', np.nan))
            
        # Now Loop through Test Set, PREDICTING then UPDATING
        for i in range(start_idx, len(matches)):
            m = matches.iloc[i]
            
            
            # 1. PREDICT
            h_team = map_team_name(str(m['home_team']))
            a_team = map_team_name(str(m['away_team']))
            
            probs, h_power, a_power = self.get_prediction(m)
            real_prob_h = probs['HOME']
            real_prob_d = probs['DRAW']
            real_prob_a = probs['AWAY']
            
            # BETTING
            odds_h = m.get('home_odds', 2.5); odds_d = m.get('draw_odds', 3.2); odds_a = m.get('away_odds', 2.8)
            choices = [('HOME', real_prob_h, odds_h), ('DRAW', real_prob_d, odds_d), ('AWAY', real_prob_a, odds_a)]
            
            best_pick = "SKIP"
            stake_amt = 0.0
            max_edge = 0.0
            odds_taken = 0.0
            MIN_EDGE = 0.0 # Just need positive EV if confidence is high
            MIN_CONFIDENCE = 0.70
            
            for label, prob, odds in choices:
                edge = (prob * odds) - 1.0
                # DEBUG PRINT
                if edge > 0:
                    print(f"DEBUG: Checking {label} Prob={prob:.2f} Odds={odds} Edge={edge:.2f} >= Conf={MIN_CONFIDENCE}?")
                
                if prob >= MIN_CONFIDENCE and edge > MIN_EDGE:
                    kelly_p = edge / (odds - 1)
                    bet_fraction = min(0.10, max(0.0, kelly_p * kelly_fraction)) # Allow slightly larger bets for high confidence
                    if edge > max_edge:
                        max_edge = edge; best_pick = label; odds_taken = odds; stake_amt = bankroll * bet_fraction
            
            # RESOLVE
            res_label = "DRAW"
            if m['home_goals'] > m['away_goals']: res_label = "HOME"
            elif m['home_goals'] < m['away_goals']: res_label = "AWAY"
            
            pnl = 0.0
            if best_pick != "SKIP" and stake_amt > 1.0:
                bets += 1
                if best_pick == res_label:
                    pnl = (stake_amt * odds_taken) - stake_amt
                    wins += 1
                else:
                    pnl = -stake_amt
                staked += stake_amt
                total_pnl += pnl
                
            bankroll += pnl
            print(f"Match {i+1} {h_team} vs {a_team}: Pick {best_pick} | Res {res_label} | PnL {pnl:.2f} | Rating ({h_power:.2f} vs {a_power:.2f})")
            
            # UPDATE STATE
            self.state.update(h_team, a_team, m['home_goals'], m['away_goals'], m.get('home_xg', np.nan), m.get('away_xg', np.nan))

        print("="*40)
        print(f"DYNAMIC ENGINE RESULTS (Matches {start_idx}-{len(matches)-1})")
        print(f"Bets: {bets} | Wins: {wins}")
        print(f"Profit: ${total_pnl:.2f}")
        print(f"ROI: {total_pnl/staked if staked>0 else 0:.1%}")
        
if __name__ == "__main__":
    RecursivePredictor().run_test_set()

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
import logging
from src.utils import normalize_name
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betting_sim_2026")

class BettingSim2026:
    def __init__(self):
        self.model_dir = Path("data/models")
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        
        # Load Models
        self.dna_model = xgb.Booster()
        self.dna_model.load_model(self.model_dir / "layer0_dna.json")
        
        self.l1_models = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            mod = xgb.Booster()
            mod.load_model(self.model_dir / f"level1_{pos}.json")
            self.l1_models[pos] = mod
            
        self.l2_home = xgb.Booster()
        self.l2_home.load_model(self.model_dir / "level2_home_offensive_power.json")
        self.l2_away = xgb.Booster()
        self.l2_away.load_model(self.model_dir / "level2_away_offensive_power.json")
        
        self.l3_matchup = xgb.Booster()
        self.l3_matchup.load_model(self.model_dir / "level3_matchup.json")
        with open(self.model_dir / "level3_features.json", "r") as f:
            self.l3_features = json.load(f)
            
        self.l4_outcome = xgb.Booster()
        self.l4_outcome.load_model(self.model_dir / "level4_outcome.json")
        with open(self.model_dir / "level4_features.json", "r") as f:
            self.l4_features = json.load(f)

        # Load Metadata
        self.identity_df = pd.read_parquet(self.processed_dir / "trinity_player_matrix.parquet")
        self.identity_df['norm_name'] = self.identity_df['player_name'].apply(normalize_name)
        # Drop duplicates before setting index to prevent ValueError
        self.player_lookup = self.identity_df.drop_duplicates('norm_name').set_index('norm_name').to_dict('index')
        
        l0_feat_df = pd.read_parquet(self.processed_dir / "layer0_training_features.parquet")
        self.dna_priors = l0_feat_df.sort_values('target_season').groupby('norm_name').tail(1)
            
        # Load Baseline (The "Market Maker")
        self.baseline_model = xgb.Booster()
        self.baseline_model.load_model(self.model_dir / "level4_baseline.json")

    def get_market_odds(self, baseline_probs: np.ndarray, margin: float = 0.05) -> np.ndarray:
        """Simulate market odds using the Baseline Model + Margin."""
        # Probs: [Home, Draw, Away]
        odds = 1.0 / (baseline_probs + (margin/3))
        return np.round(odds, 2)

    def run_sim(self):
        matches = pd.read_csv(self.raw_dir / "jan2026_matches.csv").to_dict('records')
        with open(self.raw_dir / "jan2026_lineups.json", "r") as f:
            lineups = json.load(f)['matches']
            
        bankroll = 1000.0
        staking_unit = 50.0 # Standard flat unit for simplicity
        log = []
        
        print(f"\nSTARTING 2026 BETTING SIMULATION (Bankroll: ${bankroll})")
        print("="*80)
        
        for idx, m in enumerate(matches):
            m_id = str(idx + 1)
            lineup = lineups.get(m_id)
            if not lineup: 
                # print(f"SKIP (No Lineup): {m['home_team']} vs {m['away_team']}")
                continue
            
            # --- 1. LEVEL 1: PLAYER xG ---
            team_xg = {'h': 0.0, 'a': 0.0}
            for side in ['h', 'a']:
                for p_name in lineup[f'{side}_lineup']:
                    p_norm = normalize_name(p_name)
                    p_info = self.player_lookup.get(p_norm, {})
                    
                    pos = str(p_info.get('position', 'FW')).upper()
                    pos_key = 'FWD'
                    if 'GK' in pos: pos_key = 'GK'
                    elif 'D' in pos: pos_key = 'DEF'
                    elif 'M' in pos or 'AMC' in pos: pos_key = 'MID'
                    
                    mod = self.l1_models[pos_key]
                    val_X = pd.DataFrame(0.0, index=[0], columns=mod.feature_names)
                    val_X['time'] = 90.0
                    
                    # DNA Prior
                    p_dna = self.dna_priors[self.dna_priors['norm_name'] == p_norm]
                    if not p_dna.empty:
                        val_X['dna_prior_xG'] = float(self.dna_model.predict(xgb.DMatrix(p_dna[self.dna_model.feature_names]))[0])
                    
                    team_xg[side] += mod.predict(xgb.DMatrix(val_X.values, feature_names=mod.feature_names))[0]

            # --- 2. LEVEL 2: TEAM STRENGTH ---
            # (Simplified L2 features for this sim, matching the retraining logic)
            l2_feats = pd.DataFrame(0.0, index=[0], columns=self.l2_home.feature_names)
            l2_feats['h_l2_agg_goals_p90'] = team_xg['h']
            l2_feats['a_l2_agg_goals_p90'] = team_xg['a']
            
            h_power = self.l2_home.predict(xgb.DMatrix(l2_feats))[0]
            a_power = self.l2_away.predict(xgb.DMatrix(l2_feats))[0]

            # --- 3. LEVEL 3: MATCHUP ---
            l3_X = pd.DataFrame(0.0, index=[0], columns=self.l3_features)
            # (In a real scenario, we'd fill manager styles here)
            tac_control = self.l3_matchup.predict(xgb.DMatrix(l3_X.values, feature_names=self.l3_features))[0]

            # --- 4. LEVEL 4: OUTCOME (THE DNA MODEL) ---
            l4_X = pd.DataFrame(0.0, index=[0], columns=self.l4_features)
            l4_X['h_recent_xg'] = h_power
            l4_X['a_recent_xg'] = a_power
            l4_X['l3_pred_tactical_control'] = tac_control
            
            # --- 5. PIVOT: TOTALS MARKET (Over/Under 2.5) ---
            # DNA Model Estimates
            dna_total_xg = h_power + a_power
            # Poisson Prob of > 2.5 Goals
            import math
            def poisson_prob_over_2_5(lam):
                p0 = math.exp(-lam)
                p1 = math.exp(-lam) * lam
                p2 = math.exp(-lam) * (lam**2) / 2
                return 1 - (p0 + p1 + p2)
            
            dna_prob_over = poisson_prob_over_2_5(dna_total_xg)
            dna_prob_under = 1.0 - dna_prob_over
            
            # Market Estimates (Static League Baseline)
            # Simulating a "dumb" market that sets lines based on league average
            mkt_total_xg = 2.85 # Modern PL average
            mkt_prob_over = poisson_prob_over_2_5(mkt_total_xg)
            
            # Market Odds (with margin)
            odds_over = round(1.0 / (mkt_prob_over + 0.02), 2)
            odds_under = round(1.0 / ((1 - mkt_prob_over) + 0.02), 2)
            
            # Value Finding
            pick = "NONE"
            odds_taken = 0.0
            
            # Bet OVER if DNA says more goals than market
            if dna_prob_over * odds_over > 1.01:
                 pick = "OVER"
                 odds_taken = odds_over
            # Bet UNDER if DNA says fewer goals than market
            elif dna_prob_under * odds_under > 1.01:
                 pick = "UNDER"
                 odds_taken = odds_under
                 
            # Result Determination
            total_goals = m['home_goals'] + m['away_goals']
            actual_res = "UNDER"
            if total_goals > 2.5: actual_res = "OVER"
            
            win = False
            if pick == "OVER" and total_goals > 2.5: win = True
            if pick == "UNDER" and total_goals < 2.5: win = True
            pnl = -staking_unit if pick != "NONE" else 0.0
            if win: pnl = (staking_unit * odds_taken) - staking_unit
            
            bankroll += pnl
            
            log.append({
                'Match': f"{m['home_team'][:10]}.. vs {m['away_team'][:10]}..",
                'Pick': pick, 
                'Odds': odds_taken, 
                'Res': actual_res[:4],
                'PnL': pnl
            })
            
            status = "WIN" if win else "LOSS"
            if pick == "NONE": status = "SKIP"
            print(f"{m['home_team'][:10]:<10} vs {m['away_team']:<10} | Pick:{pick:<4} | {status} | PnL:{pnl:>5.1f}")

        # REPORT
        df = pd.DataFrame(log)
        print("="*80)
        print(f"FINAL BANKROLL: ${bankroll:.2f}")
        print(f"TOTAL PROFIT:   ${bankroll - 1000.0:.2f}")
        total_bets = len(df[df['Pick'] != 'NONE'])
        win_rate = len(df[df['PnL'] > 0]) / total_bets if total_bets > 0 else 0
        print(f"WIN RATE:       {win_rate:.1%} ({len(df[df['PnL'] > 0])}/{total_bets})")
        print(f"ROI:            {((bankroll - 1000.0) / (total_bets * staking_unit)):.1%}" if total_bets > 0 else "ROI: N/A")
        print("="*80)

if __name__ == "__main__":
    sim = BettingSim2026()
    sim.run_sim()

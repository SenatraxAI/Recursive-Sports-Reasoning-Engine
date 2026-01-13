import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
import logging
from src.utils import normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blind_test_2026")

def run_blind_test():
    model_dir = Path("data/models")
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # 1. Load Data
    with open(raw_dir / "jan2026_lineups.json", "r") as f:
        lineups = json.load(f)['matches']
    matches_df = pd.read_csv(raw_dir / "jan2026_matches.csv")
    
    # Load DNA Model and Features
    dna_model = xgb.Booster(); dna_model.load_model(model_dir / "layer0_dna.json")
    l0_features_df = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    l0_feats = dna_model.feature_names
    latest_dna = l0_features_df.sort_values('target_season').groupby('norm_name').tail(1)
    
    # Load Identity matrix for attributes and positions
    identity_df = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    identity_df['norm_name'] = identity_df['player_name'].apply(normalize_name)
    
    # Create lookup dictionaries for speed
    player_attr_lookup = identity_df.drop_duplicates('norm_name').set_index('norm_name').to_dict('index')

    # 2. Config generations
    gen_configs = {
        'Legacy Standard': {'path': model_dir, 'prefix': 'level1_', 'dna': False},
        'New Prod (DNA)': {'path': model_dir / 'experiments', 'prefix': 'new_prod_level1_', 'dna': True}
    }
    
    results = []

    for gen_name, cfg in gen_configs.items():
        logger.info(f"Generating blind predictions for {gen_name}...")
        
        # Load all 4 position models
        gen_models = {}
        for k in ['GK', 'DEF', 'MID', 'FWD']:
            # Handle naming inconsistency (DEF vs DF)
            m_key = k
            if k == 'DEF': m_key = 'DF' if (cfg['path'] / f"{cfg['prefix']}DF.json").exists() else 'DEF'
            if k == 'MID': m_key = 'MF' if (cfg['path'] / f"{cfg['prefix']}MF.json").exists() else 'MID'
            if k == 'FWD': m_key = 'FW' if (cfg['path'] / f"{cfg['prefix']}FW.json").exists() else 'FWD'
            
            p = cfg['path'] / f"{cfg['prefix']}{m_key}.json"
            if p.exists():
                mod = xgb.Booster(); mod.load_model(p)
                gen_models[k] = mod

        match_preds = []
        for idx, m_row in matches_df.iterrows():
            m_id = None
            for lid, linfo in lineups.items():
                if linfo['home'] == m_row['home_team']:
                    m_id = lid; break
            
            if not m_id: continue
            
            lineup = lineups[m_id]
            pred_scores = {'home': 0.0, 'away': 0.0}
            
            for side in ['h', 'a']:
                team_total = 0.0
                players = lineup[f'{side}_lineup']
                for p_name in players:
                    p_norm = normalize_name(p_name)
                    p_info = player_attr_lookup.get(p_norm, {})
                    
                    pos = str(p_info.get('position', 'FW')).upper()
                    pos_key = 'FWD'
                    if 'GK' in pos: pos_key = 'GK'
                    elif 'DF' in pos or 'DEF' in pos: pos_key = 'DEF'
                    elif 'MF' in pos or 'MID' in pos: pos_key = 'MID'
                    
                    if pos_key in gen_models:
                        mod = gen_models[pos_key]
                        feats = mod.feature_names
                        
                        # Build Feature Vector
                        test_X = pd.DataFrame(0.0, index=[0], columns=feats)
                        
                        # Fill attributes from trinity
                        for f in feats:
                            if f in p_info:
                                val = p_info[f]
                                try: test_X[f] = float(val) if val is not None else 0.0
                                except: pass
                        
                        # Fill DNA Prior
                        if cfg['dna'] and 'dna_prior_xG' in feats:
                            p_dna = latest_dna[latest_dna['norm_name'] == p_norm]
                            if not p_dna.empty:
                                val = dna_model.predict(xgb.DMatrix(p_dna[l0_feats]))[0]
                                test_X['dna_prior_xG'] = val
                        
                        test_X['time'] = 90.0
                        
                        # Rename for legacy if needed (Position_Clean_Clean)
                        if 'Position_Clean_Clean' in feats:
                            test_X['Position_Clean_Clean'] = 1.0 # Proxy for 'is in this position'
                        
                        try:
                            # Standardize order and force types
                            test_X = test_X[feats].astype(float)
                            dtest = xgb.DMatrix(test_X.values, feature_names=feats)
                            p_val = mod.predict(dtest)[0]
                        except Exception as e:
                            logger.error(f"Prediction failed for {p_norm}: {e}")
                            raise e
                        
                        team_total += p_val
                
                pred_scores['home' if side == 'h' else 'away'] = team_total
            
            if idx == 0:
                logger.info(f"Sample Match ({m_row['home_team']}): Pred_H={pred_scores['home']:.2f}, Pred_A={pred_scores['away']:.2f}")

            match_preds.append({
                'Pred_H': pred_scores['home'],
                'Pred_A': pred_scores['away'],
                'Actual_H': m_row['home_goals'],
                'Actual_A': m_row['away_goals']
            })
            
        pdf = pd.DataFrame(match_preds)
        pdf['H_Err'] = abs(pdf['Actual_H'] - pdf['Pred_H'])
        pdf['A_Err'] = abs(pdf['Actual_A'] - pdf['Pred_A'])
        mae = (pdf['H_Err'].mean() + pdf['A_Err'].mean()) / 2
        results.append({'Gen': gen_name, 'MAE': round(mae, 4)})

    print("\n" + "="*50)
    print("PHASE 2 BATTLE: TEAM XG AGGREGATION MAE (2026)")
    print("="*50)
    print(pd.DataFrame(results).to_string())
    print("="*50)

if __name__ == "__main__":
    run_blind_test()

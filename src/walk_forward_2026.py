import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
import logging
from src.utils import normalize_name
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("walk_forward_2026")

def run_walk_forward():
    model_dir = Path("data/models")
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # 1. LOAD DATA
    with open(raw_dir / "jan2026_lineups.json", "r") as f:
        lineups = json.load(f)['matches']
    matches_df = pd.read_csv(raw_dir / "jan2026_matches.csv").sort_values('date').reset_index(drop=True)
    
    # Split into Context (First Half) and Battle (Second Half)
    midpoint = len(matches_df) // 2
    context_matches = matches_df.iloc[:midpoint]
    battle_matches = matches_df.iloc[midpoint:]
    
    logger.info(f"Context Matches (Updates Form): {len(context_matches)}")
    logger.info(f"Battle Matches (Prediction Test): {len(battle_matches)}")
    
    # Load DNA Model and Features
    dna_model = xgb.Booster(); dna_model.load_model(model_dir / "layer0_dna.json")
    l0_features_df = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    l0_feats = dna_model.feature_names
    latest_dna = l0_features_df.sort_values('target_season').groupby('norm_name').tail(1)
    
    # Load Identity matrix for attributes
    identity_df = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    identity_df['norm_name'] = identity_df['player_name'].apply(normalize_name)
    player_attr_lookup = identity_df.drop_duplicates('norm_name').set_index('norm_name').to_dict('index')

    # Load New Exp Models
    gen_models = {}
    for k in ['GK', 'DEF', 'MID', 'FWD']:
        p = model_dir / 'experiments' / f"new_exp_level1_{k}.json"
        if p.exists():
            mod = xgb.Booster(); mod.load_model(p)
            gen_models[k] = mod

    # 2. THE FORM MIRROR (Live state)
    # We initialize form from the end of 2025 if possible, but for this test we'll start clean
    # and update using the Context Matches.
    player_form = {} # {norm_name: [list of recent xG]}

    def update_form(match_id):
        # In a real system, we'd have the actual rosters for the context matches
        # Since we only have 'jan2026_lineups.json', we'll use that to see who played
        linfo = lineups.get(str(match_id))
        if not linfo: return
        
        # We also need the outcome to know the xG, but matches_df only has goals.
        # We will use 'Actual Goals' as a proxy for 'Latest Performance' if xG is unavailable.
        m_row = matches_df[matches_df.index == (int(match_id)-1)] # matches_df is 0-indexed
        if m_row.empty: return
        
        h_score = m_row.iloc[0]['home_goals']
        a_score = m_row.iloc[0]['away_goals']
        
        # Distribute goals across lineup for simplistic form update
        for side, score in [('h', h_score), ('a', a_score)]:
            players = linfo[f'{side}_lineup']
            for p in players:
                p_norm = normalize_name(p)
                if p_norm not in player_form: player_form[p_norm] = []
                # Simple update: if team scored, player gets a form boost
                # This simulates the "Momentum" the model sees.
                player_form[p_norm].append(score / len(players))
                if len(player_form[p_norm]) > 5: player_form[p_norm].pop(0)

    # PHASE 1: "SHOW THE AI" (Context Processing)
    logger.info("Processing context matches to update form mirrors...")
    for idx, row in context_matches.iterrows():
        update_form(idx + 1) # Assumes lineup keys match 1-based index

    # PHASE 2: "PREDICT" (The Battle)
    logger.info("Executing Battle Phase predictions...")
    battle_results = []
    
    for idx, m_row in battle_matches.iterrows():
        m_id = str(idx + 1)
        lineup = lineups.get(m_id)
        if not lineup: continue
        
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
                elif 'D' in pos: pos_key = 'DEF'
                elif 'M' in pos or 'AMC' in pos: pos_key = 'MID'
                
                if pos_key in gen_models:
                    mod = gen_models[pos_key]
                    feats = mod.feature_names
                    
                    test_X = pd.DataFrame(0.0, index=[0], columns=feats)
                    
                    # 1. Static Attributes
                    for f in feats:
                        if f in p_info:
                            val = p_info[f]; test_X[f] = float(val) if val else 0.0
                    
                    # 2. Recent Form (Updated from Context Phase)
                    form_avg = sum(player_form.get(p_norm, [0])) / max(len(player_form.get(p_norm, [0])), 1)
                    if 'prev_xG_5' in feats: test_X['prev_xG_5'] = form_avg
                    
                    # 3. DNA Prior
                    if 'dna_prior_xG' in feats:
                        p_dna = latest_dna[latest_dna['norm_name'] == p_norm]
                        if not p_dna.empty:
                            val = dna_model.predict(xgb.DMatrix(p_dna[l0_feats]))[0]
                            test_X['dna_prior_xG'] = val
                    
                    test_X['time'] = 90.0
                    
                    p_val = mod.predict(xgb.DMatrix(test_X.values, feature_names=feats))[0]
                    team_total += p_val
            
            pred_scores['home' if side == 'h' else 'away'] = team_total
            
        battle_results.append({
            'Match': f"{m_row['home_team']} vs {m_row['away_team']}",
            'Pred_H': float(round(pred_scores['home'], 2)),
            'Pred_A': float(round(pred_scores['away'], 2)),
            'Actual_H': int(m_row['home_goals']),
            'Actual_A': int(m_row['away_goals'])
        })

    # Save to JSON for machine retrieval
    with open('walk_forward_scores.json', 'w') as f:
        json.dump(battle_results, f, indent=4)

    # FINAL REPORT
    res_df = pd.DataFrame(battle_results)
    res_df['H_Err'] = abs(res_df['Actual_H'] - res_df['Pred_H'])
    res_df['A_Err'] = abs(res_df['Actual_A'] - res_df['Pred_A'])
    
    print("\n" + "="*70)
    print("FINAL 2026 WALK-FORWARD BATTLE RESULTS (SECOND HALF OF JAN)")
    print("="*70)
    print(res_df[['Match', 'Pred_H', 'Actual_H', 'Pred_A', 'Actual_A', 'H_Err', 'A_Err']].to_string())
    print("-" * 70)
    print(f"Global MAE on Unseen Matches: {(res_df['H_Err'].mean() + res_df['A_Err'].mean())/2:.4f}")
    print("="*70)

if __name__ == "__main__":
    run_walk_forward()

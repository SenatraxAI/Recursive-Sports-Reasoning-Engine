import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

def full_squad_audit():
    model_dir = Path("data/models/experiments")
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    print(f"{'Pos':<5} | {'DNA':<5} | {'Form':<5} | {'Pred xG':<8}")
    print("-" * 35)
    
    total_xg = 0
    # Simulate a standard 4-3-3
    squad = [('GK', 1), ('DEF', 4), ('MID', 3), ('FWD', 3)]
    
    for pos, count in squad:
        mod_path = model_dir / f"new_exp_level1_{pos}.json"
        if not mod_path.exists(): continue
        
        model = xgb.Booster()
        model.load_model(mod_path)
        feats = model.feature_names
        
        # Give them a 'typical' DNA for their position
        # GK: 0.0, DEF: 0.05, MID: 0.15, FWD: 0.45
        dna_val = 0.0
        if pos == 'DEF': dna_val = 0.05
        elif pos == 'MID': dna_val = 0.15
        elif pos == 'FWD': dna_val = 0.45
        
        test_X = pd.DataFrame(0.0, index=[0], columns=feats)
        test_X['dna_prior_xG'] = dna_val
        test_X['time'] = 90.0
        
        pred = model.predict(xgb.DMatrix(test_X.values, feature_names=feats))[0]
        print(f"{pos:<5} | {dna_val:<5.2f} | {0.0:<5.2f} | {pred:<8.4f} (x{count})")
        total_xg += pred * count
        
    print("-" * 35)
    print(f"TOTAL ESTIMATED TEAM xG: {total_xg:.2f}")

if __name__ == "__main__":
    full_squad_audit()

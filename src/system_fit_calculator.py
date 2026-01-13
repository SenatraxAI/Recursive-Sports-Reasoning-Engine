import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List
import logging
import json
from pathlib import Path
from src.manager_profiles import get_manager_profile, ManagerTacticalSetup

logger = logging.getLogger(__name__)

class SystemFitCalculator:
    def __init__(self, dna_model_path: str = "data/models/layer0_dna.json", profile_path: str = "data/models/experimental/manager_profiles_learned.json"):
        print("DEBUG: SystemFitCalculator.__init__ start")
        try:
            # self.dna_model = xgb.Booster()
            # self.dna_model.load_model(dna_model_path)
            pass # Removed unused model loading to prevent crashes
            
            # Load Learned Profiles
            self.profiles = {}
            norm_path = Path(profile_path)
            if norm_path.exists():
                with open(norm_path, "r") as f:
                    self.profiles = json.load(f)
                print(f"INFO: Loaded {len(self.profiles)} learned manager profiles.")
            else:
                print("WARNING: Learned profiles not found! Falling back to Hardcoded DB.")
        except Exception as e:
            print(f"CRITICAL ERROR in SystemFitCalculator.__init__: {e}")
            import traceback
            traceback.print_exc()
            raise e
            
    def get_learned_constraints(self, manager_name: str) -> Dict[str, float]:
        # Exact match or first substring match
        if manager_name in self.profiles:
            return self.profiles[manager_name]
            
        for key, p in self.profiles.items():
            if key in manager_name: # Simple fuzzy
                 return p
                 
        return {} # Empty means use defaults/skip

    def calculate_fit(self, player_dna: Dict[str, float], manager_name: str) -> float:
        """
        Calculates how well a player fits a manager's system.
        Returns a score between 0.0 (Terrible Fit) and 1.0 (Perfect Fit).
        """
        # print(f"DEBUG: Calculating fit for {manager_name}")
        # Try learned first, then hardcoded
        try:
            learned_c = self.get_learned_constraints(manager_name)
        except Exception as e:
            print(f"CRASH in get_learned_constraints: {e}")
            return 0.5
        
        if learned_c:
             # Use the learned dictionary directly
             # This bypasses the ManagerTacticalSetup object for speed/flexibility
             req_stamina = learned_c.get('req_stamina', 0.5)
             req_pass = learned_c.get('req_pass_completion', 0.5)
             req_def = learned_c.get('req_defensive_workrate', 0.5)
        else:
             profile = get_manager_profile(manager_name)
             req_stamina = profile.req_stamina
             req_pass = profile.req_pass_completion
             req_def = profile.req_defensive_workrate
             
        penalties = []
        
        # Stamina Check
        if 'dna_stamina' in player_dna:
            gap = max(0, req_stamina - player_dna['dna_stamina'])
            penalties.append(gap)
            
        if 'dna_passing' in player_dna:
            gap = max(0, req_pass - player_dna['dna_passing'])
            penalties.append(gap)
            
        if 'dna_defense' in player_dna:
            gap = max(0, req_def - player_dna['dna_defense'])
            penalties.append(gap)

        # 2. Calculate Final Score
        if not penalties:
            return 0.5 # Unknown fit
            
        avg_penalty = np.mean(penalties)
        
        # Fit Score = 1.0 - (Penalty * Sensitivity)
        # We want a harsh penalty for big mismatches.
        fit_score = 1.0 - (avg_penalty * 1.5)
        
        return max(0.1, min(1.0, fit_score))

    def batch_calculate_team_fit(self, players: List[Dict], manager_name: str) -> float:
        """Calculates the average system fit for a Starting XI."""
        fits = [self.calculate_fit(p, manager_name) for p in players]
        return np.mean(fits)

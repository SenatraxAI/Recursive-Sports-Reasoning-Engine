import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Optional, Any

from src.DeepBoostNetwork import DeepBoostNetwork
from src.SystemFitCalculator import SystemFitCalculator
from src.ImprovedPlayerLayer import ImprovedPlayerLayer
from src.MatchInstance import MatchInstance, ManagerContext

logger = logging.getLogger("MatchPredictor")

class MatchPredictor:
    """
    Hierarchical 2.0: The Unified Prediction Interface.
    Fuses L1, L2, L3 data into a DeepBoost L4 prediction.
    """
    
    def __init__(self):
        self.model_dir = Path("data/models")
        self.raw_dir = Path("data/raw")
        
        # Load the Absolut Machine (DeepBoost)
        self.db = DeepBoostNetwork(max_layers=3, neurons_per_layer=4)
        # Note: In production we would LOAD the saved weights here
        # self.db.load(self.model_dir / "deepboost_2_0.json") 
        
        self.fit_calc = SystemFitCalculator()
        self.player_layer = ImprovedPlayerLayer()
        
        # Load tactical metadata
        with open(self.raw_dir / "manager_master.json", "r") as f:
            self.managers = json.load(f)

    def request_prediction(self, home_team: str = None, away_team: str = None, instance: Optional[MatchInstance] = None):
        """
        Point 12 & 13: Request a full match analysis.
        Can pass team names (auto-lookup) or a custom MatchInstance.
        """
        if instance:
            h_team = instance.home_team
            a_team = instance.away_team
            h_man_logic = instance.home_manager.__dict__
            a_man_logic = instance.away_manager.__dict__
            h_lineup = instance.home_lineup
            a_lineup = instance.away_lineup
        else:
            h_team = home_team
            a_team = away_team
            h_man_logic = self.get_manager_logic(h_team)
            a_man_logic = self.get_manager_logic(a_team)
            h_lineup = [] # Would default to master roster
            a_lineup = []

        print(f"\n🏟️  ANALYZING: {h_team} vs {a_team}")
        print("="*50)
        
        # 2. SYSTEM FIT & STRENGTH (L1 + L2)
        # In a real scenario, this would loop through the actual lineups provided
        h_fit = 0.82 if h_man_logic.get('experience', 0) > 5 else 0.70
        a_fit = 0.85 if a_man_logic.get('experience', 0) > 5 else 0.72
        
        # If we have a lineup, we'd calculate squad strength from player_layer
        h_base_talent = self.player_layer.get_squad_strength(h_lineup) if h_lineup else 8.5
        a_base_talent = self.player_layer.get_squad_strength(a_lineup) if a_lineup else 8.2
        
        h_eff = self.fit_calc.get_effective_team_strength(h_base_talent, h_fit)
        a_eff = self.fit_calc.get_effective_team_strength(a_base_talent, a_fit)
        
        # 3. FEATURE FUSION
        features = np.array([[
            h_eff, a_eff,
            h_eff - a_eff,
            h_fit, a_fit,
            1.0 if h_man_logic.get('style') == 'Possession' else 0.0,
            1.0 if a_man_logic.get('style') == 'Possession' else 0.0
        ]])
        
        # 4. DEEPBOOST INFERENCE
        if self.db.final_clf is None:
            logger.warning("No trained weights found. Seeding a 2.0 Brain for this match...")
            self.db.fit(np.random.rand(20, 7), np.random.randint(0, 3, 20))
            
        profile = self.db.calculate_confidence_profile(features)
        
        # 5. TACTICAL NARRATIVE (Point 13)
        self._print_narrative(h_team, a_team, h_man_logic, a_man_logic, profile)
        
        return profile

    def get_features_from_instance(self, instance: MatchInstance) -> np.ndarray:
        """Helper for the dashboard to get the raw feature vector."""
        h_man_logic = instance.home_manager.__dict__
        a_man_logic = instance.away_manager.__dict__
        
        # Consistent with request_prediction logic
        h_fit = 0.82 if h_man_logic.get('experience', 0) > 5 else 0.70
        a_fit = 0.85 if a_man_logic.get('experience', 0) > 5 else 0.72
        
        h_base_talent = self.player_layer.get_squad_strength(instance.home_lineup) if instance.home_lineup else 8.5
        a_base_talent = self.player_layer.get_squad_strength(instance.away_lineup) if instance.away_lineup else 8.2
        
        h_eff = self.fit_calc.get_effective_team_strength(h_base_talent, h_fit)
        a_eff = self.fit_calc.get_effective_team_strength(a_base_talent, a_fit)
        
        return np.array([[
            h_eff, a_eff, h_eff - a_eff,
            h_fit, a_fit,
            1.0 if h_man_logic.get('style') == 'Possession' else 0.0,
            1.0 if a_man_logic.get('style') == 'Possession' else 0.0
        ]])

    def get_manager_logic(self, team_name: str):
        # Maps raw JSON data to tactical labels
        profile = self.managers.get(team_name, {})
        return {
            'name': profile.get('manager', 'Unknown'),
            'style': profile.get('possession_type', 'Neutral'),
            'pressing': profile.get('pressing_intensity', 'Neutral'),
            'experience': profile.get('tenure_days', 0) / 365,
            'formation': profile.get('primary_formation', '4-3-3')
        }

    def _print_narrative(self, h_team, a_team, h_man, a_man, profile):
        print(f"🧠 TACTICAL BATTLE: {h_man['name']} vs {a_man['name']}")
        print(f"   - {h_team} ({h_man['formation']}) intends to play {h_man['style']} football.")
        print(f"   - {a_team} ({a_man['formation']}) counters with {a_man['pressing']} pressing.")
        print("-" * 50)
        
        msg = "High" if profile['confidence_score'] > 0.75 else "Moderate"
        print(f"🔮 PREDICTION: {profile['prediction']}")
        print(f"📊 CONFIDENCE: {profile['confidence_score']:.1%} ({msg} Certainty)")
        print(f"🛡️ RISK:       {profile['risk_rating']}")
        
        print("-" * 50)
        print(f"💰 SIGNAL:     {profile['betting_signal']}")
        print("="*50)

if __name__ == "__main__":
    # USER INTERFACE
    predictor = MatchPredictor()
    
    # Example Request
    predictor.request_prediction("man_city", "arsenal")
    
    # You can also use this for any match in your database
    # predictor.request_prediction("liverpool", "chelsea")

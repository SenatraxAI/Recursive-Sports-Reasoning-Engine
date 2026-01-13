"""
This is the 'Injury Impact Model'. 
It answers the question: "How much worse is the team because their Star Striker is out?"

We use a concept called 'Replacement Level'. 
If our star striker is an 8/10, and his backup is a 5/10, the team loses 3 points 
of 'Value'. We use that to lower our prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .schema import PlayerInjury

class InjuryImpactModel:
    """
    Quantifies the drop in performance caused by missing players.
    """
    
    def __init__(self, player_values: Dict[str, float], replacement_levels: Dict[str, float]):
        """
        - player_values: Dictionary of [PlayerID: StrengthScore]
        - replacement_levels: Dictionary of [Position: AverageScoreForSubstitute]
        """
        self.player_values = player_values
        self.replacement_levels = replacement_levels
        
    def calculate_team_impact(self, missing_player_ids: List[str], team_squad: List[str]) -> Dict[str, float]:
        """
        Calculates the total 'Value' lost due to injuries.
        """
        total_impact = 0.0
        
        for pid in missing_player_ids:
            if pid in self.player_values:
                # 1. Identify where they play
                pos = self._get_player_position(pid)
                
                # 2. Find the difference between them and the average backup
                replacement_val = self.replacement_levels.get(pos, 0.5)
                impact = self.player_values[pid] - replacement_val
                
                # 3. Add to the total "Hurt" the team feels
                total_impact += max(0.0, impact) 
                
        return {
            "xg_reduction": total_impact * 0.1,    # Decrease predicted goals
            "defensive_drop": total_impact * 0.05, # Increase likely goals conceded
            "overall_impact_score": total_impact
        }

    def _get_player_position(self, player_id: str) -> str:
        """Helper to find player position."""
        return "midfielder" # Placeholder

    def adjust_prediction(self, base_prediction: float, impact: Dict[str, float]) -> float:
        """
        Lowers the win probability based on how many stars are missing.
        """
        # Simple math: Win prob drops by 2% for every unit of 'Impact' lost.
        return base_prediction - impact.get("overall_impact_score", 0.0) * 0.02

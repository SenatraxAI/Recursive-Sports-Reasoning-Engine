import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImprovedPlayerLayer:
    """
    Hierarchical 2.0: Point 1 & 10.
    Calculates stable player ratings using rolling historical performance with exponential decay.
    """
    
    def __init__(self, decay_rate: float = 0.95, window_size: int = 15):
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.player_db: Dict[str, Dict] = {} # norm_name -> {history: list, value: float}
        
    def update_player(self, player_name: str, match_date: datetime, stats: Dict[str, float]):
        """
        Record a new match performance and update the player's rolling valuation.
        """
        if player_name not in self.player_db:
            self.player_db[player_name] = {'history': [], 'current_value': 0.0}
            
        history = self.player_db[player_name]['history']
        history.append({'date': match_date, 'stats': stats})
        
        # Sort by date and keep only the window
        history.sort(key=lambda x: x['date'])
        self.player_db[player_name]['history'] = history[-self.window_size:]
        
        # Recalculate value with exponential decay
        self.player_db[player_name]['current_value'] = self._calculate_decayed_value(player_name)

    def _calculate_decayed_value(self, player_name: str) -> float:
        """
        Calculates the current valuation where more recent matches have higher weight.
        Formula: Value = Sum(Performance_i * Decay^(Age_i)) / Sum(Decay^(Age_i))
        """
        history = self.player_db[player_name]['history']
        if not history:
            return 0.0
            
        performances = []
        weights = []
        
        # Recent matches are at the end of the list
        for i, match in enumerate(reversed(history)):
            # match_performance is a synthetic rating (0-10) derived from stats
            perf = self._synthesize_rating(match['stats'])
            weight = self.decay_rate ** i # i=0 is most recent, weight=1.0
            
            performances.append(perf)
            weights.append(weight)
            
        return sum(p * w for p, w in zip(performances, weights)) / sum(weights)

    def _synthesize_rating(self, stats: Dict[str, float]) -> float:
        """
        Transforms raw stats into a normalized performance rating (0-10).
        This is a heuristic that will be refined by Style Normalization (Point 2).
        """
        # Basic offensive/defensive weightings
        xg = stats.get('xg', 0)
        xa = stats.get('xa', 0)
        tackles = stats.get('tackles', 0)
        interceptions = stats.get('interceptions', 0)
        passes = stats.get('progressive_passes', 0)
        
        # Simple heuristic for now - will be replaced by Layer 2 logic
        rating = (xg * 5.0) + (xa * 3.0) + (passes * 0.2) + (tackles * 0.5) + (interceptions * 0.5)
        return min(max(rating, 0), 10.0) # Clip to 0-10 range

    def get_player_value(self, player_name: str) -> float:
        """Returns the stable valuation for a player."""
        return self.player_db.get(player_name, {}).get('current_value', 0.0)

    def get_squad_strength(self, player_names: List[str]) -> float:
        """
        Point 5: Aggregates individual player values into a total team strength.
        """
        return sum(self.get_player_value(name) for name in player_names)

    def calculate_injury_impact(self, full_squad: List[str], missing_players: List[str]) -> float:
        """
        Point 8: Quantifies the drop in team strength due to absences.
        Impact = (Sum(Full) - Sum(Missing)) / Sum(Full)
        """
        full_strength = self.get_squad_strength(full_squad)
        if full_strength == 0:
            return 0.0
            
        missing_strength = sum(self.get_player_value(name) for name in missing_players)
        return missing_strength / full_strength

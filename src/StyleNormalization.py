import pandas as pd
import numpy as np
from typing import Dict, Any

class StyleNormalization:
    """
    Hierarchical 2.0: Point 2.
    Normalizes player stats by the team's tactical style to create a "Neutral Baseline".
    """
    
    # Normalization Multipliers (Deflated stats in specific contexts need Boosting)
    # If a team has 30% possession, their players' pass volumes are "Deflated".
    TACTICAL_FACTORS = {
        'possession': {
            'low': 1.3,    # Boost stats for players in low-possession teams
            'neutral': 1.0,
            'high': 0.8    # Penalize raw volume for high-possession teams (stat-padding)
        },
        'tempo': {
            'slow': 1.15,  # Boost progressive actions in slow systems
            'fast': 0.9    # Raw speed makes progression easier
        },
        'pressing': {
            'passive': 1.25, # Boost defensive stats in passive blocks
            'aggressive': 0.85 # High press makes turnovers easier
        }
    }

    @staticmethod
    def normalize_player_stats(stats: Dict[str, float], team_context: Dict[str, str]) -> Dict[str, float]:
        """
        Adjusts raw stats based on the tactical environment provided by the ManagerProfile.
        """
        normalized = stats.copy()
        
        # 1. Possession Adjustment
        pos_label = team_context.get('possession', 'neutral')
        pos_mult = StyleNormalization.TACTICAL_FACTORS['possession'].get(pos_label, 1.0)
        
        for key in ['progressive_passes', 'key_passes', 'touches']:
            if key in normalized:
                normalized[key] *= pos_mult
                
        # 2. Pressing Adjustment (Defensive Stats)
        press_label = team_context.get('pressing', 'neutral')
        press_mult = StyleNormalization.TACTICAL_FACTORS['pressing'].get(press_label, 1.0)
        
        for key in ['tackles', 'interceptions', 'blocks']:
            if key in normalized:
                normalized[key] *= press_mult
                
        # 3. Tempo Adjustment (Efficiency)
        tempo_label = team_context.get('tempo', 'neutral')
        tempo_mult = StyleNormalization.TACTICAL_FACTORS['tempo'].get(tempo_label, 1.0)
        
        if 'progressive_carries' in normalized:
            normalized['progressive_carries'] *= tempo_mult
            
        return normalized

    @staticmethod
    def get_style_fit_delta(player_preferred_style: Dict[str, str], current_team_style: Dict[str, str]) -> float:
        """
        Point 9: Calculates the delta between a player's history and current reality.
        Returns a 'Fit Multiplier' (e.g., 0.9 for mismatch, 1.1 for perfect match).
        """
        matches = 0
        total_factors = 0
        
        for style_key in ['possession', 'pressing', 'tempo']:
            if style_key in player_preferred_style and style_key in current_team_style:
                total_factors += 1
                if player_preferred_style[style_key] == current_team_style[style_key]:
                    matches += 1
        
        if total_factors == 0:
            return 1.0
            
        # Match Ratio: 1.0 is perfect, 0.0 is total mismatch
        ratio = matches / total_factors
        # Scale to a multiplier between 0.8 and 1.2
        return 0.8 + (ratio * 0.4)

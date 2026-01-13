"""
This is the 'Tactical Profiler'. 
It tries to understand the HIDDEN styles of managers and players.

Why is this important?
Because a great player might perform poorly if they are in a tactic that doesn't 
suit them (e.g., a slow striker in a high-pressing, fast team).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .schema import Formation, ManagerProfile

class ManagerTacticalProfiler:
    """
    Analyzes a manager's past matches to build a 'Tactical Profile'.
    """
    
    def build_profile_from_matches(self, manager_id: str, matches: pd.DataFrame) -> ManagerProfile:
        """
        Takes a manager's match history and calculates their tendencies.
        """
        # Average Possession: Does this manager like to control the ball?
        avg_possession = matches['possession_pct'].mean() if 'possession_pct' in matches.columns else 50.0
        
        # PPDA (Passes Per Defensive Action): A measure of how hard they press.
        # Lower PPDA = More "Gegenpressing" (like Klopp).
        # Higher PPDA = Sitting back (like Mourinho).
        ppda = matches['ppda'].mean() if 'ppda' in matches.columns else 15.0
        
        # Formation: Which setup do they use most often? (4-3-3, 3-4-3, etc)
        formation_pref = Formation.OTHER
        if 'formation' in matches.columns:
            most_common = matches['formation'].mode()
            if not most_common.empty:
                try:
                    formation_pref = Formation(most_common[0])
                except ValueError:
                    formation_pref = Formation.OTHER

        # Create the final Profile object
        profile = ManagerProfile(
            manager_id=manager_id,
            name=matches['manager_name'].iloc[0] if 'manager_name' in matches.columns else "Unknown",
            formation_preference=formation_pref,
            avg_possession=avg_possession,
            pressing_intensity=ppda,
            transition_speed=0.0 # Placeholder for now
        )
        return profile

class PlayerStyleProfiler:
    """
    Looks at a player's stats and decides what 'Kind' of player they are.
    """
    
    def compute_player_style(self, player_stats: pd.DataFrame) -> Dict[str, float]:
        """
        Groups stats into 'Technical', 'Physical', and 'Mental' buckets.
        """
        # 1. Technical Score: Based on passing accuracy, dribbles, and key passes.
        technical_metrics = ['pass_completion_pct', 'key_passes', 'dribbles_successful']
        present_metrics = [m for m in technical_metrics if m in player_stats.columns]
        
        if not present_metrics:
            technical_score = 50.0 # Default if no data
        else:
            technical_score = player_stats[present_metrics].mean().mean() * 10
            
        # 2. Physical Score: Based on winning duels, tackles, and work rate.
        physical_metrics = ['aerial_duels_won', 'tackles_won', 'distance_covered']
        present_metrics = [m for m in physical_metrics if m in player_stats.columns]
        
        if not present_metrics:
            physical_score = 50.0
        else:
            physical_score = player_stats[present_metrics].mean().mean() * 10
            
        return {
            "technical": min(100.0, max(0.0, technical_score)),
            "physical": min(100.0, max(0.0, physical_score)),
            "mental": 50.0 # Mental is harder to score, usually needs scout data.
        }

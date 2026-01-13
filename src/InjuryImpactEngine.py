from typing import List, Dict, Any
from src.ImprovedPlayerLayer import ImprovedPlayerLayer
from src.SystemFitCalculator import SystemFitCalculator
import logging

logger = logging.getLogger(__name__)

class InjuryImpactEngine:
    """
    Hierarchical 2.0: Point 8.
    Compares team strength with and without players to quantify exact tactical impact.
    """
    
    def __init__(self, player_layer: ImprovedPlayerLayer, fit_calculator: SystemFitCalculator):
        self.player_layer = player_layer
        self.fit_calculator = fit_calculator
        
    def calculate_match_impact(self, 
                               team_id: str, 
                               formation: str, 
                               full_lineup: List[str], 
                               absent_players: List[str],
                               replacement_attrs: Dict[str, Dict[str, float]]):
        """
        1. Calculate Base Effective Strength (Full).
        2. Calculate Effective Strength (Current Lineup with Replacements).
        3. Quantify the Gap.
        """
        # A. Full Strength (Hypothetical)
        # We assume for full strength we have the best players available
        full_base_talent = self.player_layer.get_squad_strength(full_lineup)
        
        # B. Current Lineup Analysis
        # Filter out absentees and identify who is playing in which slot
        available_lineup = [p for p in full_lineup if p not in absent_players]
        
        # Calculate Current Team Fit (using the Calculator)
        fit_results = self.fit_calculator.calculate_team_fit(replacement_attrs, formation)
        cohesion = fit_results.get('overall_cohesion', 0.5)
        
        # Calculate Effective Talent (Talent of those actually on the pitch)
        current_talent = self.player_layer.get_squad_strength(available_lineup)
        
        # Effective Strength Equation
        effective_strength = self.fit_calculator.get_effective_team_strength(current_talent, cohesion)
        
        # C. Impact Metrics
        total_loss = full_base_talent - effective_strength
        percent_loss = (total_loss / full_base_talent) if full_base_talent > 0 else 0.0
        
        return {
            'effective_strength': effective_strength,
            'talent_loss_raw': full_base_talent - current_talent,
            'tactical_friction_impact': current_talent - effective_strength,
            'total_percentage_drop': percent_loss,
            'weak_links_added': fit_results.get('weak_links', []),
            'is_critical_loss': percent_loss > 0.15 # >15% drop is considered critical (e.g. losing a Rodri or KDB)
        }

    def get_player_weight_in_system(self, player_name: str, formation: str, slot: str) -> float:
        """
        How important is this specific player to this specific manager's system?
        """
        player_value = self.player_layer.get_player_value(player_name)
        # Weight = Player Value * System Fit
        # To be implemented with attribute data
        return player_value

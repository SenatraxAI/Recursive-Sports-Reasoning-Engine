from typing import Dict, List, Any, Tuple
from src.SystemRole import SystemRole, RoleRequirements
import logging

logger = logging.getLogger(__name__)

class SystemFitCalculator:
    """
    Hierarchical 2.0: Points 4, 5, and 11.
    Measures how well a squad fits into a manager's specific formation and role requirements.
    """
    
    def __init__(self):
        # Point 11: Manager Tactical Setup Definitions
        # Map Formations to Role Slots
        self.FORMATION_SLOTS = {
            '4-3-3': {
                'GK': SystemRole.SWEEPER_KEEPER,
                'CB1': SystemRole.BALL_PLAYING_CB,
                'CB2': SystemRole.AGGRESSIVE_CB,
                'LB': SystemRole.OVERLAPPING_FULLBACK,
                'RB': SystemRole.OVERLAPPING_FULLBACK,
                'DM': SystemRole.DEEP_LYING_PLAYMAKER,
                'CM1': SystemRole.BOX_TO_BOX,
                'CM2': SystemRole.ADVANCED_PLAYMAKER,
                'LW': SystemRole.INSIDE_FORWARD,
                'RW': SystemRole.TRADITIONAL_WINGER,
                'ST': SystemRole.POACHER
            },
            '4-4-2': {
                'GK': SystemRole.SHOT_STOPPER,
                'CB1': SystemRole.COVER_CB,
                'CB2': SystemRole.AGGRESSIVE_CB,
                'LB': SystemRole.WINGBACK,
                'RB': SystemRole.WINGBACK,
                'LM': SystemRole.TRADITIONAL_WINGER,
                'CM1': SystemRole.HOLDING_MIDFIELDER,
                'CM2': SystemRole.BOX_TO_BOX,
                'RM': SystemRole.TRADITIONAL_WINGER,
                'ST1': SystemRole.TARGET_MAN,
                'ST2': SystemRole.POACHER
            }
            # ... more formations can be added here
        }

    def calculate_player_fit(self, player_attributes: Dict[str, float], formation: str, slot: str) -> float:
        """
        Point 4: Measures how well an individual player matches a role requirement.
        """
        role = self.FORMATION_SLOTS.get(formation, {}).get(slot)
        if not role:
            return 0.5 # Unknown role/formation
            
        return RoleRequirements.get_role_suitability(player_attributes, role)

    def calculate_team_fit(self, squad_attributes: Dict[str, Dict[str, float]], formation: str) -> Dict[str, Any]:
        """
        Point 5: Aggregates individual fits into team-level metrics.
        Returns overall_cohesion, weak_links, and strongest_positions.
        """
        required_slots = self.FORMATION_SLOTS.get(formation)
        if not required_slots:
            return {'cohesion': 0.0, 'error': 'Formation not defined'}
            
        fits = []
        slot_analysis = {}
        
        # We assume squad_attributes maps SlotName (e.g. 'ST') -> AttributeDict
        for slot, role in required_slots.items():
            player_attrs = squad_attributes.get(slot, {})
            fit_score = RoleRequirements.get_role_suitability(player_attrs, role)
            fits.append(fit_score)
            slot_analysis[slot] = {
                'role': role.value,
                'fit': fit_score
            }
            
        cohesion = sum(fits) / len(fits) if fits else 0.0
        
        # Identify weak links (score < 0.6)
        weak_links = [s for s, data in slot_analysis.items() if data['fit'] < 0.6]
        
        # Identify strongest (score > 0.85)
        strongest = [s for s, data in slot_analysis.items() if data['fit'] > 0.85]
        
        return {
            'overall_cohesion': cohesion,
            'slot_analysis': slot_analysis,
            'weak_links': weak_links,
            'strongest_positions': strongest
        }

    def get_effective_team_strength(self, base_strength: float, team_fit: float) -> float:
        """
        Final Equation: Effective Strength = Base Talent * (0.7 + 0.3 * Fit)
        This prevents raw talent from winning if they can't execute the system.
        """
        return base_strength * (0.7 + 0.3 * team_fit)

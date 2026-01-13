from enum import Enum
from typing import Dict, List, Any

class SystemRole(Enum):
    # Goalkeeper roles
    SWEEPER_KEEPER = "sweeper_keeper"
    SHOT_STOPPER = "shot_stopper"
    
    # Defensive roles
    BALL_PLAYING_CB = "ball_playing_center_back"
    AGGRESSIVE_CB = "aggressive_center_back"
    COVER_CB = "cover_center_back"
    OVERLAPPING_FULLBACK = "overlapping_fullback"
    INVERTED_FULLBACK = "inverted_fullback"
    WINGBACK = "wingback"
    
    # Midfield roles
    DEEP_LYING_PLAYMAKER = "deep_lying_playmaker"
    BOX_TO_BOX = "box_to_box_midfielder"
    ADVANCED_PLAYMAKER = "advanced_playmaker"
    HOLDING_MIDFIELDER = "holding_midfielder"
    HALF_SPACE_WINGER = "half_space_winger"
    
    # Attacking roles
    FALSE_NINE = "false_nine"
    POACHER = "poacher"
    TARGET_MAN = "target_man"
    INSIDE_FORWARD = "inside_forward"
    TRADITIONAL_WINGER = "traditional_winger"

class RoleRequirements:
    """
    Defines the statistical and attribute gatekeepers for each SystemRole.
    Used for Points 4 & 11 of the Hierarchical 2.0 Manifesto.
    """
    
    # Mapping of roles to their primary requirements (0-100 scale for attributes)
    REQUIREMENTS = {
        SystemRole.DEEP_LYING_PLAYMAKER: {
            'attributes': {'passing': 80, 'vision': 85, 'composure': 80},
            'stats': {'progressive_passes_p90': 5.0, 'pass_accuracy': 0.85},
            'priority': ['technical', 'tactical']
        },
        SystemRole.POACHER: {
            'attributes': {'finishing': 85, 'off_the_ball': 90, 'pace': 75},
            'stats': {'shots_p90': 3.5, 'goals_per_shot': 0.18},
            'priority': ['technical', 'physical']
        },
        SystemRole.FALSE_NINE: {
            'attributes': {'vision': 80, 'passing': 75, 'dribbling': 80},
            'stats': {'key_passes_p90': 2.0, 'xa_p90': 0.20},
            'priority': ['tactical', 'technical']
        },
        SystemRole.SWEEPER_KEEPER: {
            'attributes': {'passing': 70, 'rushing_out': 85, 'composure': 80},
            'stats': {'defensive_actions_outside_box': 1.5, 'pass_accuracy_long': 0.60},
            'priority': ['technical', 'tactical']
        },
        SystemRole.OVERLAPPING_FULLBACK: {
            'attributes': {'stamina': 85, 'crossing': 75, 'pace': 80},
            'stats': {'crosses_p90': 3.0, 'progressive_carries_p90': 4.0},
            'priority': ['physical', 'technical']
        },
        SystemRole.BALL_PLAYING_CB: {
            'attributes': {'passing': 75, 'positioning': 80, 'calmness': 85},
            'stats': {'progressive_passes_p90': 3.5, 'long_balls_completed_p90': 4.0},
            'priority': ['technical', 'tactical']
        }
    }

    @staticmethod
    def get_role_suitability(player_attributes: Dict[str, float], role: SystemRole) -> float:
        """
        Calculates a score (0-1) of how well a player fits a role based on attributes.
        """
        if role not in RoleRequirements.REQUIREMENTS:
            return 0.5 # Default middle-ground
        
        req = RoleRequirements.REQUIREMENTS[role]['attributes']
        scores = []
        for attr, min_val in req.items():
            player_val = player_attributes.get(attr, 50) # Fallback to average
            # Calculate how much of the requirement they meet
            ratio = min(player_val / min_val, 1.2) # Cap at 1.2 bonus
            scores.append(ratio)
        
        return sum(scores) / len(scores) if scores else 0.5

# Measuring Player Effectiveness Within Managerial Systems

## Moving Beyond Style Matching to System Performance

You've identified a crucial dimension that most prediction models miss entirely. The question isn't just whether a player's preferred style matches the manager's tactical style; it's about how individual players function within the specific system the manager deploys. This is a more nuanced analysis that captures the human element of football—the relationships between players and managers, how managers adapt their systems to their best personnel, and how players respond to specific tactical instructions.

When Pep Guardiola manages a team, he doesn't just impose a rigid 4-3-3 on every squad. He adapts his principles based on who he has available. When he had Lionel Messi, the team was built around creating spaces for Messi to receive the ball in dangerous areas. When he had Erling Haaland, the team shifted to supply more crosses and through balls into the box. The system remained fundamentally possession-based, but the specific patterns of play changed based on the players available. Your model needs to capture this flexibility.

Similarly, players respond differently to different managerial systems not just because of tactical fit but because of how the system uses their specific strengths. A creative midfielder might flourish under one manager who gives them freedom to find pockets of space and express their creativity, while struggling under another manager who demands strict positional discipline and routine passing. The same player, the same tactical style preference, but different managerial instructions create completely different performance outcomes.

## The Manager-Player-System Interaction Model

To measure how players work within managerial systems, you need to track several layers of interaction. The first layer is the direct player-manager relationship: how does the manager use this specific player, and how does the player respond to those instructions. The second layer is the player-player relationship within the system: how does this player complement or conflict with their teammates in the tactical setup. The third layer is the positional role effectiveness: how well does the player execute the specific responsibilities their position requires in this system.

Here's a comprehensive implementation that captures these interactions:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict
import numpy as np

class Position(Enum):
    GK = "goalkeeper"
    CB = "center_back"
    LB = "left_back"
    RB = "right_back"
    DM = "defensive_midfield"
    CM = "central_midfield"
    LM = "left_midfield"
    RM = "right_midfield"
    AM = "attacking_midfield"
    LW = "left_wing"
    RW = "right_wing"
    ST = "striker"

class SystemRole(Enum):
    # Goalkeeper roles
    SWEEPER_KEEPER = "sweeper_keeper"
    COMMANDER = "commander"
    SHOT_STOPPER = "shot_stopper"
    
    # Defensive roles
    BALL_PLAYING_CB = "ball_playing_center_back"
    AGGRESSIVE_CB = "aggressive_center_back"
    COVER_CB = "cover_center_back"
    OVERLAPPING_FULLBACK = "overlapping_fullback"
    INVERTED_FULLBACK = "inverted_fullback"
    WINGBACK = "wingback"
    
    # Midfield roles
    DEEP_LYRIST = "deep_lying_playmaker"
    BOX_TO_BOX = "box_to_box_midfielder"
    ADVANCED_PLAYMAKER = "advanced_playmaker"
    BALL_WASTER = "ball_waster"  # High pass completion, progressive passes
    B2B_BOX_CRASH = "box_crashing_midfielder"
    HALF_SPACE_WINGER = "half_space_winger"
    
    # Attacking roles
    FALSE_NINE = "false_nine"
    POINTER_NINE = "poacher"
    TARGET_MAN = "target_man"
    INSIDE_FORWARD = "inside_forward"
    WINGER = "winger"

@dataclass
class PlayerSystemPerformance:
    """
    Records how a player performs in a specific system role under a specific manager.
    """
    player_id: str
    manager_id: str
    system_role: SystemRole
    
    # Performance metrics for this specific role
    matches_played: int
    wins: int
    draws: int
    losses: int
    
    # Role-specific stats (normalized)
    role_specific_rating: float  # How well they execute role responsibilities
    avg_xg_contribution: float
    avg_xa_contribution: float
    avg_xg_prevented: float      # For defensive players
    
    # System integration metrics
    pass_completion_in_system: float
    progressive_action_rate: float
    turnovers_per_90: float
    recoveries_per_90: float
    
    # Form and consistency
    recent_form_trend: float     # Positive = improving, negative = declining
    performance_variance: float   # Lower = more consistent

@dataclass  
class SystemChemistryMetric:
    """
    Measures how well different players work together in the same system.
    """
    player_pair: Tuple[str, str]
    system_role_pair: Tuple[SystemRole, SystemRole]
    manager_id: str
    
    # Complementary metrics
    complementary_score: float     # How well they complement each other
    conflict_score: float          # How much they conflict
    combined_effectiveness: float  # Team output when both play
    
    # Interaction patterns
    passing_connection_strength: float
    overlapping_run_timing: float  # For fullback-winger pairs
    defensive_coverage_overlap: float
    
    # History
    matches_together: int
    win_rate_together: float

class SystemPerformanceAnalyzer:
    """
    Analyzes how players perform within specific managerial systems.
    This measures the full player-manager-system interaction.
    """
    
    def __init__(self):
        # Store performance in specific system roles
        self.system_performance: Dict[str, PlayerSystemPerformance] = {}
        
        # Chemistry between player pairs
        self.chemistry_metrics: Dict[str, SystemChemistryMetric] = {}
        
        # Manager system patterns
        self.manager_systems: Dict[str, Dict] = {}
        
        # Player preferred roles
        self.player_role_preferences: Dict[str, Dict[SystemRole, float]] = {}
        
        # System role requirements
        self.role_requirements: Dict[SystemRole, Dict] = {}
        self._initialize_role_requirements()
    
    def _initialize_role_requirements(self):
        """
        Define what each system role requires from the player.
        This is used to evaluate role fit.
        """
        self.role_requirements = {
            SystemRole.DEEP_LYRIST: {
                'min_pass_accuracy': 0.85,
                'min_progressive_passes': 4.0,
                'min_interceptions': 1.5,
                'key_traits': ['vision', 'composure', 'tackling'],
                'avg_xg_contribution': 0.05,
                'avg_xa_contribution': 0.15
            },
            SystemRole.BOX_TO_BOX: {
                'min_pass_accuracy': 0.80,
                'min_tackles': 2.0,
                'min_progressive_carries': 3.0,
                'key_traits': ['stamina', 'tackling', 'passing'],
                'avg_xg_contribution': 0.15,
                'avg_xa_contribution': 0.10
            },
            SystemRole.ADVANCED_PLAYMAKER: {
                'min_pass_accuracy': 0.82,
                'min_key_passes': 2.5,
                'min_xa': 0.20,
                'key_traits': ['vision', 'creativity', 'dribbling'],
                'avg_xg_contribution': 0.12,
                'avg_xa_contribution': 0.30
            },
            SystemRole.BALL_WASTER: {
                'min_pass_accuracy': 0.88,
                'min_progressive_passes': 5.0,
                'min_pass_completion_final_third': 0.75,
                'key_traits': ['passing', 'vision', 'positioning'],
                'avg_xg_contribution': 0.08,
                'avg_xa_contribution': 0.20
            },
            SystemRole.B2B_BOX_CRASH: {
                'min_pass_accuracy': 0.75,
                'min_tackles': 1.5,
                'min_shots': 2.0,
                'key_traits': ['stamina', 'finishing', 'timing'],
                'avg_xg_contribution': 0.25,
                'avg_xa_contribution': 0.08
            },
            SystemRole.HALF_SPACE_WINGER: {
                'min_dribbles': 2.0,
                'min_key_passes': 2.0,
                'min_crosses': 1.5,
                'key_traits': ['dribbling', 'crossing', 'creativity'],
                'avg_xg_contribution': 0.18,
                'avg_xa_contribution': 0.20
            },
            SystemRole.OVERLAPPING_FULLBACK: {
                'min_crosses': 2.5,
                'min_progressive_passes': 3.0,
                'min_tackles': 1.5,
                'key_traits': ['stamina', 'crossing', 'pace'],
                'avg_xg_contribution': 0.10,
                'avg_xa_contribution': 0.15
            },
            SystemRole.INVERTED_FULLBACK: {
                'min_pass_accuracy': 0.85,
                'min_progressive_passes': 4.0,
                'min_tackles': 1.8,
                'key_traits': ['passing', 'positioning', 'tackling'],
                'avg_xg_contribution': 0.06,
                'avg_xa_contribution': 0.12
            },
            SystemRole.BALL_PLAYING_CB: {
                'min_pass_accuracy': 0.85,
                'min_progressive_passes': 3.0,
                'min_clearances': 4.0,
                'key_traits': ['passing', 'positioning', 'heading'],
                'avg_xg_prevented': 0.35,
                'avg_xg_contribution': 0.02
            },
            SystemRole.AGGRESSIVE_CB: {
                'min_tackles': 2.5,
                'min_interceptions': 2.0,
                'min_blocks': 2.0,
                'key_traits': ['tackling', 'aggression', 'pace'],
                'avg_xg_prevented': 0.40,
                'avg_xg_contribution': 0.03
            },
            SystemRole.SWEEPER_KEEPER: {
                'min_pass_accuracy': 0.80,
                'min_crosses_claimed': 1.5,
                'min_sweeping_actions': 2.0,
                'key_traits': ['passing', 'positioning', 'decision_making'],
                'avg_xg_prevented': 0.45
            },
            SystemRole.POINTER_NINE: {
                'min_shots': 3.0,
                'min_shots_on_target': 1.2,
                'min_aerial_duels': 4.0,
                'key_traits': ['finishing', 'heading', 'strength'],
                'avg_xg_contribution': 0.35,
                'avg_xa_contribution': 0.05
            },
            SystemRole.INSIDE_FORWARD: {
                'min_dribbles': 2.5,
                'min_shots': 2.5,
                'min_key_passes': 1.5,
                'key_traits': ['dribbling', 'finishing', 'creativity'],
                'avg_xg_contribution': 0.30,
                'avg_xa_contribution': 0.15
            },
            SystemRole.FALSE_NINE: {
                'min_key_passes': 2.0,
                'min_dribbles': 2.0,
                'min_shots': 2.0,
                'key_traits': ['movement', 'creativity', 'finishing'],
                'avg_xg_contribution': 0.22,
                'avg_xa_contribution': 0.25
            }
        }
    
    def record_system_performance(
        self,
        player_id: str,
        manager_id: str,
        system_role: SystemRole,
        match_result: Dict,
        player_stats: Dict
    ):
        """
        Record how a player performed in a specific system role.
        Call this after each match.
        """
        key = f"{player_id}_{manager_id}_{system_role.value}"
        
        if key not in self.system_performance:
            self.system_performance[key] = PlayerSystemPerformance(
                player_id=player_id,
                manager_id=manager_id,
                system_role=system_role,
                matches_played=0,
                wins=0,
                draws=0,
                losses=0,
                role_specific_rating=0.5,
                avg_xg_contribution=0,
                avg_xa_contribution=0,
                avg_xg_prevented=0,
                pass_completion_in_system=0.7,
                progressive_action_rate=0,
                turnovers_per_90=0,
                recoveries_per_90=0,
                recent_form_trend=0,
                performance_variance=0
            )
        
        perf = self.system_performance[key]
        perf.matches_played += 1
        
        # Record result
        if match_result.get('win', False):
            perf.wins += 1
        elif match_result.get('draw', False):
            perf.draws += 1
        else:
            perf.losses += 1
        
        # Update rolling averages
        n = perf.matches_played
        perf.avg_xg_contribution = (
            (perf.avg_xg_contribution * (n - 1) + player_stats.get('xg', 0)) / n
        )
        perf.avg_xa_contribution = (
            (perf.avg_xa_contribution * (n - 1) + player_stats.get('xa', 0)) / n
        )
        perf.avg_xg_prevented = (
            (perf.avg_xg_prevented * (n - 1) + player_stats.get('xg_prevented', 0)) / n
        )
        perf.pass_completion_in_system = (
            (perf.pass_completion_in_system * (n - 1) + 
             player_stats.get('pass_completion', 0.7)) / n
        )
        perf.progressive_action_rate = (
            (perf.progressive_action_rate * (n - 1) + 
             player_stats.get('progressive_actions_per90', 0)) / n
        )
        perf.turnovers_per_90 = (
            (perf.turnovers_per_90 * (n - 1) + 
             player_stats.get('turnovers_per90', 0)) / n
        )
        perf.recoveries_per_90 = (
            (perf.recoveries_per_90 * (n - 1) + 
             player_stats
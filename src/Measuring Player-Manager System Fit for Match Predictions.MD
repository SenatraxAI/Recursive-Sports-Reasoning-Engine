# Measuring Player-Manager System Fit for Match Predictions

## The Core Concept: Player-System Compatibility Analysis

Your idea of measuring how players work within managerial systems is exactly the right direction, and it represents the natural evolution of your hierarchical model. The fundamental insight is that a team's predicted performance is not just the sum of individual player abilities, but the result of how well those players execute the manager's tactical system. Some players thrive in specific systems while struggling in others, and capturing this interaction is crucial for accurate predictions.

Think of it this way: if you take the same eleven players and put them under two different managers with different tactical philosophies, you will likely get different match results. The players haven't changed, but the system has, and that changes everything. Your model needs to capture this system-player interaction rather than treating player ability as a fixed quantity independent of context.

The inputs you mentioned form a complete tactical puzzle. The team provides the overall squad quality and depth. The roster tells you which specific players are available and their individual capabilities. The manager formation defines the tactical framework and role requirements. When you combine these three elements, you can calculate a "system fit score" that measures how well the available players match the manager's tactical requirements. A high system fit means the players are well-suited to execute the manager's plan. A low system fit means there will be tactical friction and the team may underperform relative to its raw talent level.

## How System Fit Affects Match Outcomes

System fit matters for several reasons, and understanding these mechanisms helps you build a more sophisticated model. First, role requirements matter because each position in a formation has specific tactical responsibilities. A number ten in a 4-2-3-1 is asked to link play between midfield and attack, find space between the lines, and create chances for the striker. The same player in a 4-3-3 as a left winger is asked to hug the touchline, beat defenders one-on-one, and deliver crosses. A player who excels at interior link-up play might have excellent ratings but poor system fit when asked to play as a wide winger.

Second, chemistry and combination play matter because football is a team sport where players need to anticipate each other's movements. When players have worked together under the same manager for months or years, they develop understanding of when to make runs, where to pass, and how to create space. A new signing might have excellent individual ability but lower system fit because they haven't yet developed these combinations with their new teammates. The system fit score should capture this by looking at how long players have been in the current system.

Third, tactical balance matters because every formation has inherent strengths and weaknesses. A 4-3-3 requires full-backs who can provide width and overlap. A 3-5-2 requires wing-backs with exceptional stamina and defensive discipline. If a team has poor options at full-back but plays a 4-3-3, the system fit will be low regardless of how good those full-backs are as individual players. Your model needs to detect these positional mismatches and adjust predictions accordingly.

## Building the Player-System Fit Calculator

Here's how you can implement a system that measures player fit within managerial formations and uses that to predict match outcomes:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

class Position(Enum):
    GK = "goalkeeper"
    CB = "center_back"
    LB = "left_back"
    RB = "right_back"
    CM = "central_midfield"
    CDM = "defensive_midfield"
    CAM = "attacking_midfield"
    LM = "left_midfield"
    RM = "right_midfield"
    LW = "left_winger"
    RW = "right_winger"
    ST = "striker"

class Formation(Enum):
    FOUR_THREE_THREE = "4-3-3"
    FOUR_TWO_THREE_ONE = "4-2-3-1"
    FOUR_FOUR_TWO = "4-4-2"
    FOUR_ONE_FOUR_ONE = "4-1-4-1"
    THREE_FIVE_TWO = "3-5-2"
    THREE_FOUR_THREE = "3-4-3"
    FOUR_THREE_TWO_ONE = "4-3-2-1"
    
    def get_positions(self) -> List[Position]:
        """Return the positions required for this formation."""
        position_maps = {
            Formation.FOUR_THREE_THREE: [
                Position.GK, Position.CB, Position.CB, 
                Position.LB, Position.RB,
                Position.CM, Position.CM, Position.CM,
                Position.LW, Position.ST, Position.RW
            ],
            Formation.FOUR_TWO_THREE_ONE: [
                Position.GK, Position.CB, Position.CB,
                Position.LB, Position.RB,
                Position.CDM, Position.CDM,
                Position.CAM, Position.CAM, Position.CAM,
                Position.ST
            ],
            Formation.FOUR_FOUR_TWO: [
                Position.GK, Position.CB, Position.CB,
                Position.LB, Position.RB,
                Position.LM, Position.CM, Position.CM, Position.RM,
                Position.ST, Position.ST
            ],
            Formation.THREE_FIVE_TWO: [
                Position.GK, Position.CB, Position.CB, Position.CB,
                Position.LM, Position.CM, Position.CM, Position.CM, Position.RM,
                Position.ST, Position.ST
            ],
            Formation.THREE_FOUR_THREE: [
                Position.GK, Position.CB, Position.CB, Position.CB,
                Position.LM, Position.CM, Position.CM, Position.CM, Position.RM,
                Position.LW, Position.ST, Position.RW
            ],
            Formation.FOUR_ONE_FOUR_ONE: [
                Position.GK, Position.CB, Position.CB,
                Position.LB, Position.RB,
                Position.CDM,
                Position.LM, Position.CAM, Position.CAM, Position.RM,
                Position.ST
            ],
            Formation.FOUR_THREE_TWO_ONE: [
                Position.GK, Position.CB, Position.CB,
                Position.LB, Position.RB,
                Position.CM, Position.CM, Position.CM,
                Position.CAM, Position.CAM,
                Position.ST
            ]
        }
        return position_maps.get(self, [])

@dataclass
class PlayerRole:
    """
    Defines what a player is capable of doing in different positions.
    This captures the player's versatility and role suitability.
    """
    player_id: str
    primary_position: Position
    secondary_positions: List[Position]
    
    # Attribute ratings on 0-1 scale
    technical_rating: float
    physical_rating: float
    tactical_rating: float
    mental_rating: float
    
    # Role-specific abilities
    role_ratings: Dict[str, float] = field(default_factory=dict)
    # Examples: "box_to_box": 0.8, "deep_playmaker": 0.6, "target_man": 0.7
    
    # System-specific performance history
    system_performance: Dict[str, Dict] = field(default_factory=dict)
    # Examples: {"4-3-3": {"matches": 20, "avg_rating": 0.75},
    #           "4-2-3-1": {"matches": 10, "avg_rating": 0.68}}

@dataclass
class ManagerTacticalSetup:
    """
    Defines a manager's tactical requirements for a specific formation.
    """
    manager_id: str
    formation: Formation
    
    # Role requirements (what the formation needs from each position)
    position_requirements: Dict[Position, Dict] = field(default_factory=dict)
    
    # Build examples of what each position should do
    # Position.CB: {"defensive_priority": 0.9, "passing_priority": 0.4, "aerial_priority": 0.8}
    # Position.CM: {"defensive_priority": 0.5, "passing_priority": 0.9, "progression_priority": 0.8}
    
    # Team-wide tactical parameters
    build_up_speed: float = 0.5  # 0-1, how fast to progress the ball
    pressing_intensity: float = 0.5  # 0-1, how aggressive the press is
    defensive_depth: float = 0.5  # 0-1, how high the defensive line is
    wide_play_focus: float = 0.5  # 0-1, importance of crossing/width
    
    # Historical success with this setup
    historical_win_rate: float = 0.5
    historical_goals_scored_avg: float = 1.5
    historical_goals_conceded_avg: float = 1.5

class PlayerSystemFitCalculator:
    """
    Calculates how well individual players fit into specific managerial systems.
    Combines player attributes with manager requirements to produce fit scores.
    """
    
    def __init__(self):
        self.player_database: Dict[str, PlayerRole] = {}
        self.manager_setups: Dict[str, List[ManagerTacticalSetup]] = {}
        self.match_results: List[Dict] = []
    
    def register_player(self, player: PlayerRole):
        """Add a player to the database."""
        self.player_database[player.player_id] = player
    
    def register_manager_setup(self, setup: ManagerTacticalSetup):
        """Register a manager's tactical setup for a specific formation."""
        key = f"{setup.manager_id}_{setup.formation.value}"
        if key not in self.manager_setups:
            self.manager_setups[key] = []
        self.manager_setups[key].append(setup)
    
    def calculate_position_fit(
        self,
        player_id: str,
        position: Position,
        manager_setup: ManagerTacticalSetup
    ) -> Dict[str, float]:
        """
        Calculate how well a player fits a specific position in a specific system.
        
        Returns a detailed fit analysis with multiple components.
        """
        player = self.player_database.get(player_id)
        if not player:
            return {
                'fit_score': 0.5,
                'position_score': 0.5,
                'role_score': 0.5,
                'system_score': 0.5,
                'confidence': 0.0,
                'notes': "Player not found in database"
            }
        
        # Check if player can actually play this position
        position_score = 0.0
        if position == player.primary_position:
            position_score = 1.0
        elif position in player.secondary_positions:
            position_score = 0.75
        elif player.tactical_rating > 0.7:
            # High tactical intelligence allows adaptation to similar positions
            position_score = 0.5
        else:
            position_score = 0.3
        
        # Get role requirements for this position
        requirements = manager_setup.position_requirements.get(position, {})
        
        # Calculate role fit based on player's role ratings
        role_score = 0.5
        if requirements:
            required_roles = requirements.get('preferred_roles', [])
            if required_roles:
                role_ratings = [
                    player.role_ratings.get(role, 0.5) 
                    for role in required_roles
                ]
                role_score = np.mean(role_ratings) if role_ratings else 0.5
        
        # Calculate system fit based on historical performance in similar systems
        system_key = f"{manager_setup.manager_id}_{manager_setup.formation.value}"
        system_history = player.system_performance.get(system_key, {})
        
        if system_history.get('matches', 0) >= 3:
            # Use actual performance in this system
            system_score = system_history.get('avg_rating', 0.5)
            confidence = min(system_history['matches'] / 15, 1.0)
        else:
            # Estimate based on similar systems or default to 0.5
            similar_score = self._find_similar_system_score(player, manager_setup)
            system_score = similar_score
            confidence = 0.3
        
        # Calculate attribute fit for specific requirements
        attribute_score = self._calculate_attribute_fit(
            player, 
            requirements
        )
        
        # Combine scores with weights
        # Position availability is binary (can they play there?)
        # Role fit measures if they have the right skills for the role
        # System fit measures if they've succeeded in this type of setup
        # Attribute fit measures if they meet specific tactical requirements
        
        fit_score = (
            position_score * 0.20 +
            role_score * 0.30 +
            system_score * 0.30 +
            attribute_score * 0.20
        )
        
        return {
            'fit_score': fit_score,
            'position_score': position_score,
            'role_score': role_score,
            'system_score': system_score,
            'attribute_score': attribute_score,
            'confidence': confidence,
            'notes': self._generate_fit_notes(position_score, role_score, system_score)
        }
    
    def _calculate_attribute_fit(
        self,
        player: PlayerRole,
        requirements: Dict
    ) -> float:
        """Calculate how well player attributes match requirements."""
        if not requirements:
            return 0.5
        
        scores = []
        
        # Map requirements to player attributes
        attr_mapping = {
            'technical_priority': player.technical_rating,
            'physical_priority': player.physical_rating,
            'tactical_priority': player.tactical_rating,
            'mental_priority': player.mental_rating,
        }
        
        for req_key, req_weight in requirements.items():
            if req_key in attr_mapping:
                player_attr = attr_mapping[req_key]
                # Requirement weight (0-1) * player attribute (0-1)
                scores.append(player_attr * req_weight)
        
        return np.mean(scores) if scores else 0.5
    
    def _find_similar_system_score(
        self,
        player: PlayerRole,
        manager_setup: ManagerTacticalSetup
    ) -> float:
        """Find player's average performance in similar systems."""
        similar_scores = []
        
        for system, history in player.system_performance.items():
            # Get the formation from the system key
            for formation in Formation:
                if formation.value in system:
                    # Compare tactical parameters
                    setup_list = self.manager_setups.get(system, [])
                    if setup_list:
                        similar_setup = setup_list[0]
                        
                        # Calculate similarity between systems
                        similarity = self._calculate_system_similarity(
                            manager_setup,
                            similar_setup
                        )
                        
                        if similarity > 0.6 and history.get('matches', 0) >= 2:
                            similar_scores.append(
                                history.get('avg_rating', 0.5) * similarity
                            )
        
        return np.mean(similar_scores) if similar_scores else 0.5
    
    def _calculate_system_similarity(
        self,
        setup1: ManagerTacticalSetup,
        setup2: ManagerTacticalSetup
    ) -> float:
        """Calculate how similar two tactical setups are."""
        params = [
            setup1.build_up_speed - setup2.build_up_speed,
            setup1.pressing_intensity - setup2.pressing_intensity,
            setup1.defensive_depth - setup2.defensive_depth,
            setup1.wide_play_focus - setup2.wide_play_focus
        ]
        
        # Inverse of squared differences (closer = higher similarity)
        max_diff = max(abs(p) for p in params) if params else 1
        return 1.0 - (max_diff / 2)  # Normalize to 0-1 range
    
    def _generate_fit_notes(
        self,
        position_score: float,
        role_score: float,
        system_score: float
    ) -> List[str]:
        """Generate human-readable notes about the fit assessment."""
        notes = []
        
        if position_score >= 0.9:
            notes.append("Primary position - excellent fit")
        elif position_score >= 0.7:
            notes.append("Secondary position - good fit")
        elif position_score >= 0.5:
            notes.append("Adaptation required - moderate fit")
        else:
            notes.append("Position unfamiliar - poor fit")
        
        if role_score >= 0.8:
            notes.append("Ideal role match")
        elif role_score >= 0.6:
            notes.append("Suitable role fit")
        
        if system_score >= 0.7:
            notes.append("Proven in similar system")
        elif system_score >= 0.5:
            notes.append("Limited system experience")
        
        return notes
    
    def calculate_team_system_fit(
        self,
        formation: Formation,
        player_ids: List[str],
        manager_setup: ManagerTacticalSetup
    ) -> Dict[str, float]:
        """
        Calculate the overall system fit for a team lineup.
        
        This aggregates individual player fits to give a team-level assessment.
        """
        required_positions = formation.get_positions()
        
        if len(player_ids) != len(required_positions):
            return {
                'overall_fit': 0.0,
                'fatal_error': "Player count doesn't match formation requirements",
                'squad_completeness': len(player_ids) / len(required_positions)
            }
        
        individual_fits = []
        position_by_position = {}
        
        for player_id, position in zip(player_ids, required_positions):
            fit = self.calculate_position_fit(player_id, position, manager_setup)
            individual_fits.append(fit['fit_score'])
            position_by_position[position.value] = {
                'player_id': player_id,
                'fit_score': fit['fit_score'],
                'confidence': fit['confidence'],
                'notes': fit['notes']
            }
        
        # Calculate overall fit with confidence weighting
        # More weight to positions where we have high confidence
        if individual_fits:
            # Weighted average by confidence
            weights = [f['confidence'] for f in position_by_position.values()]
            if sum(weights) > 0:
                overall_fit = sum(
                    fit * conf 
                    for fit, conf in zip(individual_fits, weights)
                ) / sum(weights)
            else:
                overall_fit = np.mean(individual_fits)
            
            # Also calculate raw average for comparison
            raw_average = np.mean(individual_fits)
            
            # Calculate balance - how evenly distributed is the fit?
            fit_variance = np.var(individual_fits)
            
            # Check for weak links
            weak_links = [
                pos for pos, data in position_by_position.items()
                if data['fit_score'] < 0.5
            ]
            
            return {
                'overall_fit': overall_fit,
                'raw_average': raw_average,
                'fit_variance': fit_variance,
                'squad_completeness': 1.0,
                'position_analysis': position_by_position,
                'weak_links': weak_links,
                'strongest_positions': self._find_strongest_positions(position_by_position, n=3),
                'weakest_positions': self._find_weakest_positions(position_by_position, n=3),
                'tactical_recommendations': self._generate_tactical_recommendations(
                    formation, position_by_position, manager_setup
                )
            }
        
        return {
            'overall_fit': 0.5,
            'fatal_error': "No player data available",
            'squad_completeness': 0.0
        }
    
    def _find_strongest_positions(
        self,
        position_analysis: Dict,
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """Find the n positions with highest fit scores."""
        sorted_positions = sorted(
            position_analysis.items(),
            key=lambda x: x[1]['fit_score'],
            reverse=True
        )
        return [(pos, data['fit_score']) for pos, data in sorted_positions[:n]]
    
    def _find_weakest_positions(
        self,
        position_analysis: Dict,
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """Find the n positions with lowest fit scores."""
        sorted_positions = sorted(
            position_analysis.items(),
            key=lambda x: x[1]['fit_score']
        )
        return [(pos, data['fit_score']) for pos, data in sorted_positions[:n]]
    
    def _generate_tactical_recommendations(
        self,
        formation: Formation,
        position_analysis: Dict,
        manager_setup: ManagerTacticalSetup
    ) -> List[str]:
        """Generate tactical recommendations based on the fit analysis."""
        recommendations = []
        
        weak_positions = self._find_weakest_positions(position_analysis, n=2)
        
        for pos, score in weak_positions:
            if score < 0.5:
                recommendations.append(
                    f"Weak fit at {pos} ({score:.2f}). Consider tactical adjustment "
                    f"or alternative player selection."
                )
        
        # Check for formation-specific issues
        if formation == Formation.FOUR_THREE_THREE:
            wide_positions = [p for p in position_analysis.keys() if 'winger' in p or 'wing' in p]
            wide_scores = [position_analysis[p]['fit_score'] for p in wide_positions]
            if wide_scores and min(wide_scores) < 0.6:
                recommendations.append(
                    "Wide positions show poor fit. 4-3-3 requires wingers who can "
                    "isolate and beat defenders one-on-one."
                )
        
        # Check defensive balance
        defensive_positions = [p for p in position_analysis.keys() if 'back' in p or 'midfield' in p]
        if defensive_positions:
            def_scores = [position_analysis[p]['fit_score'] for p in defensive_positions]
            if np.mean(def_scores) < 0.5:
                recommendations.append(
                    "Defensive positions underperforming. Team may struggle against "
                    "teams with strong attacking play through the middle."
                )
        
        return recommendations

class MatchPredictor:
    """
    Uses player-system fit analysis to predict match outcomes.
    """
    
    def __init__(self, fit_calculator: PlayerSystemFitCalculator):
        self.fit_calculator = fit_calculator
    
    def predict_match(
        self,
        home_team: Dict,
        away_team: Dict
    ) -> Dict[str, any]:
        """
        Predict the outcome of a match based on team compositions and systems.
        
        Args:
            home_team: {
                'team_id': str,
                'player_ids': List[str],
                'formation': Formation,
                'manager_id': str,
                'manager_setup': ManagerTacticalSetup
            }
            away_team: Same structure for away team
        
        Returns:
            Prediction dictionary with probabilities and analysis
        """
        # Calculate system fit for both teams
        home_fit = self.fit_calculator.calculate_team_system_fit(
            home_team['formation'],
            home_team['player_ids'],
            home_team['manager_setup']
        )
        
        away_fit = self.fit_calculator.calculate_team_system_fit(
            away_team['formation'],
            away_team['player_ids'],
            away_team['manager_setup']
        )
        
        # Calculate relative team strength
        # Combine raw talent (would come from player database) with system fit
        home_talent = self._calculate_squad_talent(home_team['player_ids'])
        away_talent = self._calculate_squad_talent(away_team['player_ids'])
        
        # Adjust talent by system fit
        home_effective_strength = home_talent * (0.7 + 0.3 * home_fit['overall_fit'])
        away_effective_strength = away_talent * (0.7 + 0.3 * away_fit['overall_fit'])
        
        # Calculate expected goals based on strength comparison
        home_xg, away_xg = self._calculate_expected_goals(
            home_effective_strength,
            away_effective_strength
        )
        
        # Calculate win probabilities using a simple model
        # In production, this would use a more sophisticated model
        strength_ratio = home_effective_strength / max(away_effective_strength, 0.01)
        
        if strength_ratio > 1.2:
            home_win_prob = 0.45 + (strength_ratio - 1.2) * 0.3
            draw_prob = 0.30
            away_win_prob = 1 - home_win_prob - draw_prob
        elif strength_ratio > 0.9:
            home_win_prob = 0.35
            draw_prob = 0.35
            away_win_prob = 0.30
        elif strength_ratio > 0.7:
            home_win_prob = 0.30
            draw_prob = 0.35
            away_win_prob = 0.35
        else:
            home_win_prob = 0.25
            draw_prob = 0.35
            away_win_prob = 0.40
        
        # Normalize probabilities
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Generate match narrative
        narrative = self._generate_match_narrative(
            home_team, away_team, home_fit, away_fit,
            home_xg, away_xg, home_win_prob
        )
        
        return {
            'home_win_probability': home_win_prob,
            'draw_probability': draw_prob,
            'away_win_probability': away_win_prob,
            'home_expected_goals': home_xg,
            'away_expected_goals': away_xg,
            'home_system_fit': home_fit['overall_fit'],
            'away_system_fit': away_fit['overall_fit'],
            'home_talent_score': home_talent,
            'away_talent_score': away_talent,
            'home_effective_strength': home_effective_strength,
            'away_effective_strength': away_effective_strength,
            'home_position_analysis': home_fit['position_analysis'],
            'away_position_analysis': away_fit['position_analysis'],
            'weak_links': {
                'home': home_fit['weak_links'],
                'away': away_fit['weak_links']
            },
            'tactical_insights': self._generate_tactical_insights(home_fit, away_fit),
            'match_narrative': narrative,
            'recommended_bets': self._generate_betting_recommendations(
                home_win_prob, draw_prob, away_win_prob, home_xg, away_xg
            )
        }
    
    def _calculate_squad_talent(self, player_ids: List[str]) -> float:
        """Calculate raw talent level of a squad."""
        talents = []
        for player_id in player_ids:
            player = self.fit_calculator.player_database.get(player_id)
            if player:
                # Overall talent is weighted average of attributes
                talent = (
                    player.technical_rating * 0.25 +
                    player.physical_rating * 0.25 +
                    player.tactical_rating * 0.25 +
                    player.mental_rating * 0.25
                )
                talents.append(talent)
        
        return np.mean(talents) if talents else 0.5
    
    def _calculate_expected_goals(
        self,
        home_strength: float,
        away_strength: float
    ) -> Tuple[float, float]:
        """Calculate expected goals based on team strengths."""
        avg_xg = 1.35  # Premier League average
        
        # Home advantage adds approximately 0.3 xG
        home_advantage = 0.3
        
        # Calculate relative strength
        total_strength = home_strength + away_strength
        if total_strength == 0:
            return avg_xg + home_advantage, avg_xg
        
        home_share = home_strength / total_strength
        away_share = away_strength / total_strength
        
        # Scale from average
        home_xg = (avg_xg + home_advantage) * (home_share / 0.5)
        away_xg = avg_xg * (away_share / 0.5)
        
        return home_xg, away_xg
    
    def _generate_match_narrative(
        self,
        home_team: Dict,
        away_team: Dict,
        home_fit: Dict,
        away_fit: Dict,
        home_xg: float,
        away_xg: float,
        home_win_prob: float
    ) -> str:
        """Generate a textual narrative describing the expected match."""
        narratives = []
        
        # Overall expectation
        if home_win_prob > 0.45:
            narratives.append(
                f"{home_team['team_id']} enter as favorites with a {home_win_prob:.0%} "
                f"chance of victory."
            )
        elif home_win_prob > 0.35:
            narratives.append(
                f"This looks like a closely contested match with {home_team['team_id']} "
                f"slightly favored."
            )
        else:
            narratives.append(
                f"{away_team['team_id']} may have the edge in this matchup."
            )
        
        # System fit analysis
        if home_fit['overall_fit'] > 0.7:
            narratives.append(
                f"{home_team['team_id']}'s players show excellent understanding of "
                f"their manager's tactical setup."
            )
        elif home_fit['overall_fit'] < 0.5:
            narratives.append(
                f"Concerns for {home_team['team_id']} as players may struggle to "
                f"execute the tactical plan."
            )
        
        # Key battles
        if home_fit['weak_links'] and away_fit['strongest_positions']:
            weak_pos = home_fit['weak_links'][0][0] if home_fit['weak_links'] else 'several positions'
            strong_pos = away_fit['strongest_positions'][0][0] if away_fit['strongest_positions'] else 'key areas'
            narratives.append(
                f"{away_team['team_id']} could exploit {weak_pos} while "
                f"{home_team['team_id']} looks strongest in {strong_pos}."
            )
        
        return " ".join(narratives)
    
    def _generate_tactical_insights(
        self,
        home_fit: Dict,
        away_fit: Dict
    ) -> List[str]:
        """Generate tactical insights for the match."""
        insights = []
        
        # Compare system fits
        fit_diff = home_fit['overall_fit'] - away_fit['overall_fit']
        if abs(fit_diff) > 0.15:
            better_fit = "home" if fit_diff > 0 else "away"
            insights.append(
                f"The {better_fit} team shows better tactical cohesion, which could "
                f"be decisive in a close match."
            )
        
        # Weak link exploitation
        if home_fit['weak_links'] and away_fit['strongest_positions']:
            insights.append(
                "Tactical mismatch detected: one team's weakness aligns with "
                "the other's strength."
            )
        
        # Formation clash
        home_def_variance = home_fit.get('fit_variance', 0)
        away_def_variance = away_fit.get('fit_variance', 0)
        if home_def_variance > 0.05 and away_def_variance > 0.05:
            insights.append(
                "Both teams show uneven squad composition, suggesting potential "
                "for tactical surprises or formation changes during the match."
            )
        
        return insights
    
    def _generate_betting_recommendations(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_xg: float,
        away_xg: float
    ) -> Dict[str, any]:
        """Generate betting recommendations based on the prediction."""
        recommendations = []
        
        # Value betting: if model probability exceeds implied bookmaker probability
        # This is a simplified example - real implementation would compare to actual odds
        implied_home = 1 / 3  # Hypothetical implied probability
        if home_win_prob > implied_home + 0.05:
            recommendations.append({
                'selection': 'Home Win',
                'model_prob': home_win_prob,
                'implied_prob': implied_home,
                'edge': home_win_prob - implied_home,
                'stake_recommendation': 'Medium'
            })
        
        # Total goals analysis
        total_xg = home_xg + away_xg
        if total_xg > 2.8:
            recommendations.append({
                'selection': 'Over 2.5 Goals',
                'expected_total': total_xg,
                'confidence': 'Medium-High'
            })
        elif total_xg < 2.2:
            recommendations.append({
                'selection': 'Under 2.5 Goals',
                'expected_total': total_xg,
                'confidence': 'Medium-High'
            })
        
        return {
            'recommendations': recommendations,
            'model_confidence': 'Medium',
            'notes': "Always gamble responsibly. These predictions are models, not guarantees."
        }
```

## How to Use This System for Match Predictions

The system takes the inputs you mentioned and produces detailed match predictions. When you want to predict a specific match, you would construct the input data as follows:

For the home team, you provide the team identifier, the list of player IDs in the starting lineup, the formation the manager will use, the manager's identifier, and the tactical setup object that defines the manager's specific requirements for that formation. You do the same for the away team. The system then calculates how well each player fits their assigned position in the formation, aggregates these fits into a team-level assessment, compares the two teams, and produces probabilities for different outcomes.

The key insight is that this approach captures the interaction between players and systems that your original model missed. Instead of just adding up player abilities, it measures how well those abilities translate into the specific tactical context imposed by the manager. A team with lower raw talent but higher system fit might outperform a team with higher raw talent but poor system fit, and your model will capture this effect.
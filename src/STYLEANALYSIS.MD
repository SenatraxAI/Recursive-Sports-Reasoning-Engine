# Incorporating Styles of Play Into Your Player Analysis

## Why Styles of Play Matter for Player Ratings

Your intuition here is exactly right, and this represents a significant sophistication upgrade for your model. The core problem you've already identified is that player performance is not entirely about individual ability, but about how that ability interacts with the team's tactical system. A world-class player can look mediocre in the wrong system, while a solid player can look exceptional when the system suits their strengths.

When you calculate a Player Value Rating based purely on statistics like xG, xA, and passing completion, you're capturing only part of the picture. You're measuring what a player did, but not necessarily what they're capable of doing in different contexts. If Kevin De Bruyne plays in a team that dominates possession and creates 2.5 xG per game, his xG and xA numbers will naturally be higher than if he plays in a team that sits deep and creates 0.8 xG per game on the counter-attack. The same player has different statistical outputs based on team context, and your model needs to account for this.

The style-of-play analysis also directly addresses the transfer bias problem you raised earlier. When a player moves to a new team, you're not just adding their historical stats to a new squad; you're placing them in a different tactical context. If a player transfers from a team that plays 4-3-3 possession football to a team that plays 3-5-2 counter-attacking football, their statistical profile will likely change dramatically. Your model should be able to predict how their numbers might change based on the tactical fit, rather than just taking their historical average as a fixed value.

## Defining and Categorizing Styles of Play

To incorporate styles of play into your model, you first need a framework for categorizing different approaches to football. There are several dimensions along which you can classify team styles, and each dimension affects which players thrive and which struggle.

The possession spectrum measures how much a team values keeping the ball versus accepting lower possession and hitting on the counter. Teams at the high-possession end like Manchester City and Barcelona under Guardiola create many passing opportunities and high xG through sustained pressure. Teams at the low-possession end like Atlético Madrid under Simeone or defensive-minded Premier League teams often have lower xG but higher conversion rates because they get chances on the break. A player's value should be adjusted based on where their team sits on this spectrum. A creative midfielder with excellent passing might be worth more to a high-possession team, while a direct runner with pace might be worth more to a counter-attacking team.

The pressing intensity dimension captures how aggressively a team attempts to win the ball back after losing possession. Teams with high PPDA (Passes Per Defensive Action) numbers are passive and allow opponents time on the ball. Teams with low PPDA are intense pressers who try to win the ball back quickly in dangerous areas. This affects player ratings because some players excel at pressing and their value increases in high-intensity systems, while other players are more positionally disciplined and perform better in passive defensive structures. A player with high successful pressure rates will add more value to a high-pressing team than to a passive one.

The tempo and transition dimension measures whether a team prefers slow, methodical build-up play or quick vertical transitions. Some teams move the ball quickly through thirds and look for immediate chances. Other teams are more patient, cycling the ball to draw opponents out of position before creating openings. This dimension particularly affects players who are good in one-v-one situations versus players who are good at finding space in tight structures. A quick winger might add more value to a quick-transition team, while a technical number ten might add more value to a patient build-up team.

The width and crossing dimension captures how teams use the flanks versus the central areas. Teams that cross frequently have different statistical profiles for their full-backs and wingers compared to teams that focus on central penetration. A player who excels at delivering crosses will add more value to a wide-playing team, while a player who excels at interior carries will add more value to a central-focused team.

## Building a Style-Adjusted Player Rating System

Once you've categorized teams along these dimensions, you can build a style-adjusted rating system that normalizes player statistics based on their team's tactical context. The goal is to estimate what a player's numbers would look like if they played in an average tactical environment, which makes cross-team and cross-league comparisons more meaningful.

The basic approach is to calculate league-average statistics for each tactical dimension, then adjust individual player stats based on how their team's style compares to the league average. If the league-average possession team has 55% possession and creates 1.5 xG per game, but your player's team has 70% possession and creates 2.0 xG per game, you would adjust the player's xG downward to estimate what they might produce in an average-possession environment. This adjusted value becomes their style-normalized rating.

Here's a practical implementation approach for your ImprovedPlayerLayer:

```python
class StyleAdjustedPlayerLayer(ImprovedPlayerLayer):
    """
    Extended player layer that accounts for team style of play.
    Normalizes player stats based on tactical context.
    """
    
    def __init__(self, league_reference_stats=None, **kwargs):
        super().__init__(**kwargs)
        
        # League-average statistics to normalize against
        # These would be calculated from your historical data
        self.league_reference = league_reference_stats or {
            'avg_possession_pct': 50.0,
            'avg_xg_per_match': 1.35,
            'avg_xga_per_match': 1.35,
            'avg_ppda': 15.0,  # Lower = more aggressive pressing
            'avg_crosses_per_match': 25.0,
            'avg_progressive_passes_per_match': 18.0,
            'avg_tempo': 1.0,  # Normalized transition speed
            'avg_line_height': 35.0  # Average defensive line position
        }
        
        # Store team style profiles
        self.team_style_profiles: Dict[str, Dict] = {}
    
    def set_team_style(self, team_id: str, style_profile: Dict):
        """
        Define the tactical style for a team.
        
        Args:
            team_id: Unique identifier for the team
            style_profile: Dictionary with tactical characteristics
        """
        self.team_style_profiles[team_id] = style_profile
    
    def _calculate_style_adjustment_factor(
        self,
        team_style: Dict,
        stat_type: str
    ) -> float:
        """
        Calculate how much to adjust a player's stats based on team style.
        
        Returns a multiplier: >1.0 means the style boosts this stat,
        <1.0 means the style suppresses this stat.
        """
        if stat_type in ['xg', 'xa', 'goals', 'assists']:
            # Attacking stats are boosted by high possession and high tempo
            possession_factor = team_style.get('possession_pct', 50) / self.league_reference['avg_possession_pct']
            tempo_factor = team_style.get('tempo', 1.0)
            
            # Teams with more possession create more chances
            # Teams with higher tempo have more transition opportunities
            return (possession_factor * 0.6 + tempo_factor * 0.4)
        
        elif stat_type in ['passes_completed', 'pass_completion_rate']:
            # Passing stats are boosted by possession and suppressed by pressing
            possession_factor = team_style.get('possession_pct', 50) / self.league_reference['avg_possession_pct']
            pressing_factor = self.league_reference['avg_ppda'] / max(team_style.get('ppda', 15), 1)
            
            # More possession = more passes = higher completion counts
            # Facing more pressing = lower completion rate
            return (possession_factor * 0.7 + pressing_factor * 0.3)
        
        elif stat_type in ['pressures', 'successful_pressures']:
            # Pressing stats are boosted by high PPDA (aggressive pressing)
            pressing_intensity = self.league_reference['avg_ppda'] / max(team_style.get('ppda', 15), 1)
            return pressing_intensity
        
        elif stat_type in ['progressive_passes', 'progressive_carries']:
            # Progressive actions are boosted by high tempo and central play
            tempo_factor = team_style.get('tempo', 1.0)
            central_factor = 1.0 - (team_style.get('width_focus', 0.5) * 0.3)
            
            return (tempo_factor * 0.6 + central_factor * 0.4)
        
        elif stat_type in ['tackles', 'interceptions', 'blocks']:
            # Defensive stats are boosted by low possession (more defending)
            # and by high pressing intensity
            possession_factor = self.league_reference['avg_possession_pct'] / max(team_style.get('possession_pct', 50), 1)
            pressing_factor = self.league_reference['avg_ppda'] / max(team_style.get('ppda', 15), 1)
            
            return (possession_factor * 0.5 + pressing_factor * 0.5)
        
        else:
            return 1.0  # No adjustment for unknown stats
    
    def calculate_style_adjusted_rating(
        self,
        player_id: str,
        team_id: str,
        current_date: datetime = None
    ) -> Dict[str, float]:
        """
        Calculate player rating adjusted for team style of play.
        
        This answers: "How good is this player in general, regardless of
        what team they're playing for?"
        """
        # Get base rating from parent class
        base_rating = self.calculate_player_value_rating(player_id, current_date)
        
        # Get team style profile
        team_style = self.team_style_profiles.get(team_id, {})
        
        if not team_style:
            # No style data available, return base rating
            return {**base_rating, 'style_adjusted': False}
        
        # Calculate adjustment factors
        attack_adjustment = self._calculate_style_adjustment_factor(team_style, 'xg')
        pass_adjustment = self._calculate_style_adjustment_factor(team_style, 'pass_completion_rate')
        press_adjustment = self._calculate_style_adjustment_factor(team_style, 'pressures')
        defend_adjustment = self._calculate_style_adjustment_factor(team_style, 'tackles')
        
        # Apply adjustments to create normalized ratings
        adjusted_attacking = base_rating['attacking_rating'] / attack_adjustment
        adjusted_possession = base_rating['possession_rating'] / pass_adjustment
        adjusted_defensive = base_rating['defensive_rating'] / defend_adjustment
        
        # Re-calculate overall with adjusted components
        adjusted_overall = (
            0.45 * adjusted_attacking +
            0.30 * adjusted_defensive +
            0.25 * adjusted_possession
        )
        
        # Calculate style fit score: how well does this player match their team's style?
        style_fit_score = self._calculate_style_fit(player_id, team_style)
        
        return {
            # Original base ratings
            'overall_rating': base_rating['overall_rating'],
            'attacking_rating': base_rating['attacking_rating'],
            'defensive_rating': base_rating['defensive_rating'],
            'possession_rating': base_rating['possession_rating'],
            'form_trend': base_rating['form_trend'],
            'reliability_score': base_rating['reliability_score'],
            'games_analyzed': base_rating['games_analyzed'],
            
            # Style-adjusted ratings (normalized to average team)
            'style_adjusted_overall': adjusted_overall,
            'style_adjusted_attacking': adjusted_attacking,
            'style_adjusted_defensive': adjusted_defensive,
            'style_adjusted_possession': adjusted_possession,
            
            # Style fit analysis
            'style_fit_score': style_fit_score,
            'best_system_fit': self._predict_best_system_fit(player_id),
            
            # Metadata
            'style_adjusted': True,
            'team_style_id': team_id
        }
    
    def _calculate_style_fit(
        self,
        player_id: str,
        team_style: Dict
    ) -> float:
        """
        Calculate how well a player's profile fits their team's style.
        
        Returns 0-1 score where 1.0 means perfect fit.
        """
        player_base = self.calculate_player_value_rating(player_id)
        
        # Get player's historical style context (average of their recent teams)
        # For simplicity, assume we know the player's preferred stats
        player_preferences = {
            'possession_weight': 0.5,  # How much they benefit from possession
            'pressing_weight': 0.5,     # How much they benefit from pressing
            'tempo_weight': 0.5,        # How much they benefit from tempo
            'width_weight': 0.5         # How much they benefit from width
        }
        
        # Calculate fit based on alignment between player preferences and team style
        possession_fit = 1.0 - abs(player_preferences['possession_weight'] - 
                                   (team_style.get('possession_pct', 50) / 100))
        pressing_fit = 1.0 - abs(player_preferences['pressing_weight'] - 
                                  (1 - min(team_style.get('ppda', 15) / 30, 1)))
        tempo_fit = 1.0 - abs(player_preferences['tempo_weight'] - 
                              (team_style.get('tempo', 1.0) / 2))
        
        # Weighted combination
        fit_score = (
            possession_fit * 0.3 +
            pressing_fit * 0.4 +
            tempo_fit * 0.3
        )
        
        return fit_score
    
    def _predict_best_system_fit(self, player_id: str) -> Dict:
        """
        Predict which tactical system would maximize this player's output.
        """
        player_base = self.calculate_player_value_rating(player_id)
        
        # Analyze player's relative strengths
        attack_ratio = player_base['attacking_rating'] / max(player_base['overall_rating'], 0.01)
        defend_ratio = player_base['defensive_rating'] / max(player_base['overall_rating'], 0.01)
        poss_ratio = player_base['possession_rating'] / max(player_base['overall_rating'], 0.01)
        
        best_systems = []
        
        # High attacking, lower defensive: suits counter-attacking teams
        if attack_ratio > 0.6 and defend_ratio < 0.4:
            best_systems.append({
                'system': 'Counter-Attack',
                'rationale': 'High attacking output with limited defensive duties',
                'expected_boost': '+15-20% xG contribution'
            })
        
        # High possession stats: suits possession-based teams
        if poss_ratio > 0.6 and attack_ratio > 0.4:
            best_systems.append({
                'system': 'Possession Play',
                'rationale': 'Excellent technical ability suits ball retention',
                'expected_boost': '+10-15% passing efficiency'
            })
        
        # High defensive stats: suits low-block teams
        if defend_ratio > 0.6 and attack_ratio < 0.4:
            best_systems.append({
                'system': 'Low Block / Counter-Press',
                'rationale': 'Strong defensive metrics suit defensive organization',
                'expected_boost': '+20-25% defensive contributions'
            })
        
        # Balanced profile: suits various systems
        if len(best_systems) == 0:
            best_systems.append({
                'system': 'System Agnostic',
                'rationale': 'Balanced profile adapts to multiple tactical contexts',
                'expected_boost': 'Baseline performance across systems'
            })
        
        return {
            'primary_recommendation': best_systems[0]['system'],
            'all_systems': best_systems,
            'attack_defense_balance': f"{attack_ratio:.1%} attack / {defend_ratio:.1%} defense"
        }
```

## Practical Application for Transfer Predictions

The style-adjusted rating system becomes particularly powerful when analyzing transfers. When a player moves from one team to another, you can estimate their expected performance in the new context by comparing the team styles and applying appropriate adjustments.

For example, imagine a player transferring from a high-possession team (65% average possession, low pressing intensity) to a counter-attacking team (40% average possession, high pressing intensity). Their raw statistics from the first team would likely overestimate their value in the second context. The style adjustment would tell you: "This player averaged 0.35 xG per match, but in a counter-attacking system with similar quality of chances, they might average 0.25 xG per match." The difference of 0.10 xG is the style adjustment penalty.

This analysis helps you make better predictions about how transfers will affect team strength. If a team loses a player who was perfectly suited to their style and replaces them with a player who is a poor tactical fit, the team might decline more than the raw player ratings would suggest. Conversely, if a team replaces a misfit player with a perfect tactical fit, they might improve even if the new player has lower raw ability.

## Building the Team Style Database

To make this system work, you need to systematically categorize teams by their style of play. You can do this manually based on known tactical approaches, or you can calculate style profiles automatically from match data.

For manual categorization, you would assign each team a profile based on their known tactical approach. Manchester City under Guardiola would have high possession, high tempo, and moderate pressing. Liverpool under Klopp would have moderate possession, very high tempo, and very high pressing intensity. Brentford under Frank would have moderate possession, moderate tempo, and moderate pressing. You can create these profiles by watching matches and noting tactical patterns, or by consulting tactical analysis from reputable sources.

For automatic categorization, you can calculate style profiles from the match data itself. High average possession percentages indicate possession-oriented teams. Low PPDA numbers indicate aggressive pressing teams. High pass completion rates in the final third indicate patient build-up teams. The advantage of automatic calculation is that you can track how team styles evolve over a season, which is important because managers often adjust their approach based on results and opponent analysis.

The combination of manual expert knowledge and automatic statistical calculation gives you the best of both worlds. You use expert knowledge for teams you understand well, and you use automatic calculation for teams you haven't studied in detail or for tracking tactical changes over time.
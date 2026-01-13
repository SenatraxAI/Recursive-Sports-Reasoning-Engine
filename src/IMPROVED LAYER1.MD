
# Understanding the ImprovedPlayerLayer: Fixing Your Premier League Prediction Model

## Why Your Original Level 1 Model Failed

In your original approach, Level 1 attempted to predict how each individual player would perform in a specific upcoming match. For example, you might have tried to predict that Erling Haaland would score 0.85 expected goals in the next Manchester City match. This approach has a fundamental problem that explains the negative R² score you encountered.

The issue is that predicting single-match player performance is extremely noisy. A player might score 2 goals one match and 0 the next, not because their underlying ability changed, but due to random variations in shot quality, opponent defensive positioning, goalkeeper saves, and countless other factors. When you train a model on this kind of noisy data, it struggles to find any meaningful patterns. The model essentially learns that there's no reliable signal to exploit, resulting in predictions that are worse than simply guessing the average. This is why you got negative R², which means your model performed worse than a naive baseline that just predicted the average value for every player.

Additionally, this approach doesn't align well with the injury matrix you wanted to build. If Level 1 is trying to predict match-by-match performance, it's not naturally capturing the cumulative value a player brings to their team over time. The injury matrix needs to know "how much would Manchester City miss Kevin De Bruyne if he were injured," which requires understanding De Bruyne's overall contribution to the team, not just his predicted performance in one specific match.

## The Rolling Window Concept: Measuring Player Value Over Time

The ImprovedPlayerLayer takes a completely different approach. Instead of trying to predict how a player will perform in their next match, it calculates a "Player Value Rating" based on what they have actually done in recent matches. Think of it like a credit score for football players, where you're aggregating their historical performance into a single, stable metric.

The rolling window is the timeframe you use to calculate this rating. If you set a rolling window of 10 matches, the system will look at a player's last 10 games and use those results to calculate their current value. This smooths out the noise from individual matches. If Haaland scored 2 goals in one match and 0 in the next, his 10-match rolling average will show his true scoring ability rather than being thrown off by random variation.

This approach directly supports your injury matrix because you're now calculating the actual contribution each player makes. When a player is injured, you simply subtract their value from the team's total strength. If Kevin De Bruyne has a Player Value Rating of 8.7 (on whatever scale you're using) and he's injured, you reduce Manchester City's attacking strength by 8.7 before making your prediction. This is exactly the kind of granular, player-level tracking that makes your hierarchical model powerful.

## How Exponential Decay Makes Your Ratings More Accurate

Not all matches in your rolling window should count equally. A player's performance from last weekend is more relevant than their performance from three months ago. Exponential decay is a mathematical way of giving more weight to recent matches while gradually reducing the importance of older matches.

The formula works like this: each match in your history gets multiplied by a weight that decreases exponentially the further back in time that match occurred. If you set a decay rate of 0.1, a match played this week gets a weight of 1.0, last week's match gets a weight of approximately 0.90, two weeks ago gets approximately 0.81, three weeks ago gets approximately 0.73, and so on. By the time you reach 10 matches back, that old performance is barely influencing the current rating at all.

This creates a "Form Rating" that captures not just how good a player is in general, but how well they're playing right now. A player recovering from injury might have strong historical numbers but poor recent form, and exponential decay will naturally reflect this. Similarly, a player who has been on an incredible scoring run will see their rating rise quickly because those recent performances are heavily weighted.

## The Complete ImprovedPlayerLayer Code Explained

Here's the full implementation with detailed comments explaining each component:

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class PlayerMatchPerformance:
    """
    This dataclass stores the raw performance data for a single match.
    Think of it as a snapshot of everything a player did in one game.
    """
    player_id: str
    match_id: str
    match_date: datetime
    
    # Attacking metrics - these show goal contribution
    goals: float
    assists: float
    xg: float          # Expected goals - quality of shots taken
    xa: float          # Expected assists - quality of key passes
    shots: float
    shots_on_target: float
    
    # Chance creation - how well they create opportunities
    key_passes: float
    progressive_passes: float
    progressive_carries: float
    
    # Possession and passing - technical ability
    pass_completion_rate: float
    passes_completed: float
    passes_attempted: float
    progressive_passes_completed: float
    
    # Defensive metrics - contribution without the ball
    tackles: float
    interceptions: float
    blocks: float
    clearances: float
    
    # Pressing and work rate
    pressures: float
    successful_pressures: float
    duels_won: float
    
    # Misc useful stats
    minutes_played: float
    touches: float
    carry_progress_yards: float

class ImprovedPlayerLayer:
    """
    This class replaces your original Level 1 model. Instead of predicting
    single-match performance (which proved too noisy), it calculates a stable
    Player Value Rating based on rolling historical performance with exponential decay.
    
    The key insight: We don't need to predict the future perfectly; we need to
    accurately measure what each player has been contributing lately.
    """
    
    def __init__(
        self,
        lookback_matches: int = 10,
        decay_rate: float = 0.15,
        min_minutes_threshold: float = 270,
        stats_weight_attacking: Dict[str, float] = None,
        stats_weight_defensive: Dict[str, float] = None,
        stats_weight_possession: Dict[str, float] = None
    ):
        """
        Configuration for the player layer.
        
        Args:
            lookback_matches: How many recent matches to consider (default 10)
            decay_rate: How fast older matches lose importance (0.15 means ~15% decay per match)
            min_minutes_threshold: Minimum minutes needed to be considered "active" (270 = 3 full matches)
            stats_weight_*: Dictionaries defining how much each stat type matters
        """
        self.lookback_matches = lookback_matches
        self.decay_rate = decay_rate
        self.min_minutes_threshold = min_minutes_threshold
        
        # Default weights if not provided - you can tune these based on what predicts wins
        self.stats_weight_attacking = stats_weight_attacking or {
            'goals': 2.0,           # Goals are most valuable
            'xg': 1.5,              # Shot quality matters
            'assists': 1.8,
            'xa': 1.2,              # Chance creation matters
            'shots': 0.3,
            'shots_on_target': 0.5,
            'key_passes': 1.0,
            'progressive_passes': 0.8,
            'progressive_carries': 0.7
        }
        
        self.stats_weight_defensive = stats_weight_defensive or {
            'tackles': 0.8,
            'interceptions': 0.9,
            'blocks': 0.6,
            'clearances': 0.4,
            'pressures': 0.5,
            'successful_pressures': 0.7,
            'duels_won': 0.6
        }
        
        self.stats_weight_possession = stats_weight_possession or {
            'pass_completion_rate': 0.5,
            'passes_completed': 0.3,
            'progressive_passes_completed': 0.7,
            'touches': 0.2,
            'carry_progress_yards': 0.6
        }
        
        # Cache for player history - in production you'd use a real database
        self.player_history_cache: Dict[str, List[PlayerMatchPerformance]] = {}
    
    def add_player_match_data(self, performance: PlayerMatchPerformance):
        """
        Add a new match performance to a player's history.
        Call this whenever you process a new match result.
        """
        if performance.player_id not in self.player_history_cache:
            self.player_history_cache[performance.player_id] = []
        
        self.player_history_cache[performance.player_id].append(performance)
        
        # Sort by date so oldest matches are first
        self.player_history_cache[performance.player_id].sort(
            key=lambda x: x.match_date
        )
    
    def _calculate_decay_weights(self, num_matches: int) -> np.ndarray:
        """
        Create an array of exponentially decaying weights.
        
        Example with decay_rate=0.15 and 5 matches:
        weights = [e^0, e^-0.15, e^-0.30, e^-0.45, e^-0.60]
               = [1.00, 0.86,   0.74,   0.64,   0.55]
        """
        if num_matches == 0:
            return np.array([])
        
        # Create array of exponents: 0, -decay, -2*decay, -(n-1)*decay
        exponents = -self.decay_rate * np.arange(num_matches)
        weights = np.exp(exponents)
        
        # Normalize so weights sum to 1 (makes comparison easier)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _normalize_stat(self, value: float, stat_type: str) -> float:
        """
        Normalize raw statistics to a common scale.
        
        This is crucial because 'goals' might be 0-3 per match while 'passes_completed'
        might be 20-80. Without normalization, high-volume stats would dominate.
        
        Args:
            value: The raw statistic value
            stat_type: Which stat we're normalizing
            
        Returns:
            Normalized value on a roughly 0-1 scale
        """
        # These thresholds are examples based on Premier League averages
        # In production, you'd calculate these from your actual data
        normalization_thresholds = {
            'goals': (0, 1.5),           # Most players score 0-1 per match
            'xg': (0, 0.8),
            'assists': (0, 0.5),
            'xa': (0, 0.4),
            'shots': (0, 4),
            'shots_on_target': (0, 2),
            'key_passes': (0, 3),
            'progressive_passes': (0, 5),
            'progressive_carries': (0, 6),
            'tackles': (0, 3),
            'interceptions': (0, 2.5),
            'blocks': (0, 3),
            'clearances': (0, 4),
            'pressures': (0, 15),
            'successful_pressures': (0, 8),
            'duels_won': (0, 8),
            'pass_completion_rate': (0.5, 0.95),  # Percentage
            'passes_completed': (10, 60),
            'progressive_passes_completed': (2, 8),
            'touches': (20, 80),
            'carry_progress_yards': (10, 100),
            'minutes_played': (0, 90)
        }
        
        if stat_type not in normalization_thresholds:
            return value  # No normalization if unknown
        
        min_val, max_val = normalization_thresholds[stat_type]
        
        # Handle edge cases
        if max_val - min_val == 0:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Clip to 0-1 range
        return np.clip(normalized, 0, 1)
    
    def _calculate_stat_value(self, stat_name: str, normalized_value: float) -> float:
        """
        Convert a normalized statistic to a weighted value based on importance.
        
        A normalized 'goals' value of 0.5 (average scoring) might become 1.0
        after applying the attacking weight, while 'tackles' of 0.5 might become 0.4.
        """
        # Determine which weight dictionary this stat belongs to
        all_weights = {}
        all_weights.update(self.stats_weight_attacking)
        all_weights.update(self.stats_weight_defensive)
        all_weights.update(self.stats_weight_possession)
        
        weight = all_weights.get(stat_name, 1.0)
        
        # Apply weight to normalized value
        return normalized_value * weight
    
    def calculate_player_value_rating(
        self,
        player_id: str,
        current_date: datetime = None
    ) -> Dict[str, float]:
        """
        Calculate a comprehensive value rating for a player.
        
        This is the main method you'll call when building features for Level 2.
        It returns multiple ratings so you can use them flexibly.
        
        Returns a dictionary containing:
        - overall_rating: Weighted combination of all aspects
        - attacking_rating: Goalscoring and chance creation
        - defensive_rating: Tackling, pressing, and defensive actions
        - possession_rating: Passing and ball retention
        - form_trend: Whether the player is improving or declining
        - reliability_score: How confident we are in this rating
        """
        if player_id not in self.player_history_cache:
            return {
                'overall_rating': 0.5,  # Default "average" rating
                'attacking_rating': 0.5,
                'defensive_rating': 0.5,
                'possession_rating': 0.5,
                'form_trend': 0.0,
                'reliability_score': 0.0,  # No data = low confidence
                'games_analyzed': 0
            }
        
        history = self.player_history_cache[player_id]
        
        if len(history) == 0:
            return {
                'overall_rating': 0.5,
                'attacking_rating': 0.5,
                'defensive_rating': 0.5,
                'possession_rating': 0.5,
                'form_trend': 0.0,
                'reliability_score': 0.0,
                'games_analyzed': 0
            }
        
        # Filter to only look at the most recent matches (rolling window)
        recent_matches = history[-self.lookback_matches:]
        num_matches = len(recent_matches)
        
        # Calculate exponential decay weights
        weights = self._calculate_decay_weights(num_matches)
        
        # Initialize accumulators
        attacking_sum = 0
        attacking_weight_total = 0
        defensive_sum = 0
        defensive_weight_total = 0
        possession_sum = 0
        possession_weight_total = 0
        
        total_weighted_minutes = 0
        recent_stats = []
        older_stats = []
        
        for i, match in enumerate(recent_matches):
            weight = weights[i]
            total_weighted_minutes += weight * match.minutes_played
            
            # Skip if player barely played (less than 10 minutes)
            if match.minutes_played < 10:
                continue
            
            # Split recent vs older for trend calculation
            if i < num_matches // 2:
                older_stats.append(match)
            else:
                recent_stats.append(match)
            
            # Process attacking stats
            for stat_name, weight_value in self.stats_weight_attacking.items():
                if hasattr(match, stat_name):
                    raw_value = getattr(match, stat_name)
                    normalized = self._normalize_stat(raw_value, stat_name)
                    stat_value = self._calculate_stat_value(stat_name, normalized)
                    
                    attacking_sum += stat_value * weight * weight_value
                    attacking_weight_total += weight * weight_value
            
            # Process defensive stats
            for stat_name, weight_value in self.stats_weight_defensive.items():
                if hasattr(match, stat_name):
                    raw_value = getattr(match, stat_name)
                    normalized = self._normalize_stat(raw_value, stat_name)
                    stat_value = self._calculate_stat_value(stat_name, normalized)
                    
                    defensive_sum += stat_value * weight * weight_value
                    defensive_weight_total += weight * weight_value
            
            # Process possession stats
            for stat_name, weight_value in self.stats_weight_possession.items():
                if hasattr(match, stat_name):
                    raw_value = getattr(match, stat_name)
                    normalized = self._normalize_stat(raw_value, stat_name)
                    stat_value = self._calculate_stat_value(stat_name, normalized)
                    
                    possession_sum += stat_value * weight * weight_value
                    possession_weight_total += weight * weight_value
        
        # Calculate final ratings (avoid division by zero)
        attacking_rating = (attacking_sum / attacking_weight_total 
                          if attacking_weight_total > 0 else 0.5)
        defensive_rating = (defensive_sum / defensive_weight_total 
                          if defensive_weight_total > 0 else 0.5)
        possession_rating = (possession_sum / possession_weight_total 
                           if possession_weight_total > 0 else 0.5)
        
        # Overall rating is weighted average of the three components
        # You can adjust these weights based on what matters most for your model
        overall_rating = (
            0.45 * attacking_rating +
            0.30 * defensive_rating +
            0.25 * possession_rating
        )
        
        # Calculate form trend: compare recent half vs older half
        form_trend = self._calculate_form_trend(recent_stats, older_stats)
        
        # Calculate reliability: based on minutes played and consistency
        reliability_score = self._calculate_reliability(
            total_weighted_minutes,
            attacking_weight_total + defensive_weight_total + possession_weight_total
        )
        
        return {
            'overall_rating': overall_rating,
            'attacking_rating': attacking_rating,
            'defensive_rating': defensive_rating,
            'possession_rating': possession_rating,
            'form_trend': form_trend,
            'reliability_score': reliability_score,
            'games_analyzed': num_matches
        }
    
    def _calculate_form_trend(
        self,
        recent_stats: List[PlayerMatchPerformance],
        older_stats: List[PlayerMatchPerformance]
    ) -> float:
        """
        Calculate whether a player is improving or declining.
        
        Returns:
            Positive value if improving, negative if declining, 0 if stable
        """
        if len(recent_stats) == 0 or len(older_stats) == 0:
            return 0.0
        
        def calculate_period_rating(stats: List[PlayerMatchPerformance]) -> float:
            """Calculate average rating for a time period."""
            if len(stats) == 0:
                return 0.5
            
            total = 0
            for match in stats:
                if match.minutes_played >= 10:
                    # Simple average of xG + assists (key attacking metrics)
                    total += match.xg + match.xa
            
            return total / len(stats) if len(stats) > 0 else 0.5
        
        recent_rating = calculate_period_rating(recent_stats)
        older_rating = calculate_period_rating(older_stats)
        
        # Return difference, scaled to be roughly -1 to 1
        return (recent_rating - older_rating) * 2
    
    def _calculate_reliability(
        self,
        weighted_minutes: float,
        weighted_stat_count: float
    ) -> float:
        """
        Calculate how confident we can be in this player's rating.
        
        Higher reliability means:
        - Player has played significant minutes recently
        - We have good data across multiple stat categories
        """
        # Target: 500 weighted minutes = ~6 full matches = high confidence
        minute_score = min(weighted_minutes / 500, 1.0)
        
        # Target: 50 weighted stat entries = reasonable coverage
        stat_score = min(weighted_stat_count / 50, 1.0)
        
        # Combined reliability score
        return (minute_score * 0.7 + stat_score * 0.3)
    
    def get_team_aggregate_value(
        self,
        player_ids: List[str],
        exclude_injured: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate the total value of a team's squad.
        
        This is crucial for your injury matrix. You pass in a list of player IDs,
        optionally excluding injured players, and get back the team's total strength.
        
        Args:
            player_ids: List of all players in the matchday squad
            exclude_injured: List of player IDs who are injured/unavailable
            
        Returns:
            Dictionary with aggregate team ratings
        """
        exclude_injured = exclude_injured or []
        
        attacking_total = 0
        defensive_total = 0
        possession_total = 0
        overall_total = 0
        reliability_accumulator = 0
        valid_player_count = 0
        
        for player_id in player_ids:
            if player_id in exclude_injured:
                continue
            
            ratings = self.calculate_player_value_rating(player_id)
            
            # Only count players with reasonable reliability
            if ratings['reliability_score'] > 0.3:
                attacking_total += ratings['attacking_rating']
                defensive_total += ratings['defensive_rating']
                possession_total += ratings['possession_rating']
                overall_total += ratings['overall_rating']
                reliability_accumulator += ratings['reliability_score']
                valid_player_count += 1
        
        if valid_player_count == 0:
            return {
                'attacking_total': 0,
                'defensive_total': 0,
                'possession_total': 0,
                'overall_total': 0,
                'avg_attacking': 0.5,
                'avg_defensive': 0.5,
                'avg_possession': 0.5,
                'avg_overall': 0.5,
                'team_reliability': 0.0,
                'players_analyzed': 0
            }
        
        return {
            'attacking_total': attacking_total,
            'defensive_total': defensive_total,
            'possession_total': possession_total,
            'overall_total': overall_total,
            'avg_attacking': attacking_total / valid_player_count,
            'avg_defensive': defensive_total / valid_player_count,
            'avg_possession': possession_total / valid_player_count,
            'avg_overall': overall_total / valid_player_count,
            'team_reliability': reliability_accumulator / valid_player_count,
            'players_analyzed': valid_player_count
        }
    
    def calculate_injury_impact(
        self,
        player_id: str,
        team_player_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate how much a player's injury hurts their team.
        
        This directly addresses your original concern about the injury matrix.
        Compare team strength with vs without the injured player.
        
        Args:
            player_id: The injured player
            team_player_ids: All players in the team
            
        Returns:
            Impact metrics showing how much the team loses
        """
        # Calculate team strength with everyone
        with_player = self.get_team_aggregate_value(team_player_ids)
        
        # Calculate team strength without the injured player
        without_player = self.get_team_aggregate_value(
            team_player_ids,
            exclude_injured=[player_id]
        )
        
        # Calculate the difference
        player_rating = self.calculate_player_value_rating(player_id)
        
        return {
            'player_overall_value': player_rating['overall_rating'],
            'player_attacking_value': player_rating['attacking_rating'],
            'player_defensive_value': player_rating['defensive_rating'],
            
            'team_impact_overall': with_player['overall_total'] - without_player['overall_total'],
            'team_impact_attacking': with_player['attacking_total'] - without_player['attacking_total'],
            'team_impact_defensive': with_player['defensive_total'] - without_player['defensive_total'],
            
            'percent_attacking_impact': (
                (with_player['attacking_total'] - without_player['attacking_total']) /
                max(with_player['attacking_total'], 0.1) * 100
            ),
            
            'player_reliability': player_rating['reliability_score']
        }
```

## How This Fixes Your Injury Matrix

The code includes a `calculate_injury_impact` method that directly solves your original problem. When you want to calculate how much Manchester City would miss Kevin De Bruyne, you call this method with De Bruyne's player ID and the list of all Manchester City player IDs. The method calculates the team's total attacking, defensive, and overall strength both with and without De Bruyne, then returns the difference.

For example, it might tell you that De Bruyne's removal reduces Manchester City's attacking strength by 12.3% and their overall team strength by 8.7. You can then feed these impact values directly into Level 2 of your hierarchical model. This preserves the player-level granularity that makes your model sophisticated while avoiding the noise problem that caused your original Level 1 to fail.

## Integration With Your Hierarchical Model

Here's how this ImprovedPlayerLayer fits into your overall architecture. In your Level 2 (Team/Tactics layer), instead of trying to predict xG from scratch, you now have access to a pre-calculated Team Strength value based on actual player contributions:

```python
def build_team_features(home_team_id, away_team_id, injured_players):
    """
    Build features for Level 2 using the ImprovedPlayerLayer.
    """
    # Get all players for each team from your database
    home_players = get_team_players(home_team_id)
    away_players = get_team_players(away_team_id)
    
    # Calculate team values (with injury adjustments)
    home_strength = player_layer.get_team_aggregate_value(
        home_players,
        exclude_injured=injured_players.get(home_team_id, [])
    )
    away_strength = player_layer.get_team_aggregate_value(
        away_players,
        exclude_injured=injured_players.get(away_team_id, [])
    )
    
    # Build difference features for Level 2
    features = {
        'attack_diff': home_strength['avg_attacking'] - away_strength['avg_attacking'],
        'defense_diff': home_strength['avg_defensive'] - away_strength['avg_defensive'],
        'possession_diff': home_strength['avg_possession'] - away_strength['avg_possession'],
        'team_strength_diff': home_strength['avg_overall'] - away_strength['avg_overall'],
        'home_reliability': home_strength['team_reliability'],
        'away_reliability': away_strength['team_reliability'],
        'home_players_available': home_strength['players_analyzed'],
        'away_players_available': away_strength['players_analyzed'],
        
        # Injury impact features
        'home_injury_impact': sum([
            player_layer.calculate_injury_impact(p, home_players)['team_impact_overall']
            for p in injured_players.get(home_team_id, [])
        ]),
        'away_injury_impact': sum([
            player_layer.calculate_injury_impact(p, away_players)['team_impact_overall']
            for p in injured_players.get(away_team_id, [])
        ])
    }
    
    return features
```

This approach gives you the best of both worlds: you're capturing player-level granularity through the ImprovedPlayerLayer's value calculations, but you're not trying to make noisy single-match predictions that the model can't learn from. The rolling window with exponential decay creates stable, meaningful features that Level 2 can actually use to improve its predictions.
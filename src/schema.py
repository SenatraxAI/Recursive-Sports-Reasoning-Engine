"""
This file defines the 'Data Schemas' for our project. 
Think of a schema as a blueprint or a recipe. It tells the computer exactly 
what information we expect to see for a Player, a Match, or an Injury.

Using @dataclass makes it very easy to create these "containers" for data.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict
from enum import Enum
import numpy as np

# --- Level 1: Player Data ---
# This matches 'Level 1' of our 4-level hierarchy.
# We focus on individual performance metrics here.

@dataclass
class PlayerMatchStats:
    """
    Stores everything a player did in a single match.
    We use these individual stats to predict how well a player will perform 
    in the future (Level 1 Models).
    """
    player_id: str
    player_name: str
    team_id: str
    match_date: date
    season: str
    matchweek: int
    home_or_away: str
    minutes_played: int
    
    # --- IDENTITY & PHYSICAL (From idea.md) ---
    dob: Optional[date] = None
    nationality: Optional[str] = None
    height_cm: Optional[int] = None
    preferred_foot: Optional[str] = None # "Left", "Right", "Both"
    
    # Offensive Metrics
    goals: int = 0
    assists: int = 0
    xg: float = 0.0  # Expected Goals (quality of shots)
    xa: float = 0.0  # Expected Assists (quality of passes leading to shots)
    
    # Passing & Progression
    passes_completed: int = 0
    passes_attempted: int = 0
    key_passes: int = 0  # Passes that lead to a shot
    progressive_passes: int = 0 # Passes that move the ball significantly forward
    progressive_carries: int = 0 # Dribbles that move the ball significantly forward
    
    # --- SUPER QUALITY METRICS (The "Blink" Level) ---
    # These metrics give us the deepest possible insight into a player's physical 
    # and technical performance during a match.
    
    distance_covered_km: float = 0.0 # Total ground covered
    sprints_completed: int = 0      # High-intensity bursts
    top_speed_kmh: float = 0.0      # Fastest recorded speed
    
    total_touches: int = 0          # Every time they touched the ball
    touches_in_box: int = 0         # How dangerous are they in the penalty area?
    passes_into_final_third: int = 0 # Playmaking impact
    
    # Pressure & Workrate
    pressures_applied: int = 0      # How many times did they close down an opponent?
    ball_recoveries: int = 0        # Winning the 'second ball'
    
    # Errors (To track consistency)
    miscontrols: int = 0            # Losing the ball through poor touch
    dispossessed: int = 0           # Being tackled while on the ball

class InjuryType(Enum):
    """A list of fixed categories for injuries to help the model group similar types."""
    MUSCLE = "muscle"     # Hamstrings, quads, etc.
    KNEE = "knee"         # ACL, meniscus, etc.
    ANKLE = "ankle"
    ILLNESS = "illness"
    SUSPENSION = "suspension" # Red cards or yellow card accumulation
    UNKNOWN = "unknown"

@dataclass
class PlayerInjury:
    """
    Tracks when a player is unavailable. 
    This is CRITICAL for our 'Injury Impact Model' to know who is missing.
    """
    player_id: str
    team_id: str
    injury_start: date
    injury_end: Optional[date] # Can be None if the player is still out
    injury_type: InjuryType
    description: Optional[str] = None

# --- Level 2: Manager & Team Data ---
# Level 2 aggregates the players into a team and adds tactical context.

class Formation(Enum):
    """Common tactical setups."""
    F433 = "4-3-3"
    F4231 = "4-2-3-1"
    F442 = "4-4-2"
    F343 = "3-4-3"
    F352 = "3-5-2"
    OTHER = "other"

@dataclass
class ManagerProfile:
    """
    Captures the 'DNA' of a manager's tactics.
    We use this in Level 2 to see if a manager's style fits the available players.
    """
    manager_id: str
    name: str
    formation_preference: Formation
    avg_possession: float   # Does the team like the ball?
    pressing_intensity: float # PPDA: Lower means higher intensity pressing
    transition_speed: float  # How fast do they counter-attack? (-1 to 1)
    
    # --- CAREER HISTORY (From idea.md) ---
    previous_clubs: List[str] = field(default_factory=list)
    win_rate_career: float = 0.0
    trophies_won: int = 0
    tenure_avg_days: int = 0

# --- Level 3 & 4: Game State (Markov/Monte Carlo) ---
# This is for the 'Live' part of the project.

@dataclass
class GameState:
    """
    A snapshot of a match at any given minute.
    Used by the Markov Chain to predict what state follows next (e.g., a goal or a card).
    """
    home_score: int = 0
    away_score: int = 0
    minute: int = 0
    home_reds: int = 0
    away_reds: int = 0
    home_possession: float = 0.5
    momentum: float = 0.0 # Positive favors home, Negative favors away
    home_energy: float = 1.0 # Fatigue level (starts at 1.0, goes down)
    away_energy: float = 1.0

    def to_vector(self) -> np.ndarray:
        """Converts the snapshot into numbers that a Machine Learning model can understand."""
        return np.array([
            self.home_score, self.away_score, self.minute / 90.0,
            self.home_reds, self.away_reds, self.home_possession,
            (self.momentum + 1) / 2, self.home_energy, self.away_energy
        ])

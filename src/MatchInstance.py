from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ManagerContext:
    name: str
    style: str = "Possession" # Possession, Counter, High-Press
    formation: str = "4-3-3"
    pressing: str = "Neutral" # Aggressive, Passive, Balanced
    tempo: str = "Normal" # Slow, Fast, Normal

@dataclass
class MatchInstance:
    """
    Hierarchical 2.0: The Match Instance object.
    Allows for specific custom simulations of a game.
    """
    home_team: str
    away_team: str
    home_manager: ManagerContext
    away_manager: ManagerContext
    home_lineup: List[str] = field(default_factory=list)
    away_lineup: List[str] = field(default_factory=list)
    
    match_id: str = "custom_sim_01"
    match_date: str = "2026-01-12"

    def to_dict(self):
        return {
            "h_team": self.home_team,
            "a_team": self.away_team,
            "h_manager": self.home_manager.__dict__,
            "a_manager": self.away_manager.__dict__,
            "h_lineup_size": len(self.home_lineup),
            "a_lineup_size": len(self.away_lineup)
        }

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class Formation(Enum):
    F_433 = "4-3-3"
    F_4231 = "4-2-3-1"
    F_343 = "3-4-3"
    F_352 = "3-5-2"
    F_442 = "4-4-2"
    F_532 = "5-3-2"
    F_4141 = "4-1-4-1"
    F_4321 = "4-3-2-1" # The Christmas Tree
    
    @staticmethod
    def get_structure(fmt_str: str) -> Dict[str, int]:
        """
        Converts a formation string into structural counts for the model to learn from.
        This allows XGBoost to spot patterns like '2 Strikers = Traffic Jam'.
        """
        # Default fallback
        structure = {'def': 4, 'mid': 3, 'att': 3, 'central_def': 2, 'wide_att': 2}
        
        if fmt_str == "4-3-3":
            structure = {'def': 4, 'mid': 3, 'att': 3, 'central_def': 2, 'wide_att': 2}
        elif fmt_str == "4-2-3-1":
            structure = {'def': 4, 'mid': 5, 'att': 1, 'central_def': 2, 'wide_att': 2} # 3 AMs count as Mid in this schema for density
        elif fmt_str == "3-4-3":
            structure = {'def': 3, 'mid': 4, 'att': 3, 'central_def': 3, 'wide_att': 2}
        elif fmt_str == "3-5-2":
            structure = {'def': 3, 'mid': 5, 'att': 2, 'central_def': 3, 'wide_att': 0}
        elif fmt_str == "4-4-2":
            structure = {'def': 4, 'mid': 4, 'att': 2, 'central_def': 2, 'wide_att': 2}
        
        # Derived densities (Features for the model)
        structure['density_central'] = structure['central_def'] + (structure['mid'] - 2 if structure['mid'] > 2 else 0)
        structure['density_attack'] = structure['att']
        
        return structure

@dataclass
class ManagerTacticalSetup:
    name: str
    style: str # "Gegenpress", "Possession", "Counter", "Low Block"
    primary_formation: str
    
    # Tactical Constraints (The "Filter" settings)
    # These define what the manager DEMANDS from players.
    # Scale 0.0 - 1.0 (0=Irrelevant, 1=Critical)
    
    # Physical Demands
    req_stamina: float = 0.5 
    req_sprint_speed: float = 0.5
    
    # Technical Demands
    req_pass_completion: float = 0.5
    req_progressive_actions: float = 0.5
    
    # Tactical Demands
    req_defensive_workrate: float = 0.5
    req_positional_discipline: float = 0.5
    
    def get_profile_vector(self) -> List[float]:
        """Returns the tactical constraints as a vector for the model."""
        return [
            self.req_stamina, self.req_sprint_speed,
            self.req_pass_completion, self.req_progressive_actions,
            self.req_defensive_workrate, self.req_positional_discipline
        ]

# --- THE 2026 MANAGER DATABASE ---
# Defining the archetypes for the current PL managers (Jan 2026 projection)

MANAGER_DB = {
    # The Elite Pressers
    "Guardiola": ManagerTacticalSetup(
        name="Pep Guardiola", style="Positional Play", primary_formation="3-2-4-1",
        req_pass_completion=0.95, req_positional_discipline=1.0, req_stamina=0.7,
        req_progressive_actions=0.9
    ),
    "Klopp": ManagerTacticalSetup(
        name="Jurgen Klopp", style="Gegenpress", primary_formation="4-3-3",
        req_stamina=1.0, req_sprint_speed=0.8, req_defensive_workrate=1.0,
        req_progressive_actions=0.7
    ),
    "Arteta": ManagerTacticalSetup(
        name="Mikel Arteta", style="Control Possession", primary_formation="4-3-3",
        req_pass_completion=0.9, req_positional_discipline=0.9, req_defensive_workrate=0.8
    ),
    
    # The Pragmatists
    "Dyche": ManagerTacticalSetup(
        name="Sean Dyche", style="Low Block", primary_formation="4-4-1-1",
        req_defensive_workrate=1.0, req_positional_discipline=0.9,
        req_pass_completion=0.3, # Happy to clear long
        req_stamina=0.8
    ),
    "Emery": ManagerTacticalSetup(
        name="Unai Emery", style="Structured Counter", primary_formation="4-4-2",
        req_positional_discipline=1.0, req_sprint_speed=0.8,
        req_defensive_workrate=0.7
    ),
    
    # The Chaos Agents
    "Postecoglou": ManagerTacticalSetup(
        name="Ange Postecoglou", style="All Out Attack", primary_formation="4-3-3",
        req_sprint_speed=0.9, req_stamina=0.9, req_progressive_actions=1.0,
        req_defensive_workrate=0.4 # Risks allowed
    ),
    "De Zerbi": ManagerTacticalSetup(
        name="Roberto De Zerbi", style="Artificial Transition", primary_formation="4-2-3-1",
        req_pass_completion=0.95, req_progressive_actions=0.9,
        req_positional_discipline=0.8
    ),
    
    # The Transition Managers (Mid-Table / Generic)
    "Howe": ManagerTacticalSetup(
        name="Eddie Howe", style="High Intensity", primary_formation="4-3-3",
        req_stamina=0.9, req_sprint_speed=0.8
    ),
    "Frank": ManagerTacticalSetup(
        name="Thomas Frank", style="Direct Counter", primary_formation="3-5-2",
        req_stamina=0.9, req_positional_discipline=0.9
    ),
    
    # --- CLASS OF 2026 ADDITIONS ---
    "Slot": ManagerTacticalSetup(
        name="Arne Slot", style="Controlled Press", primary_formation="4-2-3-1",
        req_pass_completion=0.9, req_positional_discipline=0.9, req_stamina=0.85,
        req_progressive_actions=0.8
    ),
    "Maresca": ManagerTacticalSetup(
        name="Enzo Maresca", style="Positional Possession", primary_formation="4-3-3",
        req_pass_completion=0.95, req_positional_discipline=1.0, req_defensive_workrate=0.6,
        req_progressive_actions=0.7
    ),
    "Amorim": ManagerTacticalSetup(
        name="Ruben Amorim", style="3-4-3 Transition", primary_formation="3-4-3",
        req_sprint_speed=0.85, req_defensive_workrate=0.9, req_positional_discipline=0.9,
        req_progressive_actions=0.85
    ),
    "Hurzeler": ManagerTacticalSetup(
        name="Fabian Hurzeler", style="Dynamic Structure", primary_formation="3-4-3",
        req_pass_completion=0.9, req_positional_discipline=0.95, req_progressive_actions=0.9
    ),
    "Glasner": ManagerTacticalSetup(
        name="Oliver Glasner", style="Direct Press", primary_formation="3-4-2-1",
        req_stamina=0.95, req_sprint_speed=0.9, req_defensive_workrate=0.9
    ),
    "Iraola": ManagerTacticalSetup(
        name="Andoni Iraola", style="Chaos Press", primary_formation="4-2-3-1",
        req_stamina=1.0, req_defensive_workrate=0.8, req_progressive_actions=0.9
    ),
    "Silva": ManagerTacticalSetup(
        name="Marco Silva", style="Balanced", primary_formation="4-2-3-1",
        req_pass_completion=0.8, req_positional_discipline=0.8, req_defensive_workrate=0.8
    ),
    "O'Neil": ManagerTacticalSetup(
        name="Gary O'Neil", style="Adaptive Block", primary_formation="5-4-1",
        req_defensive_workrate=1.0, req_positional_discipline=0.9, req_stamina=0.8
    ),
    "Lopetegui": ManagerTacticalSetup(
        name="Julen Lopetegui", style="Structured Possession", primary_formation="4-3-3",
        req_pass_completion=0.9, req_positional_discipline=0.9
    ),
    "Kompany": ManagerTacticalSetup(
        name="Vincent Kompany", style="Man City Lite", primary_formation="4-4-2",
        req_pass_completion=0.9, req_progressive_actions=0.9, req_stamina=0.8
    ),
    # Umlaut Handling
    "Hürzeler": ManagerTacticalSetup(
        name="Fabian Hurzeler", style="Dynamic Structure", primary_formation="3-4-3",
        req_pass_completion=0.9, req_positional_discipline=0.95, req_progressive_actions=0.9
    ),
    
    # Championship Promoted / Others
    "Nuno": ManagerTacticalSetup(
        name="Nuno Espirito Santo", style="Counter Attack", primary_formation="5-3-2",
        req_defensive_workrate=1.0, req_positional_discipline=1.0, req_pass_completion=0.7
    ),
    "Farke": ManagerTacticalSetup(
        name="Daniel Farke", style="Possession", primary_formation="4-2-3-1",
        req_pass_completion=0.9, req_progressive_actions=0.9
    ),
    "Le Bris": ManagerTacticalSetup(
        name="Regis Le Bris", style="Dynamic Youth", primary_formation="4-3-3", 
        req_sprint_speed=0.9, req_stamina=0.9, req_progressive_actions=0.8
    ),
    "Martin": ManagerTacticalSetup(
        name="Russell Martin", style="Extreme Possession", primary_formation="4-3-3", 
        req_pass_completion=0.95, req_positional_discipline=0.9, req_progressive_actions=0.7
    ),
    "McKenna": ManagerTacticalSetup(
        name="Kieran McKenna", style="Hybrid Press", primary_formation="4-2-3-1",
        req_stamina=0.9, req_progressive_actions=0.9, req_positional_discipline=0.85
    ),
    "Cooper": ManagerTacticalSetup(
        name="Steve Cooper", style="Solid Structure", primary_formation="4-3-3",
        req_defensive_workrate=0.9, req_positional_discipline=0.9
    )
}

def get_manager_profile(manager_name: str) -> ManagerTacticalSetup:
    # Fuzzy match or returning generic fallback
    for key, profile in MANAGER_DB.items():
        if key.lower() in manager_name.lower():
            return profile
            
    # Generic Fallback
    return ManagerTacticalSetup(name="Generic", style="Balanced", primary_formation="4-4-2")

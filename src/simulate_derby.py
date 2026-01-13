from src.predict_match import MatchPredictor
from src.MatchInstance import MatchInstance, ManagerContext

def simulate_manchester_derby():
    """
    Demonstration of how to create a specific match instance with custom managers and lineups.
    """
    
    # 1. Define Manager Contexts
    pep = ManagerContext(
        name="Pep Guardiola",
        style="Possession",
        formation="4-3-3",
        pressing="Aggressive",
        tempo="Normal"
    )
    
    ten_hag = ManagerContext(
        name="Erik ten Hag",
        style="Counter",
        formation="4-2-3-1",
        pressing="Balanced",
        tempo="Fast"
    )
    
    # 2. Define Specific Lineups (Example Names)
    city_lineup = [
        "Ederson", "Kyle Walker", "Ruben Dias", "Manuel Akanji", "Josko Gvardiol",
        "Rodri", "Kevin De Bruyne", "Bernardo Silva", 
        "Phil Foden", "Jeremy Doku", "Erling Haaland"
    ]
    
    united_lineup = [
        "Andre Onana", "Diogo Dalot", "Lisandro Martinez", "Raphael Varane", "Luke Shaw",
        "Casemiro", "Kobbie Mainoo", "Bruno Fernandes",
        "Alejandro Garnacho", "Marcus Rashford", "Rasmus Hojlund"
    ]
    
    # 3. Create the Match Instance
    derby = MatchInstance(
        home_team="man_city",
        away_team="man_utd",
        home_manager=pep,
        away_manager=ten_hag,
        home_lineup=city_lineup,
        away_lineup=united_lineup
    )
    
    # 4. Initialize Predictor and Run
    predictor = MatchPredictor()
    predictor.request_prediction(instance=derby)

if __name__ == "__main__":
    simulate_manchester_derby()

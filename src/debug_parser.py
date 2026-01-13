from src.lineup_parser import LineupParser
import pandas as pd

def debug():
    print("Initializing Parser...")
    parser = LineupParser()
    
    print(f"\nTotal Teams in DB: {len(parser.team_rosters)}")
    
    # Check Arsenal
    team = "arsenal"
    if team in parser.team_rosters:
        print(f"\nArsenal Roster ({len(parser.team_rosters[team])} players):")
        print(list(parser.team_rosters[team])[:10]) # Show first 10
    else:
        print("\nERROR: Arsenal not found in DB!")
        
    # Check Liverpool
    team = "liverpool"
    if team in parser.team_rosters:
        print(f"\nLiverpool Roster ({len(parser.team_rosters[team])} players):")
        print(list(parser.team_rosters[team])[:10])
    else:
        print("\nERROR: Liverpool not found in DB!")
        
    # Test Matching
    print("\n--- Testing Match ---")
    
    sample_arsenal = "Raya, White, Saliba, Gabriel, Timber, Rice, Merino, Odegaard, Saka, Martinelli, Havertz"
    print(f"Input: {sample_arsenal}")
    score = parser.calculate_strength("Arsenal", sample_arsenal)
    print(f"Score: {score}%")
    
    sample_pool = "Alisson, Trent, Konate, Van Dijk, Robertson, Gravenberch, Mac Allister, Szoboszlai, Salah, Jota, Diaz"
    print(f"Input: {sample_pool}")
    score_pool = parser.calculate_strength("Liverpool", sample_pool)
    print(f"Score: {score_pool}%")

if __name__ == "__main__":
    debug()

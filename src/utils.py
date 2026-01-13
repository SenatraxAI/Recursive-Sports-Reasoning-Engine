import re
import unicodedata

def normalize_name(name: str) -> str:
    """
    Standardizes a name for matching across sources.
    - Lowers case
    - Removes accents (é -> e)
    - Removes punctuation
    - Replaces spaces with underscores
    """
    if not isinstance(name, str):
        return ""
        
    # Remove accents
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # Lowercase and clean
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s_]", "", name)
    name = "_".join(name.split())
    
    return name

def map_team_name(name: str) -> str:
    """
    Dictionary to map various team name aliases.
    """
    mapping = {
        "manchester_united": "man_utd",
        "man_united": "man_utd",
        "manchester_utd": "man_utd",
        "man_utd_fc": "man_utd",
        "manchester_city": "man_city",
        "man_city_fc": "man_city",
        "tottenham_hotspur": "tottenham",
        "spurs": "tottenham",
        "wolverhampton_wanderers": "wolves",
        "leicester_city": "leicester",
        "brighton_and_hove_albion": "brighton",
        "west_ham_united": "west_ham",
        "newcastle_united": "newcastle",
        "sheffield_united": "sheffield_utd",
        "sheff_utd": "sheffield_utd",
        "norwich_city": "norwich",
        "ipswich_town": "ipswich",
        "leeds_united": "leeds",
        "leeds_utd": "leeds",
        "bournemouth": "afc_bournemouth",
        "nottm_forest": "nottingham_forest",
        "nottingham_forest_fc": "nottingham_forest",
    }
    
    norm = normalize_name(name)
    return mapping.get(norm, norm)

def get_current_manager(team_name: str) -> str:
    """
    Fallback: Returns the known manager for the team in Jan 2026.
    Used when the historical match database has 'Unknown' or stale data.
    """
    db = {
        "arsenal": "Mikel Arteta",
        "aston_villa": "Unai Emery",
        "afc_bournemouth": "Andoni Iraola",
        "brentford": "Thomas Frank",
        "brighton": "Fabian Hürzeler",
        "burnley": "Vincent Kompany",
        "chelsea": "Enzo Maresca",
        "crystal_palace": "Oliver Glasner",
        "everton": "Sean Dyche",
        "fulham": "Marco Silva",
        "ipswich": "Kieran McKenna",
        "leicester": "Steve Cooper",
        "leeds": "Daniel Farke",
        "liverpool": "Arne Slot",
        "man_city": "Pep Guardiola",
        "man_utd": "Rúben Amorim",
        "newcastle": "Eddie Howe",
        "nottingham_forest": "Nuno Espirito Santo",
        "southampton": "Russell Martin",
        "sunderland": "Regis Le Bris",
        "tottenham": "Ange Postecoglou",
        "west_ham": "Julen Lopetegui",
        "wolves": "Gary O'Neil",
    }
    
    norm = map_team_name(team_name)
    return db.get(norm, "Unknown")

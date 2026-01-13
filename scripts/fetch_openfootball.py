import requests
import pandas as pd
from pathlib import Path
import json

def fetch_openfootball_data(data_dir="data/raw"):
    """
    Downloads Premier League match results from openfootball/football.json
    Source: https://github.com/openfootball/england
    """
    base_url = "https://raw.githubusercontent.com/openfootball/football.json/master"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Season mapping for openfootball
    # They usually follow '2020-21/en.1.json' format
    seasons = [
        "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
    ]
    
    all_matches = []
    
    for season in seasons:
        url = f"{base_url}/{season}/en.1.json"
        print(f"Downloading: {url}")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                # Add season column to each match
                for m in matches:
                    m['season'] = season
                    
                    # Flatten the score structure for easier CSV
                    if 'score' in m and 'ft' in m['score']:
                        m['home_score'] = m['score']['ft'][0]
                        m['away_score'] = m['score']['ft'][1]
                    else:
                        m['home_score'] = None
                        m['away_score'] = None
                        
                all_matches.extend(matches)
                print(f"  -> Found {len(matches)} matches.")
            else:
                print(f"  -> Failed (Status {response.status_code}). Note: 2024-25 might be in a different path or repo.")
                
        except Exception as e:
            print(f"  -> Error: {e}")

    # Convert to DataFrame
    if all_matches:
        df = pd.DataFrame(all_matches)
        # Clean up columns if needed
        output_path = data_dir / "openfootball_matches.json"
        
        # Save as JSON (raw) and CSV (flat)
        with open(data_dir / "openfootball_matches.json", 'w') as f:
            json.dump(all_matches, f, indent=4)
            
        # Simplified CSV for immediate view
        simple_df = df[['date', 'season', 'team1', 'team2', 'home_score', 'away_score']]
        simple_df.to_csv(data_dir / "openfootball_matches.csv", index=False)
        print(f"Saved {len(all_matches)} total matches to {output_path}")
    else:
        print("No matches downloaded.")

if __name__ == "__main__":
    fetch_openfootball_data()

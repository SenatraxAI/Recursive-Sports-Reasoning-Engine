import pandas as pd
import json
from src.utils import map_team_name
from pathlib import Path

def debug():
    df = pd.read_parquet('data/processed/processed_matches.parquet')
    utd = df[df['home_team'] == 'man_utd'].iloc[0]
    
    match_date = utd.date
    print(f"Match Date: {match_date}")
    print(f"Match Date TZ: {match_date.tzinfo}")
    
    with open('data/raw/manager_master.json') as f:
        m = json.load(f)
        
    for t in m['tenures']:
        t_mapped = map_team_name(t['team'])
        m_mapped = map_team_name('man_utd')
        
        if t_mapped == m_mapped:
            start = pd.to_datetime(t['start']).tz_localize(None)
            end = pd.to_datetime(t['end']).tz_localize(None) if t['end'] != "Present" else pd.to_datetime('2099-01-01')
            
            print(f"MATCH FOUND: {t['manager']}")
            match_date_naive = match_date.replace(tzinfo=None)
            print(f"Comparing: {start} <= {match_date_naive} <= {end}")
            
            in_range = start <= match_date_naive <= end
            print(f"Result: {in_range}")
        else:
            if "utd" in t_mapped or "man" in t_mapped:
                print(f"NO MATCH: t_mapped='{t_mapped}', m_mapped='{m_mapped}'")

if __name__ == "__main__":
    debug()

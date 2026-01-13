from statsbombpy import sb
import pandas as pd

# Check available competitions
comps = sb.competitions()

# Filter for Premier League
pl = comps[comps['competition_name'] == 'Premier League']
print(pl[['season_name', 'season_id', 'competition_id']])

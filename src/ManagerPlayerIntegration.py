from typing import Dict, List, Any
from src.StyleNormalization import StyleNormalization
from src.ImprovedPlayerLayer import ImprovedPlayerLayer
import logging

logger = logging.getLogger(__name__)

class ManagerPlayerIntegration:
    """
    Hierarchical 2.0: Point 3.
    Bridges the Manager's Level 3 tactics with the Player's Level 1 stats.
    """
    
    def __init__(self, player_layer: ImprovedPlayerLayer):
        self.player_layer = player_layer
        self.manager_context_cache: Dict[str, Dict] = {} # match_id -> context
        
    def process_match_performance(self, player_name: str, match_data: Dict[str, Any], manager_profile: Dict[str, Any]):
        """
        1. Extract Tactical Context from Manager.
        2. Normalize Player Stats using that Context.
        3. Update ImprovedPlayerLayer with Context-Aware Stats.
        """
        # Extract context from Manager Profile
        # Expected profile keys: 'possession_type', 'pressing_intensity', 'tempo'
        context = {
            'possession': manager_profile.get('possession_type', 'neutral'),
            'pressing': manager_profile.get('pressing_intensity', 'neutral'),
            'tempo': manager_profile.get('tempo', 'neutral'),
            'formation': manager_profile.get('primary_formation', 'Unknown')
        }
        
        # Get raw stats from match data
        raw_stats = match_data.get('player_stats', {})
        
        # Normalize stats by the manager's tactical "Soup"
        normalized_stats = StyleNormalization.normalize_player_stats(raw_stats, context)
        
        # Save to the player layer
        self.player_layer.update_player(
            player_name=player_name,
            match_date=match_data.get('date'),
            stats=normalized_stats
        )
        
        # Store metadata about the context the player succeeded in
        # (This powers the Transfer Prediction - Point 9)
        self._record_style_experience(player_name, context)

    def _record_style_experience(self, player_name: str, context: Dict[str, str]):
        """
        Builds a 'Style DNA' for the player based on which systems they've played in.
        """
        player_entry = self.player_layer.player_db.setdefault(player_name, {})
        style_dna = player_entry.setdefault('style_dna', {})
        
        for k, v in context.items():
            if k == 'formation': continue
            style_key = f"{k}_{v}"
            style_dna[style_key] = style_dna.get(style_key, 0) + 1
            
    def get_player_preferred_style(self, player_name: str) -> Dict[str, str]:
        """
        Analyzes the player's history to find their 'Home' system.
        """
        dna = self.player_layer.player_db.get(player_name, {}).get('style_dna', {})
        if not dna:
            return {}
            
        # Simplistic version: Most frequent label in each category
        preferred = {}
        for category in ['possession', 'pressing', 'tempo']:
            relevant = {k: v for k, v in dna.items() if k.startswith(category)}
            if relevant:
                # Find the label with max count (e.g., 'possession_high')
                best = max(relevant, key=relevant.get)
                preferred[category] = best.split('_')[1]
                
        return preferred

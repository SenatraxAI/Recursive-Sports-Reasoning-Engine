from typing import Dict, Any
from src.ManagerPlayerIntegration import ManagerPlayerIntegration
from src.StyleNormalization import StyleNormalization
import logging

logger = logging.getLogger(__name__)

class TransferSim:
    """
    Hierarchical 2.0: Point 9.
    Projects player performance across different managerial contexts.
    """
    
    def __init__(self, integration_layer: ManagerPlayerIntegration):
        self.integration = integration_layer
        
    def predict_transfer_success(self, player_name: str, target_manager_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare Player's 'Preferred Style' with Target Manager's Style.
        """
        player_prefs = self.integration.get_player_preferred_style(player_name)
        if not player_prefs:
            return {'confidence': 'low', 'reason': 'No historical style data for player'}
            
        target_style = {
            'possession': target_manager_profile.get('possession_type', 'neutral'),
            'pressing': target_manager_profile.get('pressing_intensity', 'neutral'),
            'tempo': target_manager_profile.get('tempo', 'neutral')
        }
        
        # Calculate the delta
        fit_multiplier = StyleNormalization.get_style_fit_delta(player_prefs, target_style)
        
        # Success Logic
        success_prob = "High" if fit_multiplier > 1.1 else "Medium" if fit_multiplier >= 0.95 else "Low"
        
        return {
            'player_name': player_name,
            'fit_score': fit_multiplier,
            'success_prediction': success_prob,
            'style_match': player_prefs == target_style,
            'delta_analysis': {
                'possession_clash': player_prefs.get('possession') != target_style.get('possession'),
                'pressing_clash': player_prefs.get('pressing') != target_style.get('pressing')
            }
        }

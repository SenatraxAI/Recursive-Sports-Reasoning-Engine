import numpy as np
import pandas as pd
from typing import List, Dict
from src.DeepBoostNetwork import DeepBoostNetwork

class DeepBoostVisualizer:
    """
    Hierarchical 2.0: DeepBoost-Network Visualizer.
    Generates Mermaid diagrams and "Inference Flow" summaries.
    """
    
    def __init__(self, network: DeepBoostNetwork):
        self.network = network

    def generate_mermaid_diagram(self) -> str:
        """
        Creates a Mermaid flowchart string representing the Cascaded Forest.
        """
        mermaid = ["graph TD"]
        mermaid.append("    subgraph InputLayer[Original Features]")
        mermaid.append("        F[Raw Match Features] --> L1")
        mermaid.append("    end")
        
        for i, layer in enumerate(self.network.layers):
            layer_id = f"Layer_{i+1}"
            prev_layer = f"Layer_{i}" if i > 0 else "InputLayer"
            
            mermaid.append(f"    subgraph {layer_id}[DeepBoost Layer {i+1}]")
            for j in range(layer.n_neurons):
                neuron_id = f"L{i+1}N{j+1}"
                mermaid.append(f"        {neuron_id}[XGB Neuron {j+1}]")
            mermaid.append("    end")
            
            # Connections
            mermaid.append(f"    F --> {layer_id}")
            if i > 0:
                mermaid.append(f"    {prev_layer} -- Stacked Probs --> {layer_id}")
                
        # Final prediction
        mermaid.append("    subgraph OutputNode[Final Aggregator]")
        mermaid.append(f"        Final[Final XGBoost Classifier] --> Res{{Match Result}}")
        mermaid.append("    end")
        
        last_layer = f"Layer_{len(self.network.layers)}"
        mermaid.append(f"    {last_layer} -- Enriched Features --> Final")
        mermaid.append("    F -- Residual Flow --> Final")
        
        return "\n".join(mermaid)

    def simulate_inference_flow(self, sample_input: np.ndarray) -> List[str]:
        """
        Traces how a single match "travels" through the network.
        """
        flow = []
        flow.append("🚀 Starting Inference Simulation...")
        flow.append(f"   Inputs: {sample_input.shape[1]} raw tactical features detected.")
        
        current_X = sample_input
        for i, layer in enumerate(self.network.layers):
            flow.append(f"\n🌊 Entering Layer {i+1}:")
            layer_output = layer.transform(current_X)
            flow.append(f"   - {layer.n_neurons} neurons activated.")
            flow.append(f"   - Produced {layer_output.shape[1]} probability dimensions.")
            
            # Show feature enrichment
            current_X = np.hstack([sample_input, layer_output])
            flow.append(f"   - Stacking: New feature vector size = {current_X.shape[1]}")
            
        flow.append("\n🎯 Final Decision Layer:")
        final_probs = self.network.final_clf.predict_proba(current_X)[0]
        results = ["Home Win", "Draw", "Away Win"]
        flow.append(f"   - Final Prediction: {results[np.argmax(final_probs)]}")
        flow.append(f"   - Confidence Profile: H:{final_probs[0]:.2f}, D:{final_probs[1]:.2f}, A:{final_probs[2]:.2f}")
        
        return flow

# Example usage:
if __name__ == "__main__":
    # Create a dummy network for visualization demonstration
    mock_net = DeepBoostNetwork(max_layers=2, neurons_per_layer=3)
    # (In a real scenario, we'd load a trained model)
    viz = DeepBoostVisualizer(mock_net)
    print(viz.generate_mermaid_diagram())

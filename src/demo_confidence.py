import numpy as np
import json
from src.DeepBoostNetwork import DeepBoostNetwork

def run_confidence_demo():
    print("🧠 DEEPBOOST CONFIDENCE MATRIX: DEMO MODE")
    print("="*50)

    # 1. Initialize a mock network (replicating a trained state)
    # In a real scenario, we'd load level4_deepboost.json
    db = DeepBoostNetwork(max_layers=2, neurons_per_layer=4)
    
    # 2. Simulate a match with 24 features
    # (Effective Strength, Fit, Tactical Deltas)
    X_sample = np.random.rand(1, 24)
    
    # Mocking the fit state so neurons can produce outputs
    # (Since neurons are XGBoost models, they'll produce random-ish but structured probs)
    print("🧬 Processing 'High-Consensus' Scenario...")
    print("   Scene: Man City (Home) vs. Mid-Table Team")
    
    # We "cheat" the mock by fitting it on a single biased sample for demonstration
    y_target = np.array([0]) # Home Win
    db.fit(np.random.rand(10, 24), np.random.randint(0, 3, 10))
    
    # 3. Calculate Confidence Matrix
    profile = db.calculate_confidence_profile(X_sample)
    
    print("\n📊 CONFIDENCE MATRIX RESULTS")
    print("-" * 30)
    print(f"Outcome:       {profile['prediction']}")
    print(f"Final Score:    {profile['confidence_score']:.2%}")
    print(f"Risk Level:     {profile['risk_rating']}")
    print("-" * 30)
    
    metrics = profile['consensus_metrics']
    print(f"Neuron Consensus: {metrics['neuron_agreement_pct']:.1f}% Agreement")
    print(f"Dominance Gap:    {metrics['probability_dominance']:.2f} (Gap to 2nd choice)")
    print(f"System Entropy:   {metrics['entropy']:.4f}")
    print("-" * 30)
    print(f"💰 BETTING SIGNAL: {profile['betting_signal']}")
    
    # 4. Show the Trace (The raw flow)
    trace = db.trace_predict(X_sample)
    print("\n🔍 LAYER TRACE (Why did it pick this?)")
    for t in trace:
        if t['layer'] == 0:
            print(f"Input: {t['features']} features injected.")
        elif isinstance(t['layer'], int):
            print(f"Layer {t['layer']}: {len(t['neurons'])} neurons fired.")
            decisions = [n['dominant'] for n in t['neurons']]
            print(f"       Neuron Votes: {decisions}")
        else:
            print(f"Output: Final aggregator probabilities: {np.round(t['probabilities'], 3)}")

if __name__ == "__main__":
    run_confidence_demo()

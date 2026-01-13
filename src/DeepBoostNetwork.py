import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DeepBoostLayer:
    """
    A single layer in the DeepBoost pipeline containing multiple XGBoost neurons.
    Point 6 & 7 of the Manifesto.
    """
    def __init__(self, n_neurons: int, max_depth: int = 3, learning_rate: float = 0.1):
        self.n_neurons = n_neurons
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.neurons: List[xgb.XGBClassifier] = []
        self.fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
        """
        Trains neurons using Cross-Validation and returns OOF (Out-Of-Fold) predictions.
        This prevents data leakage between layers.
        """
        oof_predictions = np.zeros((X.shape[0], self.n_neurons * 3)) # 3 classes: H, D, A
        self.neurons = []
        
        # We vary the random seed for each neuron to ensure diversity (Cascade principle)
        for i in range(self.n_neurons):
            neuron_oof = np.zeros((X.shape[0], 3))
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + i)
            
            # Train the neuron on CV folds
            for train_idx, val_idx in kf.split(X):
                clf = xgb.XGBClassifier(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    n_estimators=50,
                    random_state=42 + i,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                clf.fit(X[train_idx], y[train_idx])
                neuron_oof[val_idx] = clf.predict_proba(X[val_idx])
            
            # Final train on full data for inference
            final_clf = xgb.XGBClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=50,
                random_state=42 + i,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            final_clf.fit(X, y)
            self.neurons.append(final_clf)
            
            # Add to the layer's OOF matrix
            oof_predictions[:, i*3:(i+1)*3] = neuron_oof
            
        self.fitted = True
        return oof_predictions

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for inference."""
        preds = []
        for clf in self.neurons:
            preds.append(clf.predict_proba(X))
        return np.hstack(preds)

class DeepBoostNetwork:
    """
    Neural Network Architecture where every node is an XGBoost model.
    Point 6 of the Manifesto.
    """
    def __init__(self, max_layers: int = 3, neurons_per_layer: int = 4):
        self.max_layers = max_layers
        self.neurons_per_layer = neurons_per_layer
        self.layers: List[DeepBoostLayer] = []
        self.final_clf = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Point 7: Layer-wise Greedy Training.
        """
        current_X = X
        best_logloss = float('inf')
        
        for i in range(self.max_layers):
            logger.info(f"Training DeepBoost Layer {i+1}...")
            layer = DeepBoostLayer(n_neurons=self.neurons_per_layer, max_depth=3+i)
            
            # Get OOF predictions from this layer
            layer_output = layer.fit_transform(current_X, y)
            
            # Strategy: Concatenate new layer predictions with original features
            # This is the "Cascade" part of Deep Forest
            new_X = np.hstack([X, layer_output])
            
            # Evaluate if this layer actually helped
            # We check the average logloss of the layer's neurons
            layer_avg_probs = np.mean([layer_output[:, j*3:(j+1)*3] for j in range(self.neurons_per_layer)], axis=0)
            current_logloss = log_loss(y, layer_avg_probs)
            
            if current_logloss < best_logloss:
                logger.info(f"Layer {i+1} improved logloss to {current_logloss:.4f}. Accepted.")
                self.layers.append(layer)
                current_X = new_X
                best_logloss = current_logloss
            else:
                logger.info(f"Layer {i+1} did not improve performance. Stopping growth.")
                break
                
        # Final Classifier on the enriched feature set
        logger.info("Training final DeepBoost output node...")
        self.final_clf = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
        self.final_clf.fit(current_X, y)

    def trace_predict(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Hierarchical 2.0: Trace prediction flow for live visualization.
        Returns a list of layer states.
        """
        trace = []
        current_X = X
        
        # Input Layer
        trace.append({
            'layer': 0,
            'name': 'Input Layer',
            'features': current_X.shape[1],
            'type': 'raw_tactical'
        })
        
        for i, layer in enumerate(self.layers):
            layer_output = layer.transform(current_X)
            
            # Record individual neuron "activations" (probabilities)
            neurons = []
            for j in range(layer.n_neurons):
                neuron_probs = layer_output[:, j*3:(j+1)*3][0]
                neurons.append({
                    'id': f"L{i+1}N{j+1}",
                    'probs': neuron_probs.tolist(),
                    'dominant': ['Home', 'Draw', 'Away'][np.argmax(neuron_probs)]
                })
                
            trace.append({
                'layer': i + 1,
                'name': f"DeepBoost Layer {i+1}",
                'neurons': neurons,
                'output_dim': layer_output.shape[1]
            })
            
            current_X = np.hstack([X, layer_output])
            
        # Final Output
        if self.final_clf is None:
            final_probs = np.array([0.33, 0.33, 0.34]) # Default uniform
        else:
            final_probs = self.final_clf.predict_proba(current_X)[0]
        trace.append({
            'layer': 'final',
            'name': 'Decision Node',
            'probabilities': final_probs.tolist(),
            'prediction': ['Home Win', 'Draw', 'Away Win'][np.argmax(final_probs)]
        })
        
        return trace

    def calculate_confidence_profile(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Hierarchical 2.0: Point 12.
        Calculates the Confidence Matrix for a prediction.
        Measures:
        1. Final Probability Spread (Dominance)
        2. Neuron Consensus (Do all neurons agree?)
        3. Tactical Stability (Variance across layers)
        """
        current_X = X
        layer_outputs = []
        
        # Capture all intermediate activations
        for layer in self.layers:
            out = layer.transform(current_X)
            layer_outputs.append(out)
            current_X = np.hstack([X, out])
            
        if self.final_clf is None:
            return {
                'prediction': 'Unknown',
                'confidence_score': 0.0,
                'risk_rating': 'HIGH',
                'consensus_metrics': {'neuron_agreement_pct': 0, 'probability_dominance': 0, 'entropy': 0},
                'betting_signal': 'AVOID'
            }
            
        final_probs = self.final_clf.predict_proba(current_X)[0]
        prediction_idx = np.argmax(final_probs)
        raw_conf = final_probs[prediction_idx]
        
        # Neuron Consensus: What % of neurons in the last layer picked the same result?
        last_layer = self.layers[-1]
        last_layer_out = layer_outputs[-1]
        agreements = 0
        for j in range(last_layer.n_neurons):
            neuron_probs = last_layer_out[:, j*3:(j+1)*3][0]
            if np.argmax(neuron_probs) == prediction_idx:
                agreements += 1
        
        consensus_score = agreements / last_layer.n_neurons
        
        # Final Confidence Calibration
        # We weigh the raw probability (70%) and the neuron consensus (30%)
        calibrated_confidence = (raw_conf * 0.7) + (consensus_score * 0.3)
        
        # Risk Rating
        risk = "LOW" if calibrated_confidence > 0.75 else "MEDIUM" if calibrated_confidence > 0.6 else "HIGH"
        
        return {
            'prediction': ['Home Win', 'Draw', 'Away Win'][prediction_idx],
            'confidence_score': calibrated_confidence,
            'risk_rating': risk,
            'consensus_metrics': {
                'neuron_agreement_pct': consensus_score * 100,
                'probability_dominance': raw_conf - np.sort(final_probs)[-2], # Gap to 2nd choice
                'entropy': -np.sum(final_probs * np.log(final_probs + 1e-15))
            },
            'betting_signal': "STAKE" if calibrated_confidence > 0.8 else "AVOID"
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the cascaded forest.
        """
        current_X = X
        for layer in self.layers:
            layer_output = layer.transform(current_X)
            current_X = np.hstack([X, layer_output])
            
        return self.final_clf.predict_proba(current_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.final_clf.predict(self.transform_features(X))

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Returns the final enriched feature set for a given input."""
        current_X = X
        for layer in self.layers:
            layer_output = layer.transform(current_X)
            current_X = np.hstack([X, layer_output])
        return current_X

def log_loss(y_true, y_pred):
    """Simple log loss implementation since we might not have sklearn metrics everywhere."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(np.eye(y_pred.shape[1])[y_true] * np.log(y_pred), axis=1))
    return loss

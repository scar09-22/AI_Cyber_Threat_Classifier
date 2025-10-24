"""
Explainability module using SHAP for feature importance analysis.
Generates visualizations explaining model predictions.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ThreatExplainer:
    def __init__(self, model_dir='models', output_dir='outputs', data_dir='data'):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.explainer = None
        self.shap_values = None
        
    def load_model_and_data(self):
        """Load trained model and test data."""
        print("Loading model and data...")
        
        model = joblib.load(self.model_dir / 'threat_classifier.pkl')
        X_test = np.load(self.data_dir / 'X_test.npy')
        y_test = np.load(self.data_dir / 'y_test.npy')
        
        with open(self.data_dir / 'feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print(f"Loaded model and {len(X_test)} test samples")
        
        return model, X_test, y_test, feature_names
    
    def create_explainer(self, model, X_background):
        """Create SHAP explainer."""
        print("Creating SHAP explainer...")
        
        background_sample = shap.sample(X_background, min(100, len(X_background)))
        
        try:
            self.explainer = shap.TreeExplainer(model, background_sample)
            print("SHAP explainer created")
        except Exception as e:
            print(f"Using simpler explainer: {e}")
            self.explainer = shap.TreeExplainer(model)
            print("SHAP explainer created (simple mode)")
        
        return self.explainer
    
    def compute_shap_values(self, X_test, max_samples=1000):
        """Compute SHAP values for test set."""
        print(f"Computing SHAP values for {min(max_samples, len(X_test))} samples...")
        
        X_sample = X_test[:max_samples]
        
        try:
            self.shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
        except TypeError:
            print("Using older SHAP API")
            self.shap_values = self.explainer.shap_values(X_sample)
        except Exception as e:
            print(f"Computing in batches: {e}")
            batch_size = 100
            shap_list = []
            
            for i in range(0, len(X_sample), batch_size):
                batch = X_sample[i:i+batch_size]
                try:
                    batch_shap = self.explainer.shap_values(batch, check_additivity=False)
                except TypeError:
                    batch_shap = self.explainer.shap_values(batch)
                shap_list.append(batch_shap)
                if (i // batch_size) % 5 == 0:
                    print(f"Progress: {min(i+batch_size, len(X_sample))}/{len(X_sample)}")
            
            if isinstance(shap_list[0], list):
                self.shap_values = [np.vstack([b[i] for b in shap_list]) for i in range(len(shap_list[0]))]
            else:
                self.shap_values = np.vstack(shap_list)
        
        # Extract positive class for binary classification
        if isinstance(self.shap_values, list):
            print("Extracting positive class values")
            self.shap_values = self.shap_values[1]
        
        print(f"SHAP values computed (shape: {self.shap_values.shape})")
        
        return self.shap_values, X_sample
    
    def plot_feature_importance(self, shap_values, X_sample, feature_names):
        """Plot global feature importance."""
        print("Generating feature importance plot...")
        
        try:
            # Create figure with proper aspect ratio
            fig, ax = plt.subplots(figsize=(12, 10))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=feature_names,
                plot_type='bar',
                show=False,
                max_display=20
            )
            plt.tight_layout(pad=2.0)
            plt.savefig(self.output_dir / 'shap_feature_importance.png', 
                    dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved feature importance plot")
        except Exception as e:
            print(f"Could not create bar plot: {e}")

    def plot_summary(self, shap_values, X_sample, feature_names):
        """Plot SHAP summary."""
        print("Generating SHAP summary plot...")
        
        try:
            # Create figure with wider aspect ratio
            fig, ax = plt.subplots(figsize=(14, 10))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=20
            )
            plt.tight_layout(pad=2.0)
            plt.savefig(self.output_dir / 'shap_summary.png', 
                    dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved SHAP summary plot")
        except Exception as e:
            print(f"Could not create summary plot: {e}")

    
    def explain_single_prediction(self, model, X_sample, feature_names, idx=0):
        """Explain a single prediction."""
        print(f"Explaining prediction for sample {idx}...")
        
        try:
            instance = X_sample[idx:idx+1]
            prediction = model.predict(instance)[0]
            proba = model.predict_proba(instance)[0]
            
            print(f"Prediction: {'Attack' if prediction == 1 else 'Benign'}")
            print(f"Probability: {proba[prediction]:.4f}")
            
            # Compute SHAP for single instance
            try:
                shap_value = self.explainer.shap_values(instance, check_additivity=False)
            except TypeError:
                shap_value = self.explainer.shap_values(instance)
            
            # Handle multi-output
            if isinstance(shap_value, list):
                shap_value = shap_value[1]
            
            # Ensure 1D array for waterfall plot
            if len(shap_value.shape) > 1:
                shap_value = shap_value[0]
            
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_value,
                    base_values=expected_value,
                    data=instance[0],
                    feature_names=feature_names
                ),
                show=False,
                max_display=15
            )
            plt.tight_layout()
            plt.savefig(self.output_dir / f'shap_waterfall_sample_{idx}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved waterfall plot for sample {idx}")
            
            return prediction, proba
            
        except Exception as e:
            print(f"Could not explain sample {idx}: {e}")
            return None, None
    
    def get_top_features(self, shap_values, feature_names, n=15):
        """Get top N most important features."""
        print(f"Extracting top {n} features...")
        
        # Ensure 2D shape
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top N features
        n = min(n, len(mean_abs_shap), len(feature_names))
        top_indices = np.argsort(mean_abs_shap)[-n:][::-1]
        
        top_features = []
        for i in top_indices:
            idx = int(i)
            if idx < len(feature_names):
                top_features.append((feature_names[idx], float(mean_abs_shap[idx])))
        
        print("Top Features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        top_features_dict = {feature: importance for feature, importance in top_features}
        
        with open(self.output_dir / 'top_features.json', 'w') as f:
            json.dump(top_features_dict, f, indent=4)
        
        return top_features
    
    def generate_all_explanations(self, max_samples=1000):
        """Complete explainability pipeline."""
        print("\n" + "="*60)
        print("Starting Explainability Pipeline")
        print("="*60 + "\n")
        
        model, X_test, y_test, feature_names = self.load_model_and_data()
        self.create_explainer(model, X_test)
        shap_values, X_sample = self.compute_shap_values(X_test, max_samples)
        
        self.plot_feature_importance(shap_values, X_sample, feature_names)
        self.plot_summary(shap_values, X_sample, feature_names)
        
        self.explain_single_prediction(model, X_sample, feature_names, idx=0)
        self.explain_single_prediction(model, X_sample, feature_names, idx=10)
        
        top_features = self.get_top_features(shap_values, feature_names, n=15)
        
        print("\n" + "="*60)
        print("Explainability Pipeline Complete")
        print("="*60)
        
        return shap_values, top_features

if __name__ == "__main__":
    explainer = ThreatExplainer()
    shap_values, top_features = explainer.generate_all_explanations(max_samples=1000)

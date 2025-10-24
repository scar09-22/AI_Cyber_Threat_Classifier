"""
Model training module for threat classification.
Trains and evaluates RandomForest, XGBoost, and LightGBM models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ThreatClassifierTrainer:
    def __init__(self, model_dir='models', output_dir='outputs'):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.results = {}
        
    def load_processed_data(self, data_dir='data'):
        """Load preprocessed data."""
        print("Loading processed data...")
        
        data_dir = Path(data_dir)
        
        X_train = np.load(data_dir / 'X_train.npy')
        X_test = np.load(data_dir / 'X_test.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        with open(data_dir / 'feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print(f"Loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier."""
        print("\nTraining Random Forest...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rf_model.fit(X_train, y_train)
        
        self.models['RandomForest'] = rf_model
        print("Random Forest trained")
        
        return rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost classifier."""
        print("\nTraining XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        
        self.models['XGBoost'] = xgb_model
        print("XGBoost trained")
        
        return xgb_model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM classifier."""
        print("\nTraining LightGBM...")
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        
        self.models['LightGBM'] = lgb_model
        print("LightGBM trained")
        
        return lgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance."""
        print(f"\nEvaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary')
        }
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name}.png', dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot")
    
    def compare_models(self):
        """Compare all trained models."""
        print("\nModel Comparison:")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            model_name: results['metrics']
            for model_name, results in self.results.items()
        }).T
        
        print(comparison_df.to_string())
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv')
        
        # Plot comparison
        comparison_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.legend(loc='lower right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300)
        plt.close()
        
        # Select best model
        best_model_name = comparison_df['f1'].idxmax()
        print(f"\nBest Model: {best_model_name}")
        
        return best_model_name
    
    def save_best_model(self, best_model_name):
        """Save the best performing model."""
        best_model = self.models[best_model_name]
        
        model_path = self.model_dir / 'threat_classifier.pkl'
        joblib.dump(best_model, model_path)
        
        # Save model metadata
        metadata = {
            'model_type': best_model_name,
            'metrics': self.results[best_model_name]['metrics'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.model_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Best model saved: {model_path}")
    
    def train_all_models(self):
        """Complete training pipeline."""
        print("\n" + "="*60)
        print("Starting Model Training Pipeline")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_processed_data()
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        # Evaluate all models
        for model_name, model in self.models.items():
            metrics, cm = self.evaluate_model(model, X_test, y_test, model_name)
            self.plot_confusion_matrix(cm, model_name)
        
        # Compare and save best
        best_model_name = self.compare_models()
        self.save_best_model(best_model_name)
        
        print("\n" + "="*60)
        print("Training Pipeline Complete")
        print("="*60)
        
        return self.models, self.results, feature_names

if __name__ == "__main__":
    trainer = ThreatClassifierTrainer()
    models, results, feature_names = trainer.train_all_models()

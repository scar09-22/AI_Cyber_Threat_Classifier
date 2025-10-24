"""
Data preprocessing pipeline for CICIDS2017 dataset.
Handles cleaning, scaling, and class balancing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import json
from pathlib import Path

class ThreatDataPreprocessor:
    def __init__(self, data_dir='data', output_dir='data'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.scaler = StandardScaler()
        
    def load_all_data(self, sample_per_file=50000):
        """Load and combine all CSV files."""
        print("Loading all CSV files...")
        
        csv_files = sorted(list(self.data_dir.glob('*.csv')))
        print(f"Found {len(csv_files)} files")
        
        dfs = []
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] Loading {csv_file.name}...")
            try:
                df_chunk = pd.read_csv(csv_file, nrows=sample_per_file)
                dfs.append(df_chunk)
                print(f"Loaded {len(df_chunk)} rows")
            except Exception as e:
                print(f"Error: {e}")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"\nCombined dataset: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare data."""
        print("\nCleaning data...")
        
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values with median")
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(
                df_clean[numeric_columns].median()
            )
        
        # Handle infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)
        print("Handled infinite values")
        
        # Remove constant columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        constant_cols = [col for col in numeric_columns 
                        if df_clean[col].nunique() <= 1]
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            print(f"Removed {len(constant_cols)} constant columns")
        
        print(f"Final shape: {df_clean.shape}")
        
        return df_clean
    
    def prepare_features(self, df):
        """Separate features and target."""
        print("\nPreparing features and target...")
        
        # Find label column
        label_col = ' Label' if ' Label' in df.columns else 'Label'
        
        if label_col not in df.columns:
            raise ValueError("Label column not found")
        
        # Separate features and target
        X = df.drop(columns=[label_col])
        X = X.select_dtypes(include=[np.number])
        y = df[label_col]
        
        # Binary classification: BENIGN vs ATTACK
        y_binary = y.apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
        
        print(f"Features: {X.shape[1]} columns")
        print(f"Samples: {len(X)}")
        print(f"\nTarget distribution:")
        counts = y_binary.value_counts()
        print(f"Benign (0): {counts[0]:,} ({counts[0]/len(y_binary)*100:.1f}%)")
        print(f"Attack (1): {counts[1]:,} ({counts[1]/len(y_binary)*100:.1f}%)")
        
        print(f"\nOriginal attack types:")
        for attack, count in y.value_counts().head(10).items():
            print(f"{attack}: {count:,}")
        
        return X, y_binary, y
    
    def scale_features(self, X_train, X_test):
        """Standardize features."""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        print("Features scaled and scaler saved")
        
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train):
        """Balance dataset using SMOTE."""
        print("\nBalancing dataset with SMOTE...")
        
        before = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"Before: Benign={before[0]:,}, Attack={before[1]:,}")
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            after = dict(zip(*np.unique(y_balanced, return_counts=True)))
            print(f"After:  Benign={after[0]:,}, Attack={after[1]:,}")
            
            return X_balanced, y_balanced
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Using original data")
            return X_train, y_train
    
    def preprocess_pipeline(self, sample_per_file=50000, test_size=0.2, balance=True):
        """Complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("Starting Preprocessing Pipeline")
        print("="*60 + "\n")
        
        # Load data
        df = self.load_all_data(sample_per_file=sample_per_file)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Prepare features
        X, y_binary, y_original = self.prepare_features(df_clean)
        
        # Split data
        print("\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        print(f"Train: {len(X_train):,} samples")
        print(f"Test:  {len(X_test):,} samples")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Handle imbalance
        if balance:
            X_train_balanced, y_train_balanced = self.handle_imbalance(
                X_train_scaled, y_train
            )
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        # Save processed data
        print("\nSaving processed data...")
        np.save(self.output_dir / 'X_train.npy', X_train_balanced)
        np.save(self.output_dir / 'X_test.npy', X_test_scaled)
        np.save(self.output_dir / 'y_train.npy', y_train_balanced)
        np.save(self.output_dir / 'y_test.npy', y_test)
        
        # Save feature names
        feature_names = list(X.columns)
        with open(self.output_dir / 'feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        print(f"Saved to {self.output_dir}/")
        print(f"Feature names: {len(feature_names)} features")
        
        print("\n" + "="*60)
        print("Preprocessing Complete")
        print("="*60)
        
        return {
            'X_train': X_train_balanced,
            'X_test': X_test_scaled,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'feature_names': feature_names
        }

if __name__ == "__main__":
    preprocessor = ThreatDataPreprocessor()
    data = preprocessor.preprocess_pipeline(
        sample_per_file=50000,
        balance=True
    )

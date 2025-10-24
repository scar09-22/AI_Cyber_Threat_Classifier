"""
Data acquisition module for CICIDS2017 dataset.
Downloads and loads network intrusion detection data.
"""

import os
import pandas as pd
from pathlib import Path
import kaggle

class DatasetDownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_cicids2017(self):
        """Download CICIDS2017 dataset from Kaggle."""
        print("Downloading CICIDS2017 dataset...")
        
        dataset = 'chethuhn/network-intrusion-dataset'
        
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset, 
                path=self.data_dir, 
                unzip=True
            )
            print("Dataset downloaded successfully.")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Download manually from:")
            print("https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset")
            return False
    
    def load_sample(self, sample_size=100000):
        """Load a sample of the dataset."""
        print(f"Loading {sample_size} rows...")
        
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            print("No CSV files found. Please download the dataset first.")
            return None
        
        df = pd.read_csv(csv_files[0], nrows=sample_size)
        
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        return df
    
    def get_dataset_info(self):
        """Display dataset information."""
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if csv_files:
            df = pd.read_csv(csv_files[0], nrows=1000)
            print("\nDataset Overview:")
            print(f"Total Columns: {len(df.columns)}")
            print(f"Target Column: {df.columns[-1]}")
            print(f"Unique Classes: {df.iloc[:, -1].unique()}")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_cicids2017()
    df = downloader.load_sample(50000)
    if df is not None:
        downloader.get_dataset_info()

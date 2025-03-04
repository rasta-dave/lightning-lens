#!/usr/bin/env python3
"""
Train Initial Model for Online Learning

This script trains an initial model from existing CSV data files,
which can then be used as a starting point for online learning.
"""

import os
import argparse
import pandas as pd
from datetime import datetime
import glob
from src.models.trainer import ModelTrainer
from src.models.features import FeatureProcessor

def find_latest_features_file():
    """Find the most recent features CSV file"""
    files = glob.glob("data/processed/features_*.csv")
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def train_initial_model(features_path=None):
    """Train initial model from features CSV"""
    # Find latest features file if not specified
    if not features_path:
        features_path = find_latest_features_file()
        if not features_path:
            print("No features files found. Please run data collection first.")
            return False
    
    print(f"Training initial model using: {features_path}")
    
    try:
        # Load features
        features_df = pd.read_csv(features_path)
        
        if 'balance_ratio' not in features_df.columns:
            print("Error: Features file does not contain 'balance_ratio' column")
            return False
        
        # Train model
        trainer = ModelTrainer()
        model, scaler = trainer.train_model(
            features_df, 
            target_column='balance_ratio'
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/models", exist_ok=True)
        
        model_path = f"data/models/model_initial_{timestamp}.pkl"
        scaler_path = f"data/models/scaler_initial_{timestamp}.pkl"
        
        trainer.save_model(model, model_path)
        trainer.save_model(scaler, scaler_path)
        
        print(f"Successfully trained and saved initial model to {model_path}")
        print(f"This model can now be used as a starting point for online learning")
        
        return True
        
    except Exception as e:
        print(f"Error training initial model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train initial model for online learning")
    parser.add_argument("--features", help="Path to features CSV file (optional)")
    
    args = parser.parse_args()
    
    print("LightningLens Initial Model Training")
    print("====================================")
    
    train_initial_model(args.features)

if __name__ == "__main__":
    main() 
"""
Online Learning Module for LightningLens

This module provides incremental learning capabilities to adapt the model
in real-time as new transaction data arrives from the Lightning Network.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict, Tuple, Optional, Any

from src.models.features import FeatureProcessor

# Configure logging
logger = logging.getLogger("OnlineLearner")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class OnlineLearner:
    """
    Provides real-time learning capabilities for LightningLens models.
    
    This class maintains a transaction buffer and performs incremental
    model updates as new data arrives from the Lightning Network.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 buffer_size: int = 1000,
                 min_samples_for_update: int = 20,
                 update_interval: int = 10):
        """
        Initialize the online learner.
        
        Args:
            model_path: Path to initial model (if None, will create a new model)
            buffer_size: Maximum number of transactions to keep in buffer
            min_samples_for_update: Minimum samples needed before updating model
            update_interval: Update model every N transactions
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.min_samples_for_update = min_samples_for_update
        self.update_interval = update_interval
        self.transaction_count = 0
        self.feature_processor = FeatureProcessor()
        self.last_update_time = datetime.now()
        
        # Load or create model and scaler
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading initial model from {model_path}")
            self.model, self.scaler = self._load_model(model_path)
        else:
            logger.info("Creating new model for online learning")
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
            self.is_fitted = False
        
    def _load_model(self, model_path: str) -> Tuple[Any, Any]:
        """Load model and scaler from disk"""
        model = joblib.load(model_path)
        
        # Try to load corresponding scaler
        scaler_path = model_path.replace('model_', 'scaler_')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = StandardScaler()
            
        self.is_fitted = True
        return model, scaler
    
    def add_transaction(self, transaction_data: Dict) -> bool:
        """
        Add new transaction to buffer and trigger learning if needed.
        
        Args:
            transaction_data: Transaction data from Lightning Network
            
        Returns:
            bool: True if model was updated, False otherwise
        """
        # Add to buffer
        self.buffer.append(transaction_data)
        self.transaction_count += 1
        
        # Trim buffer if it exceeds max size
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        # Check if we should update the model
        should_update = (
            len(self.buffer) >= self.min_samples_for_update and
            self.transaction_count % self.update_interval == 0
        )
        
        if should_update:
            return self._update_model()
        return False
    
    def add_channel_state(self, channel_data: Dict) -> bool:
        """
        Add channel state data to the buffer.
        
        Args:
            channel_data: Channel state data from Lightning Network
            
        Returns:
            bool: True if model was updated, False otherwise
        """
        # Add to buffer with type marker
        channel_data['_data_type'] = 'channel_state'
        return self.add_transaction(channel_data)
    
    def _update_model(self) -> bool:
        """
        Perform incremental model update using buffered data.
        
        Returns:
            bool: True if model was updated successfully
        """
        if not self.buffer:
            return False
            
        try:
            # Convert buffer to DataFrame
            buffer_df = pd.DataFrame(self.buffer)
            
            # Process features
            features_df = self.feature_processor.process_features(buffer_df)
            
            if features_df.empty:
                logger.warning("No valid features extracted from buffer")
                return False
            
            # Prepare data for model update
            X = features_df.drop(['channel_id', 'timestamp'], axis=1, errors='ignore')
            y = features_df['balance_ratio']  # Target is current balance ratio
            
            if len(X) < self.min_samples_for_update:
                return False
                
            # Handle first-time fit vs. incremental update
            if not self.is_fitted:
                # First-time fit
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.model.fit(X_scaled, y)
                self.is_fitted = True
                logger.info(f"Initial model fit with {len(X)} samples")
            else:
                # Incremental update (partial_fit for scaler)
                X_scaled = self.scaler.transform(X)
                
                # For RandomForest, we need to retrain on combined data
                # In a production system, you might use a true online learning algorithm
                self.model.fit(X_scaled, y)
                logger.info(f"Model updated with {len(X)} new samples")
            
            # Save snapshot periodically
            current_time = datetime.now()
            time_diff = (current_time - self.last_update_time).total_seconds()
            if time_diff > 300:  # Save every 5 minutes
                self._save_model_snapshot()
                self.last_update_time = current_time
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return False
    
    def _save_model_snapshot(self):
        """Save current model and scaler to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("data/models", exist_ok=True)
            
            model_path = f"data/models/model_online_{timestamp}.pkl"
            scaler_path = f"data/models/scaler_online_{timestamp}.pkl"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Saved model snapshot to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model snapshot: {str(e)}")
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the current model.
        
        Args:
            features_df: DataFrame with processed features
            
        Returns:
            np.ndarray: Predicted optimal balance ratios
        """
        if not self.is_fitted:
            # Return default predictions if model isn't trained yet
            return np.full(len(features_df), 0.5)
            
        try:
            # Prepare features
            X = features_df.drop(['channel_id', 'timestamp'], axis=1, errors='ignore')
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.full(len(features_df), 0.5) 
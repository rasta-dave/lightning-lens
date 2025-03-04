#!/usr/bin/env python3
"""
HTTP server to handle Lightning Network simulation API requests and generate rebalancing suggestions
"""
import sys
import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.trainer import ModelTrainer
from src.models.features import FeatureProcessor
from src.utils.config import load_config
from src.models.online_learner import OnlineLearner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lightning_lens_http.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LightningLensHTTP")

app = Flask(__name__)

# Global variables to store data and suggestions
transaction_history = []
channel_data = {}
rebalance_suggestions = []

# Initialize ML components
feature_processor = FeatureProcessor()
trainer = ModelTrainer()
config = load_config()

# Load the latest model
def load_latest_model():
    model_dir = 'data/models'
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
        if models:
            latest_model = sorted(models)[-1]
            model_path = os.path.join(model_dir, latest_model)
            scaler_path = os.path.join(model_dir, latest_model.replace('model_', 'scaler_'))
            logger.info(f"Using model: {model_path}")
            
            try:
                model, scaler = trainer.load_model(model_path, scaler_path)
                return model, scaler
            except Exception as e:
                logger.error(f"Error loading model: {e}")
    
    logger.warning("No model found. Please train a model first.")
    return None, None

model, scaler = load_latest_model()

# Add this after initializing the Flask app
online_learner = OnlineLearner()
logger.info("Initialized online learning module")

@app.route('/api/update', methods=['POST'])
def update():
    global channel_data, transaction_history, online_learner
    
    data = request.json
    event_type = data.get('event_type')
    
    logger.info(f"Received {event_type} update")
    
    if event_type == 'transaction':
        # Process transaction data
        transaction = {
            'timestamp': datetime.now(),
            'sender': data.get('sender'),
            'receiver': data.get('receiver'),
            'amount': data.get('amount'),
            'success': data.get('success', False)
        }
        transaction_history.append(transaction)
        
        # Add to online learner
        online_learner.add_transaction(transaction)
        
    elif event_type == 'channel_state':
        # Process channel state data
        channel_id = data.get('channel_id')
        channel_data[channel_id] = {
            'node': data.get('node'),
            'remote_pubkey': data.get('remote_pubkey'),
            'capacity': data.get('capacity'),
            'local_balance': data.get('local_balance'),
            'remote_balance': data.get('remote_balance'),
            'timestamp': datetime.now()
        }
        
        # Add to online learner
        online_learner.add_channel_state(channel_data[channel_id])
    
    # Generate suggestions using the online model
    try:
        suggestions = generate_suggestions(use_online_model=True)
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        suggestions = []
    
    return jsonify({"status": "success", "suggestions": suggestions})

@app.route('/api/get_suggestions', methods=['GET'])
def get_suggestions():
    global rebalance_suggestions
    
    # Log the request
    logger.info(f"Suggestion request received. Returning {len(rebalance_suggestions)} suggestions")
    
    # Return current suggestions
    return jsonify(rebalance_suggestions)

def generate_suggestions(use_online_model=False):
    global model, scaler, channel_data, online_learner
    
    if not channel_data:
        return []
    
    try:
        # Prepare channel data for feature extraction
        channels_df = pd.DataFrame([
            {
                'timestamp': data['timestamp'],
                'channel_id': channel_id,
                'capacity': data['capacity'],
                'local_balance': data['local_balance'],
                'remote_balance': data['remote_balance'],
                'remote_pubkey': data['remote_pubkey'],
                'balance_ratio': data['local_balance'] / data['capacity'] if data['capacity'] > 0 else 0
            }
            for channel_id, data in channel_data.items()
        ])
        
        # Process features
        feature_processor = FeatureProcessor()
        features_df = feature_processor.process_features(channels_df)
        
        if features_df.empty:
            logger.warning("No valid features to generate suggestions")
            return []
        
        # Make predictions
        if use_online_model and online_learner.is_fitted:
            # Use online model
            predictions = online_learner.predict(features_df)
            logger.info("Generated suggestions using online model")
        elif model is not None and scaler is not None:
            # Use batch-trained model
            X = features_df.drop(['channel_id', 'timestamp'], axis=1)
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            logger.info("Generated suggestions using batch model")
        else:
            logger.warning("No model available for predictions")
            return []
        
        # Calculate adjustments needed
        features_df['predicted_optimal_ratio'] = predictions
        features_df['adjustment_needed'] = features_df['predicted_optimal_ratio'] - features_df['balance_ratio']
        
        # Sort by absolute adjustment needed
        features_df['abs_adjustment'] = features_df['adjustment_needed'].abs()
        features_df = features_df.sort_values('abs_adjustment', ascending=False)
        
        # Generate suggestions
        suggestions = []
        for _, row in features_df.iterrows():
            channel_id = row['channel_id']
            current = row['balance_ratio']
            optimal = row['predicted_optimal_ratio']
            adjustment = row['adjustment_needed']
            
            # Only suggest significant adjustments
            if abs(adjustment) > 0.1:
                direction = "increase" if adjustment > 0 else "decrease"
                suggestions.append({
                    'channel_id': channel_id,
                    'current_ratio': float(current),
                    'optimal_ratio': float(optimal),
                    'adjustment': float(adjustment),
                    'direction': direction,
                    'priority': 'high' if abs(adjustment) > 0.3 else 'medium'
                })
        
        return suggestions[:5]  # Return top 5 suggestions
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return []

# Add these new functions for continuous learning

def save_transaction_data():
    """Save transaction and channel data to CSV for model training"""
    global transaction_history, channel_data
    
    if not transaction_history:
        logger.info("No transaction data to save")
        return
    
    try:
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save transaction history
        tx_df = pd.DataFrame(transaction_history)
        tx_path = f"data/processed/transactions_{timestamp}.csv"
        tx_df.to_csv(tx_path, index=False)
        logger.info(f"Saved {len(transaction_history)} transactions to {tx_path}")
        
        # Save channel data
        channel_rows = []
        for node, channels in channel_data.items():
            for channel in channels:
                try:
                    channel_rows.append({
                        'node': node,
                        'remote_pubkey': channel.get('remote_pubkey', 'unknown'),
                        'capacity': int(channel.get('capacity', 0)),
                        'local_balance': int(channel.get('local_balance', 0)),
                        'remote_balance': int(channel.get('remote_balance', 0)),
                        'timestamp': datetime.now().isoformat()
                    })
                except (ValueError, TypeError):
                    pass
        
        if channel_rows:
            channel_df = pd.DataFrame(channel_rows)
            channel_path = f"data/processed/channels_{timestamp}.csv"
            channel_df.to_csv(channel_path, index=False)
            logger.info(f"Saved {len(channel_rows)} channel states to {channel_path}")
            
            # Create features file for training
            features_df = prepare_features()
            if features_df is not None:
                features_path = f"data/processed/features_{timestamp}.csv"
                features_df.to_csv(features_path, index=False)
                logger.info(f"Saved features to {features_path}")
                
                # Return the features path for training
                return features_path
    
    except Exception as e:
        logger.error(f"Error saving data: {e}")
    
    return None

def retrain_model():
    """Retrain the model with new data"""
    global model, scaler, trainer
    
    # Save current data and get features path
    features_path = save_transaction_data()
    
    if not features_path or not os.path.exists(features_path):
        logger.warning("No features available for training")
        return
    
    try:
        logger.info("Starting model retraining...")
        
        # Load features
        features_df = pd.read_csv(features_path)
        
        # Add target column (using current balance_ratio as a simple target)
        # In a real system, you might want a more sophisticated target
        features_df['optimal_ratio'] = features_df['balance_ratio']
        
        # Train model
        new_model, new_scaler = trainer.train_model(
            features_df, 
            target_column='optimal_ratio'
        )
        
        # Update global model and scaler
        model = new_model
        scaler = new_scaler
        
        logger.info("Model retraining complete")
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")

# Add a background task for periodic operations
def start_background_tasks():
    """Start background tasks for data saving and model retraining"""
    import threading
    import time
    
    def background_worker():
        save_interval = 300  # Save data every 5 minutes
        retrain_interval = 1800  # Retrain model every 30 minutes
        
        last_save = time.time()
        last_retrain = time.time()
        
        while True:
            current_time = time.time()
            
            # Save data periodically
            if current_time - last_save >= save_interval:
                logger.info("Running scheduled data save")
                save_transaction_data()
                last_save = current_time
            
            # Retrain model periodically
            if current_time - last_retrain >= retrain_interval:
                logger.info("Running scheduled model retraining")
                retrain_model()
                last_retrain = current_time
            
            # Sleep to avoid high CPU usage
            time.sleep(60)
    
    # Start the background thread
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()
    logger.info("Background tasks started")

# Update the main function to start background tasks
if __name__ == '__main__':
    # Create data directories if they don't exist
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Load model on startup
    model, scaler = load_latest_model()
    
    # Start background tasks
    start_background_tasks()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000) 
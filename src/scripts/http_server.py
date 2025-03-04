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

@app.route('/api/update', methods=['POST'])
def update():
    global channel_data, transaction_history
    
    data = request.json
    event_type = data.get('event_type')
    
    logger.info(f"Received {event_type} update")
    
    if event_type == 'transaction':
        # Store transaction data
        transaction = data.get('transaction', {})
        transaction_history.append({
            'timestamp': data.get('timestamp'),
            'sender': transaction.get('sender'),
            'receiver': transaction.get('receiver'),
            'amount': transaction.get('amount'),
            'success': transaction.get('success')
        })
        
        # Update channel data
        new_channels = data.get('channels', {})
        channel_data.update(new_channels)
        
        # Generate new suggestions if we have a model
        if model is not None and len(channel_data) > 0:
            generate_suggestions()
    
    return jsonify({"status": "success"})

@app.route('/api/get_suggestions', methods=['GET'])
def get_suggestions():
    global rebalance_suggestions
    
    # Log the request
    logger.info(f"Suggestion request received. Returning {len(rebalance_suggestions)} suggestions")
    
    # Return current suggestions
    return jsonify(rebalance_suggestions)

def generate_suggestions():
    """Generate rebalancing suggestions based on current channel data"""
    global rebalance_suggestions, channel_data, model, scaler
    
    if not model or not scaler:
        logger.warning("Cannot generate suggestions: No model loaded")
        return
    
    try:
        # Convert channel data to features
        features = prepare_features()
        
        if features is None or features.empty:
            logger.warning("No valid features to generate suggestions")
            return
        
        # Make predictions
        predictions = trainer.predict(model, features)
        
        # Create suggestions based on predictions
        suggestions = []
        for i, row in features.iterrows():
            channel_id = row['channel_id']
            current_ratio = row['balance_ratio']
            optimal_ratio = predictions[i]
            
            # Calculate adjustment needed
            adjustment = optimal_ratio - current_ratio
            
            # Only suggest significant adjustments
            if abs(adjustment) > 0.1:  # 10% threshold
                # Find node names from channel_id
                for node, channels in channel_data.items():
                    for channel in channels:
                        # This is a simplified matching - you may need to adjust based on your data structure
                        if str(channel_id) in str(channel):
                            # Determine direction based on adjustment
                            if adjustment > 0:
                                # Need to increase local balance
                                from_node = node
                                to_node = "peer"  # You'll need to determine the actual peer
                            else:
                                # Need to decrease local balance
                                from_node = "peer"  # You'll need to determine the actual peer
                                to_node = node
                            
                            # Calculate amount in sats (example)
                            capacity = channel.get('capacity', 1000000)
                            amount = int(abs(adjustment) * capacity)
                            
                            suggestions.append({
                                'from_node': from_node,
                                'to_node': to_node,
                                'amount': amount,
                                'channel_id': channel_id,
                                'current_ratio': current_ratio,
                                'optimal_ratio': optimal_ratio,
                                'adjustment': adjustment
                            })
        
        # Update global suggestions
        rebalance_suggestions = suggestions
        logger.info(f"Generated {len(suggestions)} rebalancing suggestions")
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")

def prepare_features():
    """Prepare features from channel data for prediction"""
    global channel_data
    
    if not channel_data:
        return None
    
    try:
        rows = []
        timestamp = datetime.now()
        
        # Process each node's channels
        for node, channels in channel_data.items():
            for channel in channels:
                try:
                    # Extract channel data and convert to appropriate types
                    capacity = int(channel.get('capacity', 0))
                    local_balance = int(channel.get('local_balance', 0))
                    remote_balance = int(channel.get('remote_balance', 0))
                    
                    # Skip invalid channels
                    if capacity <= 0:
                        continue
                    
                    # Calculate balance ratio
                    balance_ratio = local_balance / capacity if capacity > 0 else 0
                    
                    # Create a row for this channel
                    rows.append({
                        'timestamp': timestamp,
                        'channel_id': str(channel.get('remote_pubkey', 'unknown')),
                        'capacity': capacity,
                        'local_balance': local_balance,
                        'remote_balance': remote_balance,
                        'balance_ratio': balance_ratio
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping channel due to data error: {e}")
                    logger.debug(f"Problem channel data: {channel}")
        
        if not rows:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Process features
        features_df = feature_processor.process_features(df)
        return features_df
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return None

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
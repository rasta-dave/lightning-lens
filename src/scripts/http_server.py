#!/usr/bin/env python3
"""
HTTP Server for LightningLens

This script starts an HTTP server that receives channel state and transaction updates,
processes them using the model, and provides recommendations.
"""

import os
import sys
import argparse
import logging
import json
import pickle
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the model and online learning modules
from src.models.online_learner import OnlineLearner

# Set up logging
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

# Global variables
model = None
online_learner = None
data_buffer = []
last_save_time = time.time()
last_retrain_time = time.time()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LightningLens HTTP Server')
    parser.add_argument('--model', type=str, default='data/models/model_initial_20250305_144628.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the HTTP server on')
    return parser.parse_args()

def load_model(model_path):
    """Load the trained model from a file"""
    try:
        logger.info(f"Using model: {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def save_data():
    """Save the accumulated data to a file"""
    global data_buffer, last_save_time
    
    if not data_buffer:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/http_data_{timestamp}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(data_buffer, f)
    
    logger.info(f"Saved {len(data_buffer)} records to {filename}")
    data_buffer = []
    last_save_time = time.time()

def retrain_model():
    """Retrain the model using accumulated data"""
    global model, online_learner, last_retrain_time
    
    if not online_learner:
        return
    
    try:
        # Retrain the model
        online_learner.retrain()
        model = online_learner.get_model()
        
        # Save the retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"data/models/model_retrained_{timestamp}.pkl"
        
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Retrained model saved to {model_filename}")
        last_retrain_time = time.time()
    
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")

def background_tasks():
    """Run background tasks periodically"""
    while True:
        # Save data every 5 minutes
        if time.time() - last_save_time >= 300:  # 5 minutes
            save_data()
        
        # Retrain model every 30 minutes
        if time.time() - last_retrain_time >= 1800:  # 30 minutes
            retrain_model()
        
        # Sleep for a bit to avoid consuming too much CPU
        time.sleep(10)

@app.route('/api/update', methods=['POST'])
def update():
    """Receive channel state or transaction updates"""
    try:
        data = request.json
        event_type = data.get('event_type', 'unknown')
        
        logger.info(f"Received {event_type} update")
        
        # Add to data buffer
        data_buffer.append(data)
        
        # Process with model
        if model and online_learner:
            try:
                # Prepare features manually since OnlineLearner doesn't have prepare_features
                features = [
                    data.get('capacity', 0),
                    data.get('local_balance', 0),
                    data.get('remote_balance', 0),
                    data.get('balance_ratio', 0.0),
                    data.get('tx_count', 0),
                    data.get('success_rate', 1.0),
                    data.get('avg_amount', 0)
                ]
                
                # Make prediction
                prediction = model.predict([features])[0]
                
                # Update model with new data (if update method exists)
                if hasattr(online_learner, 'update'):
                    online_learner.update(features, prediction)
                
                return jsonify({
                    "status": "success",
                    "prediction": float(prediction),
                    "message": "Update processed successfully"
                })
            
            except Exception as e:
                logger.error(f"Error preparing features: {str(e)}")
                return jsonify({
                    "status": "success",
                    "message": "Update received but not processed due to feature mismatch"
                })
        
        return jsonify({
            "status": "success",
            "message": "Update received"
        })
    
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def main():
    """Main function"""
    global model, online_learner
    
    # Parse command line arguments
    args = parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Initialize online learning
    online_learner = OnlineLearner(model_path=args.model)
    logger.info("Initialized online learning module")
    
    # Load the model again (in case online_learner modified it)
    model = load_model(args.model)
    
    # Start background tasks
    bg_thread = threading.Thread(target=background_tasks, daemon=True)
    bg_thread.start()
    logger.info("Background tasks started")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=args.port)

if __name__ == "__main__":
    main() 
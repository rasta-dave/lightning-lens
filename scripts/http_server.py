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
import pandas as pd
import numpy as np

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
transaction_success_rates = []  # Track success rates over time
model_performance_metrics = []  # Track various performance metrics

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
    global model, online_learner, last_retrain_time, model_performance_metrics
    
    if not online_learner:
        return
    
    try:
        # Calculate current performance metrics before retraining
        current_metrics = calculate_performance_metrics()
        
        # Retrain the model
        logger.info("Retraining model with new data...")
        online_learner.retrain()
        model = online_learner.get_model()
        
        # Save the retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"data/models/model_retrained_{timestamp}.pkl"
        scaler_filename = f"data/models/scaler_retrained_{timestamp}.pkl"
        
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Save the scaler if available
        if hasattr(online_learner, 'scaler') and online_learner.scaler is not None:
            with open(scaler_filename, 'wb') as f:
                pickle.dump(online_learner.scaler, f)
            logger.info(f"Scaler saved to {scaler_filename}")
        
        logger.info(f"Retrained model saved to {model_filename}")
        last_retrain_time = time.time()
        
        # Calculate new performance metrics after retraining
        new_metrics = calculate_performance_metrics()
        
        # Calculate improvement
        improvement = {
            'timestamp': timestamp,
            'before': current_metrics,
            'after': new_metrics,
            'improvement': {
                k: new_metrics.get(k, 0) - current_metrics.get(k, 0)
                for k in new_metrics.keys()
            }
        }
        
        # Store metrics
        model_performance_metrics.append(improvement)
        
        # Save metrics to file
        with open('data/models/performance_metrics.json', 'w') as f:
            json.dump(model_performance_metrics, f, indent=2)
        
        # Log improvement
        logger.info(f"Model improvement: {improvement['improvement']}")
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")

def calculate_performance_metrics():
    """Calculate current model performance metrics"""
    try:
        # Get the latest predictions
        latest_predictions = pd.read_csv('data/predictions/latest_predictions.csv')
        
        # Calculate metrics
        metrics = {
            'avg_adjustment': abs(latest_predictions['adjustment_needed']).mean(),
            'max_adjustment': abs(latest_predictions['adjustment_needed']).max(),
            'balanced_channels': (abs(latest_predictions['adjustment_needed']) < 0.1).mean(),
            'severely_imbalanced': (abs(latest_predictions['adjustment_needed']) > 0.4).mean()
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {}

def background_tasks():
    """Run background tasks periodically"""
    global model, online_learner, last_save_time, last_retrain_time
    
    last_prediction_time = 0
    prediction_interval = 300  # Generate predictions every 5 minutes
    
    while True:
        # Save data every 5 minutes
        if time.time() - last_save_time >= 300:  # 5 minutes
            save_data()
        
        # Retrain model every 30 minutes
        if time.time() - last_retrain_time >= 1800:  # 30 minutes
            retrain_model()
        
        # Generate predictions periodically
        current_time = time.time()
        if current_time - last_prediction_time > prediction_interval:
            try:
                logger.info("Generating predictions...")
                # Import here to avoid circular imports
                from scripts.generate_predictions import generate_predictions
                if generate_predictions():
                    logger.info("Predictions generated successfully")
                    
                    # Also generate visualizations
                    try:
                        from scripts.visualize_model_evolution import visualize_model_evolution
                        if visualize_model_evolution():
                            logger.info("Model evolution visualizations generated")
                    except Exception as viz_error:
                        logger.error(f"Error generating visualizations: {str(viz_error)}")
                else:
                    logger.warning("Failed to generate predictions")
                last_prediction_time = current_time
            except Exception as e:
                logger.error(f"Error generating predictions: {str(e)}")
        
        time.sleep(10)  # Check every 10 seconds

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

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get model performance metrics"""
    try:
        if not model_performance_metrics:
            return jsonify({
                "status": "info",
                "message": "No performance metrics available yet"
            })
        
        # Return the latest metrics
        return jsonify({
            "status": "success",
            "metrics": model_performance_metrics[-1]
        })
    
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/dashboard', methods=['GET'])
def generate_dashboard():
    """Generate the learning dashboard"""
    try:
        from scripts.learning_dashboard import generate_dashboard
        if generate_dashboard():
            return jsonify({
                "status": "success",
                "message": "Dashboard generated successfully",
                "url": "/dashboard/"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to generate dashboard"
            }), 500
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
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
    
    # Initialize online learning with custom parameters
    online_learner = OnlineLearner(
        model_path=args.model,
        buffer_size=500,           # Increase for more data before retraining
        min_samples_for_update=50, # Minimum samples needed to trigger retraining
        update_interval=15         # Minutes between update attempts
    )
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
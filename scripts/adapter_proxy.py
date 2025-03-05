#!/usr/bin/env python3
"""
Feature Adapter Proxy for LightningLens

This script acts as a proxy between the simulation and the HTTP server,
transforming the data to match the expected format.
"""

import os
import sys
import json
import requests
from flask import Flask, request, jsonify
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("adapter_proxy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdapterProxy")

app = Flask(__name__)

# The real HTTP server
REAL_SERVER = "http://localhost:5001/api/update"

def adapt_channel_state(channel_data):
    """
    Adapt channel state data to match the format expected by the model
    
    Args:
        channel_data: Dictionary containing channel state information
        
    Returns:
        Dictionary with adapted features
    """
    # Create a new dictionary with the required format
    adapted_data = {
        "event_type": channel_data.get("event_type", "channel_state"),
        "channel_id": channel_data.get("channel_id", None),
        "capacity": channel_data.get("capacity", 0),
        "local_balance": channel_data.get("local_balance", 0),
        "remote_balance": channel_data.get("remote_balance", 0),
        "balance_ratio": channel_data.get("balance_ratio", 0.0),
        "tx_count": channel_data.get("tx_count", 0),
        "success_rate": channel_data.get("success_rate", 1.0),
        "avg_amount": channel_data.get("avg_amount", 0)
    }
    
    # Generate channel_id if missing
    if not adapted_data["channel_id"] and "node" in channel_data and "remote_pubkey" in channel_data:
        adapted_data["channel_id"] = f"{channel_data['node']}_{channel_data['remote_pubkey'][:8]}"
    
    # Calculate balance_ratio if missing
    if adapted_data["balance_ratio"] == 0.0 and adapted_data["capacity"] > 0 and adapted_data["local_balance"] > 0:
        adapted_data["balance_ratio"] = float(adapted_data["local_balance"]) / float(adapted_data["capacity"])
    
    # Remove any fields that aren't in our expected format
    for key in list(adapted_data.keys()):
        if key not in ["event_type", "channel_id", "capacity", "local_balance", "remote_balance", 
                      "balance_ratio", "tx_count", "success_rate", "avg_amount"]:
            del adapted_data[key]
    
    # Log the transformation
    logger.info(f"Transformed data: {json.dumps(adapted_data)}")
    
    return adapted_data

@app.route('/api/update', methods=['POST'])
def proxy_update():
    """Receive data from simulation, adapt it, and forward to real server"""
    try:
        data = request.json
        logger.info(f"Received {data.get('event_type', 'unknown')} update: {json.dumps(data)}")
        
        # Adapt the data
        adapted_data = adapt_channel_state(data)
        
        # Forward to real server
        response = requests.post(REAL_SERVER, json=adapted_data, timeout=2)
        
        if response.status_code == 200:
            logger.info("Successfully forwarded to real server")
            return jsonify({"status": "success"})
        else:
            logger.warning(f"Real server returned status code: {response.status_code}")
            return jsonify({"status": "error", "message": f"Real server returned {response.status_code}"}), 500
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting adapter proxy on port 5050")
    app.run(host='0.0.0.0', port=5050) 
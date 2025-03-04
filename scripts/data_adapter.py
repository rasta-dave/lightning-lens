#!/usr/bin/env python3
"""
LightningLens Data Adapter

This script adapts data from your Lightning Network simulation to the format
expected by the LightningLens online learning module.
"""

import requests
import json
import time
import sys
import os
from datetime import datetime

# Configuration
HTTP_SERVER_URL = "http://localhost:5000/api/update"

def adapt_channel_state(channel_data):
    """
    Adapt channel state data to the format expected by the online learner
    
    Expected format:
    {
        "event_type": "channel_state",
        "channel_id": "lnd-alice_lnd-bob",
        "node": "lnd-alice",
        "remote_pubkey": "lnd-bob",
        "capacity": 1000000,
        "local_balance": 564329,
        "remote_balance": 435671,
        "balance_ratio": 0.564329  # This is what's missing
    }
    """
    # If balance_ratio is missing, calculate it
    if 'balance_ratio' not in channel_data and 'local_balance' in channel_data and 'capacity' in channel_data:
        channel_data['balance_ratio'] = channel_data['local_balance'] / channel_data['capacity']
    
    # If channel_id is missing, create it
    if 'channel_id' not in channel_data and 'node' in channel_data and 'remote_pubkey' in channel_data:
        channel_data['channel_id'] = f"{channel_data['node']}_{channel_data['remote_pubkey']}"
    
    return channel_data

def send_to_server(data):
    """Send adapted data to the HTTP server"""
    try:
        # Adapt data based on event type
        if data.get('event_type') == 'channel_state':
            data = adapt_channel_state(data)
        
        # Send to server
        response = requests.post(HTTP_SERVER_URL, json=data)
        if response.status_code == 200:
            print(f"Successfully sent {data.get('event_type', 'unknown')} data to server")
            return True
        else:
            print(f"Error sending data: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sending data: {str(e)}")
        return False

def main():
    """Main function to process command line input"""
    if len(sys.argv) < 2:
        print("Usage: python data_adapter.py <json_data>")
        print("Example: python data_adapter.py '{\"event_type\": \"channel_state\", \"node\": \"lnd-alice\", \"remote_pubkey\": \"lnd-bob\", \"capacity\": 1000000, \"local_balance\": 564329, \"remote_balance\": 435671}'")
        return
    
    try:
        # Parse JSON data from command line
        data = json.loads(sys.argv[1])
        send_to_server(data)
    except json.JSONDecodeError:
        print("Error: Invalid JSON data")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
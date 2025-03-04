#!/usr/bin/env python3
"""
LightningLens WebSocket Client

Connects to a Lightning Network simulation via WebSocket, processes transaction data,
and provides rebalancing suggestions based on ML predictions.
"""

import asyncio
import json
import logging
import os
import pandas as pd
import websockets
from datetime import datetime
import argparse
import numpy as np
import sys
from pathlib import Path

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
        logging.FileHandler("lightning_lens_ws.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LightningLens")

class LightningLensClient:
    """Client that connects to Lightning Network simulation and provides rebalancing suggestions"""
    
    def __init__(self, websocket_uri="ws://localhost:6789", model_path=None, scaler_path=None):
        """Initialize the client with model paths and WebSocket URI"""
        self.websocket_uri = websocket_uri
        self.channel_data = {}  # Store latest channel data
        self.transaction_history = []  # Store recent transactions
        self.feature_processor = FeatureProcessor()
        self.trainer = ModelTrainer()
        self.config = load_config()
        
        # Find latest model if not specified
        if not model_path:
            model_dir = 'data/models'
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
                if models:
                    latest_model = sorted(models)[-1]
                    model_path = os.path.join(model_dir, latest_model)
                    scaler_path = os.path.join(model_dir, latest_model.replace('model_', 'scaler_'))
                    logger.info(f"Using latest model: {model_path}")
                else:
                    logger.warning("No models found. Will use default balance targets.")
            else:
                logger.warning("Model directory not found. Will use default balance targets.")
        
        # Load model if available
        self.model = None
        self.scaler = None
        if model_path and scaler_path and os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model, self.scaler = self.trainer.load_model(model_path, scaler_path)
                self.trainer.scaler = self.scaler
                logger.info(f"Successfully loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("No model available. Using default balance targets.")
    
    async def connect(self):
        """Connect to the WebSocket server and process messages"""
        logger.info(f"Connecting to {self.websocket_uri}...")
        
        try:
            async with websockets.connect(self.websocket_uri) as websocket:
                logger.info("Connected to Lightning Network simulation")
                
                # Send initial message
                await websocket.send(json.dumps({
                    "type": "register",
                    "client": "lightning_lens",
                    "message": "LightningLens AI optimization connected"
                }))
                
                # Process incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.process_message(data, websocket)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {message}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            # Try to reconnect after a delay
            logger.info("Attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)
            await self.connect()
    
    async def process_message(self, data, websocket):
        """Process incoming WebSocket messages"""
        message_type = data.get("event_type")
        
        if message_type == "transaction":
            # Process transaction data
            logger.info(f"Received transaction: {data['transaction']['sender']} -> {data['transaction']['receiver']}")
            
            # Update channel data
            self.update_channel_data(data.get("channels", {}))
            
            # Store transaction in history
            self.transaction_history.append({
                "timestamp": data["timestamp"],
                "sender": data["transaction"]["sender"],
                "receiver": data["transaction"]["receiver"],
                "amount": data["transaction"]["amount"],
                "success": data["transaction"]["success"]
            })
            
            # Limit history size
            if len(self.transaction_history) > 100:
                self.transaction_history = self.transaction_history[-100:]
            
            # Generate suggestions after processing data
            suggestions = self.generate_suggestions()
            if suggestions:
                await self.send_suggestions(websocket, suggestions)
        
        elif message_type == "snapshot":
            # Process channel snapshot
            logger.info("Received channel snapshot")
            self.update_channel_data(data.get("channels", {}))
            
            # Generate suggestions based on snapshot
            suggestions = self.generate_suggestions()
            if suggestions:
                await self.send_suggestions(websocket, suggestions)
        
        elif message_type == "request_suggestions":
            # Explicit request for suggestions
            logger.info("Received request for suggestions")
            suggestions = self.generate_suggestions()
            await self.send_suggestions(websocket, suggestions)
    
    def update_channel_data(self, channels):
        """Update stored channel data with new information"""
        for node, channel_list in channels.items():
            self.channel_data[node] = channel_list
    
    def generate_suggestions(self):
        """Generate rebalancing suggestions based on channel data and model predictions"""
        if not self.channel_data:
            logger.warning("No channel data available for generating suggestions")
            return []
        
        suggestions = []
        
        # Convert channel data to features
        features_df = self.prepare_features()
        
        if features_df is None or features_df.empty:
            logger.warning("Could not prepare features from channel data")
            return []
        
        # Make predictions if model is available
        if self.model and self.scaler:
            try:
                predictions = self.trainer.predict(self.model, features_df)
                features_df['predicted_optimal_ratio'] = predictions
                features_df['adjustment_needed'] = predictions - features_df['balance_ratio']
            except Exception as e:
                logger.error(f"Error making predictions: {e}")
                # Fall back to default balance target
                features_df['predicted_optimal_ratio'] = 0.5
                features_df['adjustment_needed'] = 0.5 - features_df['balance_ratio']
        else:
            # Use default balance target of 0.5 (balanced channels)
            features_df['predicted_optimal_ratio'] = 0.5
            features_df['adjustment_needed'] = 0.5 - features_df['balance_ratio']
        
        # Find channels that need significant rebalancing
        threshold = self.config.get('rebalancing', {}).get('threshold_pct', 10) / 100
        rebalance_candidates = features_df[abs(features_df['adjustment_needed']) > threshold]
        
        # Sort by adjustment magnitude (largest first)
        rebalance_candidates = rebalance_candidates.sort_values(by='adjustment_needed', key=abs, ascending=False)
        
        # Generate suggestions for top 3 channels
        for _, row in rebalance_candidates.head(3).iterrows():
            channel_id = row['channel_id']
            adjustment = row['adjustment_needed']
            
            # Find nodes for this channel
            from_node = None
            to_node = None
            
            for node, channels in self.channel_data.items():
                for channel in channels:
                    if str(channel_id) in str(channel):
                        from_node = node
                        # Find the remote node
                        for other_node in self.channel_data.keys():
                            if other_node != node:
                                to_node = other_node
                                break
                        break
                if from_node and to_node:
                    break
            
            if not from_node or not to_node:
                logger.warning(f"Could not identify nodes for channel {channel_id}")
                continue
            
            # Calculate amount to rebalance (as percentage of capacity)
            capacity = features_df.loc[features_df['channel_id'] == channel_id, 'capacity'].values[0]
            amount = int(abs(adjustment) * capacity)
            
            # Apply min/max constraints
            min_tx = self.config.get('rebalancing', {}).get('min_tx_value_sats', 10000)
            max_tx = self.config.get('rebalancing', {}).get('max_tx_value_sats', 1000000)
            amount = max(min_tx, min(max_tx, amount))
            
            # Determine direction
            if adjustment < 0:  # Too much local balance
                suggestion = {
                    "from_node": from_node,
                    "to_node": to_node,
                    "amount": amount,
                    "reason": f"Channel is {abs(adjustment)*100:.1f}% too full on local side"
                }
            else:  # Too little local balance
                suggestion = {
                    "from_node": to_node,
                    "to_node": from_node,
                    "amount": amount,
                    "reason": f"Channel is {abs(adjustment)*100:.1f}% too empty on local side"
                }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def prepare_features(self):
        """Convert channel data to features for model input"""
        rows = []
        timestamp = datetime.now()
        
        for node, channels in self.channel_data.items():
            for channel in channels:
                try:
                    # Extract channel data
                    capacity = int(channel.get('capacity', 0))
                    local_balance = int(channel.get('local_balance', 0))
                    remote_balance = int(channel.get('remote_balance', 0))
                    remote_pubkey = channel.get('remote_pubkey', '')
                    
                    # Skip channels with zero capacity
                    if capacity == 0:
                        continue
                    
                    # Calculate balance ratio
                    balance_ratio = local_balance / capacity if capacity > 0 else 0.5
                    
                    # Create row
                    row = {
                        'channel_id': f"{node}_{remote_pubkey[:8]}",
                        'timestamp': timestamp,
                        'capacity': capacity,
                        'local_balance': local_balance,
                        'remote_balance': remote_balance,
                        'remote_pubkey': remote_pubkey,
                        'balance_ratio': balance_ratio
                    }
                    rows.append(row)
                except Exception as e:
                    logger.error(f"Error processing channel data: {e}")
        
        if not rows:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Process features
        try:
            features_df = self.feature_processor.process_features(df)
            return features_df
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            return None
    
    async def send_suggestions(self, websocket, suggestions):
        """Send rebalancing suggestions back to the server"""
        if not suggestions:
            return
        
        logger.info(f"Sending {len(suggestions)} rebalancing suggestions")
        
        try:
            message = {
                "type": "suggestions",
                "client": "lightning_lens",
                "timestamp": datetime.now().isoformat(),
                "suggestions": suggestions
            }
            
            await websocket.send(json.dumps(message))
            logger.info("Suggestions sent successfully")
        except Exception as e:
            logger.error(f"Error sending suggestions: {e}")

async def main():
    """Main function to run the WebSocket client"""
    parser = argparse.ArgumentParser(description="LightningLens WebSocket Client")
    parser.add_argument("--uri", default="ws://localhost:6789", help="WebSocket URI")
    parser.add_argument("--model", help="Path to custom model (optional)")
    parser.add_argument("--scaler", help="Path to custom scaler (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Create data directories if they don't exist
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Initialize and run client
    client = LightningLensClient(
        websocket_uri=args.uri,
        model_path=args.model,
        scaler_path=args.scaler if args.scaler else (args.model.replace('model_', 'scaler_') if args.model else None)
    )
    
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Visualize how model predictions evolve over time

This script analyzes multiple prediction files to track how the model's
recommendations for specific channels change as it continues to learn.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def visualize_model_evolution():
    """Visualize how model predictions evolve over time"""
    print("LightningLens Model Evolution Visualizer")
    print("========================================")
    
    # Get all prediction files
    files = sorted(glob.glob("data/predictions/predictions_*.csv"))
    
    if not files:
        print("No prediction files found. Please generate predictions first.")
        return False
    
    print(f"Found {len(files)} prediction files")
    
    # Track a few specific channels (or automatically detect most active ones)
    channels_to_track = []
    
    # Read the first file to get channel IDs
    first_df = pd.read_csv(files[0])
    channel_counts = first_df['channel_id'].value_counts()
    
    # Get the top 5 most frequent channels
    channels_to_track = channel_counts.head(5).index.tolist()
    
    print(f"Tracking channels: {channels_to_track}")
    
    # Create a dataframe to store the evolution
    evolution_data = []
    
    for file in files:
        # Extract timestamp from filename
        timestamp = file.split("_")[1].split(".")[0]
        time_obj = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        formatted_time = time_obj.strftime("%m-%d %H:%M")
        
        df = pd.read_csv(file)
        
        # Get average values for each channel
        for channel in channels_to_track:
            channel_data = df[df['channel_id'] == channel]
            if not channel_data.empty:
                avg_optimal = channel_data['optimal_ratio'].mean()
                avg_current = channel_data['balance_ratio'].mean()
                avg_adjustment = channel_data['adjustment_needed'].mean()
                
                evolution_data.append({
                    'timestamp': formatted_time,
                    'raw_timestamp': timestamp,
                    'channel': channel,
                    'optimal_ratio': avg_optimal,
                    'current_ratio': avg_current,
                    'adjustment_needed': avg_adjustment
                })
    
    # Create the evolution dataframe
    evolution_df = pd.DataFrame(evolution_data)
    
    # Create output directory
    os.makedirs("data/visualizations", exist_ok=True)
    
    # Save the evolution data
    evolution_df.to_csv("data/visualizations/model_evolution.csv", index=False)
    
    # Plot the optimal ratio evolution
    plt.figure(figsize=(12, 6))
    
    for channel in channels_to_track:
        channel_data = evolution_df[evolution_df['channel'] == channel]
        if not channel_data.empty:
            plt.plot(
                channel_data['timestamp'], 
                channel_data['optimal_ratio'],
                marker='o',
                label=channel
            )
    
    plt.title('Evolution of Optimal Ratio Recommendations')
    plt.xlabel('Time')
    plt.ylabel('Optimal Ratio')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('data/visualizations/optimal_ratio_evolution.png')
    print(f"Saved visualization to data/visualizations/optimal_ratio_evolution.png")
    
    # Plot the adjustment needed evolution
    plt.figure(figsize=(12, 6))
    
    for channel in channels_to_track:
        channel_data = evolution_df[evolution_df['channel'] == channel]
        if not channel_data.empty:
            plt.plot(
                channel_data['timestamp'], 
                channel_data['adjustment_needed'],
                marker='o',
                label=channel
            )
    
    plt.title('Evolution of Adjustment Recommendations')
    plt.xlabel('Time')
    plt.ylabel('Adjustment Needed')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('data/visualizations/adjustment_evolution.png')
    print(f"Saved visualization to data/visualizations/adjustment_evolution.png")
    
    return True

if __name__ == "__main__":
    visualize_model_evolution() 